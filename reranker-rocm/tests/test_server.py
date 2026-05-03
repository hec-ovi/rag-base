"""Unit tests for the reranker-rocm sidecar.

The CrossEncoder is replaced with a deterministic fake so these tests run
without torch, ROCm, or any large model on disk.
"""
from __future__ import annotations

from typing import Iterable

import pytest
from fastapi.testclient import TestClient

from server import RerankRequest, _maybe_sigmoid, create_app


class FakeCrossEncoder:
    """Deterministic stand-in for sentence_transformers.CrossEncoder.

    Returns scores from a (query, text) -> float dict; defaults to len(text)/100
    for any pair not in the dict so ordering is stable but not all-equal.
    """

    def __init__(self, scores: dict | None = None) -> None:
        self._scores = scores or {}
        self.calls: list[tuple[list, int]] = []

    def predict(
        self,
        pairs: Iterable,
        batch_size: int = 8,
        convert_to_numpy: bool = True,
    ):
        materialised = list(pairs)
        self.calls.append((materialised, batch_size))
        out = []
        for q, t in materialised:
            out.append(self._scores.get((q, t), len(t) / 100.0))
        return out  # plain list is fine; server only iterates + casts to float


@pytest.fixture
def fake_model():
    return FakeCrossEncoder({
        ("frontend frameworks", "React is a JS library"):       0.95,
        ("frontend frameworks", "PostgreSQL is a database"):    0.05,
        ("frontend frameworks", "Vue is a frontend framework"): 0.92,
        ("frontend frameworks", "Tomatoes are fruit"):          0.01,
    })


@pytest.fixture
def client(fake_model):
    app = create_app(model=fake_model)
    with TestClient(app) as c:
        yield c


def test_request_model_validates():
    """The Pydantic body model accepts the TEI shape and rejects bad input."""
    r = RerankRequest(query="q", texts=["a", "b"])
    assert r.query == "q"
    assert r.texts == ["a", "b"]
    assert r.raw_scores is False
    assert r.return_text is False

    with pytest.raises(Exception):
        RerankRequest(query="q")  # missing texts

    with pytest.raises(Exception):
        RerankRequest(texts=["a"])  # missing query


def test_rerank_basic_ordering(client):
    body = {
        "query": "frontend frameworks",
        "texts": [
            "React is a JS library",        # high
            "PostgreSQL is a database",     # low
            "Vue is a frontend framework",  # high
            "Tomatoes are fruit",           # lowest
        ],
    }
    r = client.post("/rerank", json=body)
    assert r.status_code == 200
    data = r.json()

    assert len(data) == 4
    for d in data:
        assert set(d.keys()) == {"index", "score"}
        assert isinstance(d["index"], int)
        assert isinstance(d["score"], float)

    # Each input slot is accounted for exactly once.
    assert sorted(d["index"] for d in data) == [0, 1, 2, 3]

    # Sorted descending.
    scores = [d["score"] for d in data]
    assert scores == sorted(scores, reverse=True)

    # Top is the React or Vue line; bottom is the tomato line.
    assert data[0]["index"] in (0, 2)
    assert data[-1]["index"] == 3


def test_rerank_return_text_includes_originals(client):
    body = {
        "query": "frontend frameworks",
        "texts": ["React is a JS library", "PostgreSQL is a database"],
        "return_text": True,
    }
    r = client.post("/rerank", json=body)
    assert r.status_code == 200
    data = r.json()
    assert all("text" in d for d in data)
    assert {d["text"] for d in data} == {
        "React is a JS library",
        "PostgreSQL is a database",
    }
    # text matches the original index it points at
    for d in data:
        assert d["text"] == body["texts"][d["index"]]


def test_rerank_empty_texts_returns_empty_list(client):
    r = client.post("/rerank", json={"query": "anything", "texts": []})
    assert r.status_code == 200
    assert r.json() == []


def test_rerank_uses_default_batch_size(client, fake_model):
    client.post("/rerank", json={"query": "q", "texts": ["a", "b"]})
    assert fake_model.calls
    _, batch = fake_model.calls[-1]
    assert batch == 8


def test_rerank_passes_pairs_in_input_order(client, fake_model):
    texts = ["x", "y", "z"]
    client.post("/rerank", json={"query": "Q", "texts": texts})
    pairs, _ = fake_model.calls[-1]
    assert pairs == [("Q", "x"), ("Q", "y"), ("Q", "z")]


def test_rerank_validation_missing_query(client):
    r = client.post("/rerank", json={"texts": ["a"]})
    assert r.status_code == 422


def test_rerank_validation_missing_texts(client):
    r = client.post("/rerank", json={"query": "q"})
    assert r.status_code == 422


def test_health_ok_when_model_loaded(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"ok": True}


def test_health_503_when_loader_returns_none():
    """If the loader returns None (e.g. model file missing), /health = 503."""
    app = create_app(model_loader=lambda: None)
    with TestClient(app) as c:
        r = c.get("/health")
        assert r.status_code == 503


def test_rerank_503_when_loader_returns_none():
    app = create_app(model_loader=lambda: None)
    with TestClient(app) as c:
        r = c.post("/rerank", json={"query": "q", "texts": ["a"]})
        assert r.status_code == 503


# ---- _maybe_sigmoid -----------------------------------------------------------

def test_maybe_sigmoid_passes_through_in_unit_range():
    """BGE-style scores in [0, 1] are returned unchanged."""
    scores = [0.95, 0.5, 0.01, 0.001]
    assert _maybe_sigmoid(scores) == scores


def test_maybe_sigmoid_normalizes_logits():
    """Qwen-style logits get sigmoid-mapped into (0, 1) and keep rank order."""
    logits = [3.66, -0.7, -10.64, -11.44]
    out = _maybe_sigmoid(logits)
    assert all(0.0 <= s <= 1.0 for s in out)
    # rank order preserved
    assert sorted(range(len(out)), key=lambda i: -out[i]) == \
           sorted(range(len(logits)), key=lambda i: -logits[i])
    # sanity: large positive logit -> high prob
    assert out[0] > 0.95


def test_maybe_sigmoid_handles_empty():
    assert _maybe_sigmoid([]) == []


def test_maybe_sigmoid_normalizes_when_any_score_escapes_unit():
    """A single out-of-range score triggers normalization for the whole batch."""
    mixed = [0.5, 0.9, 5.0]
    out = _maybe_sigmoid(mixed)
    assert all(0.0 <= s <= 1.0 for s in out)
    # 5.0 -> ~0.993; original 0.9 maps to ~0.71; original 0.5 maps to ~0.62
    assert out[2] > out[1] > out[0]


# ---- rerank endpoint normalization integration -------------------------------

class _LogitFake:
    """Fake CrossEncoder that returns Qwen-style raw logits (out of [0,1])."""

    def __init__(self, scores_by_text: dict[str, float]):
        self._scores = scores_by_text

    def predict(self, pairs, batch_size: int = 8, convert_to_numpy: bool = True):
        return [self._scores[t] for _q, t in pairs]


def test_rerank_default_path_sigmoids_logits():
    fake = _LogitFake({"good": 4.0, "neutral": -0.5, "bad": -10.0})
    app = create_app(model=fake)
    with TestClient(app) as c:
        r = c.post("/rerank", json={"query": "x", "texts": ["good", "neutral", "bad"]})
        assert r.status_code == 200
        data = r.json()
    # All scores in [0, 1] after default normalization.
    assert all(0.0 <= d["score"] <= 1.0 for d in data)
    # Top is "good" (index 0)
    assert data[0]["index"] == 0
    assert data[0]["score"] > 0.95


def test_rerank_raw_scores_true_skips_sigmoid():
    fake = _LogitFake({"good": 4.0, "bad": -10.0})
    app = create_app(model=fake)
    with TestClient(app) as c:
        r = c.post("/rerank", json={"query": "x", "texts": ["good", "bad"], "raw_scores": True})
        assert r.status_code == 200
        data = r.json()
    # Raw logits preserved; rank still descending.
    scores = [d["score"] for d in data]
    assert max(scores) > 1.0
    assert min(scores) < 0.0
    assert scores == sorted(scores, reverse=True)
