"""Tests for the rerank-client selector used by the search router.

Three modes plus the historical default. The selector silently falls back to
the default CPU TEI client when the requested sidecar is unavailable; this
matches the user-chosen fallback policy "fallback to the cpu one".
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "api"))

from src.services.reranking import select_rerank_client  # noqa: E402


@pytest.fixture
def state():
    """A bare app.state stand-in. Tests fill in the clients they need."""
    return SimpleNamespace(
        rerank_client=None,
        bge_gpu_rerank_client=None,
        qwen_rerank_client=None,
        qwen_8b_rerank_client=None,
    )


def test_default_path_unchanged_when_mode_none(state):
    state.rerank_client = "default-client"
    client, fallback = select_rerank_client(state, None)
    assert client == "default-client"
    assert fallback is False


def test_default_path_unchanged_when_mode_default(state):
    state.rerank_client = "default-client"
    client, fallback = select_rerank_client(state, "default")
    assert client == "default-client"
    assert fallback is False


def test_bge_gpu_returns_its_own_client(state):
    state.rerank_client = "default-client"
    state.bge_gpu_rerank_client = "bge-gpu-client"
    client, fallback = select_rerank_client(state, "bge-gpu")
    assert client == "bge-gpu-client"
    assert fallback is False


def test_qwen_returns_its_own_client(state):
    state.rerank_client = "default-client"
    state.qwen_rerank_client = "qwen-client"
    client, fallback = select_rerank_client(state, "qwen-4b")
    assert client == "qwen-client"
    assert fallback is False


def test_bge_gpu_falls_back_to_default_when_sidecar_unavailable(state, caplog):
    state.rerank_client = "default-client"
    state.bge_gpu_rerank_client = None
    with caplog.at_level("WARNING"):
        client, fallback = select_rerank_client(state, "bge-gpu")
    assert client == "default-client"
    assert fallback is True
    assert any("bge-gpu" in m and "fall" in m.lower() for m in caplog.messages)


def test_qwen_falls_back_to_default_when_sidecar_unavailable(state, caplog):
    state.rerank_client = "default-client"
    state.qwen_rerank_client = None
    with caplog.at_level("WARNING"):
        client, fallback = select_rerank_client(state, "qwen-4b")
    assert client == "default-client"
    assert fallback is True
    assert any("qwen-4b" in m and "fall" in m.lower() for m in caplog.messages)


def test_qwen_8b_returns_its_own_client(state):
    state.rerank_client = "default-client"
    state.qwen_8b_rerank_client = "qwen-8b-client"
    client, fallback = select_rerank_client(state, "qwen-8b")
    assert client == "qwen-8b-client"
    assert fallback is False


def test_qwen_8b_falls_back_to_default_when_sidecar_unavailable(state, caplog):
    state.rerank_client = "default-client"
    state.qwen_8b_rerank_client = None
    with caplog.at_level("WARNING"):
        client, fallback = select_rerank_client(state, "qwen-8b")
    assert client == "default-client"
    assert fallback is True
    assert any("qwen-8b" in m and "fall" in m.lower() for m in caplog.messages)


def test_returns_none_when_no_client_at_all(state):
    """When the default itself is missing AND a sidecar mode is requested,
    nothing to fall back to. Caller (router) must handle the None client."""
    client, fallback = select_rerank_client(state, "qwen-4b")
    assert client is None
    assert fallback is False


def test_default_none_when_mode_none(state):
    """No clients at all and no mode requested -> None, no fallback."""
    client, fallback = select_rerank_client(state, None)
    assert client is None
    assert fallback is False


def test_unknown_mode_returns_default(state):
    """A mode value not in the Literal (defensive path) returns the default
    client and does not raise."""
    state.rerank_client = "default-client"
    client, fallback = select_rerank_client(state, "made-up-mode")  # type: ignore[arg-type]
    assert client == "default-client"
    assert fallback is False


def test_request_model_accepts_rerank_model_field():
    """SearchRequest accepts the new rerank_model field while remaining
    byte-identical when the field is omitted."""
    from src.models.search import SearchRequest

    # Omitted -> None (default behavior preserved)
    r = SearchRequest(query="hello")
    assert r.rerank_model is None

    # Each Literal value parses
    for v in ("default", "bge-gpu", "qwen-4b", "qwen-8b"):
        assert SearchRequest(query="hello", rerank_model=v).rerank_model == v

    # Invalid values are rejected
    with pytest.raises(Exception):
        SearchRequest(query="hello", rerank_model="bogus")  # type: ignore[arg-type]
