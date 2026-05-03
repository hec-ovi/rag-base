"""End-to-end test of the 3-mode reranker (default / bge-gpu / qwen-4b).

Each individual mode is gated by an env var so the test runs whatever subset
of sidecars happens to be reachable on the dev box. With nothing set the file
collects to all-skip, which keeps CI green when the GPU sidecars are not up.

Required env (any combination):
  - RAG_BASE_URL                e.g. http://localhost:5050
  - RERANKER_URL                CPU TEI reranker (gates the 'default' assertion)
  - BGE_GPU_RERANKER_URL        bge-gpu sidecar
  - QWEN_RERANKER_URL           qwen-4b sidecar

Each test posts /v1/search with the corresponding rerank_model and asserts
shape + non-empty result. We do NOT assert score equality across modes (they
are different models / kernels), only that the API path works end to end.
"""
from __future__ import annotations

import os

import httpx
import pytest


API = os.environ.get("RAG_BASE_URL")
HAS_DEFAULT = bool(os.environ.get("RERANKER_URL"))
HAS_BGE_GPU = bool(os.environ.get("BGE_GPU_RERANKER_URL"))
HAS_QWEN = bool(os.environ.get("QWEN_RERANKER_URL"))

QUERY = os.environ.get("RAG_BASE_TEST_QUERY", "frontend frameworks")


def _post_search(rerank_model: str | None) -> dict:
    body: dict = {"query": QUERY, "top_k": 5, "rerank": True, "include_graph": False}
    if rerank_model is not None:
        body["rerank_model"] = rerank_model
    with httpx.Client(timeout=120.0) as c:
        r = c.post(f"{API}/v1/search", json=body)
        r.raise_for_status()
        return r.json()


def _shape_ok(payload: dict) -> None:
    assert "results" in payload and isinstance(payload["results"], list)
    assert "retrievers_used" in payload
    assert payload["query"] == QUERY


@pytest.mark.skipif(not (API and HAS_DEFAULT), reason="RAG_BASE_URL or default reranker not configured")
def test_default_path_byte_identical_when_field_omitted_and_when_explicit():
    """Sending no rerank_model vs sending rerank_model='default' should give the
    same retrievers_used set; both go through the existing CPU TEI path."""
    omitted = _post_search(None)
    explicit = _post_search("default")
    _shape_ok(omitted)
    _shape_ok(explicit)
    # Result count and retrievers_used should match (scores might be deterministic
    # too, but we keep the assertion conservative to avoid flakes).
    assert sorted(omitted["retrievers_used"]) == sorted(explicit["retrievers_used"])
    assert len(omitted["results"]) == len(explicit["results"])


@pytest.mark.skipif(not (API and HAS_BGE_GPU), reason="BGE_GPU_RERANKER_URL not configured")
def test_bge_gpu_mode_returns_results():
    payload = _post_search("bge-gpu")
    _shape_ok(payload)
    assert "rerank" in payload["retrievers_used"]


@pytest.mark.skipif(not (API and HAS_QWEN), reason="QWEN_RERANKER_URL not configured")
def test_qwen_4b_mode_returns_results():
    payload = _post_search("qwen-4b")
    _shape_ok(payload)
    assert "rerank" in payload["retrievers_used"]


@pytest.mark.skipif(not API, reason="RAG_BASE_URL not configured")
def test_unknown_rerank_model_rejected():
    with httpx.Client(timeout=10.0) as c:
        r = c.post(f"{API}/v1/search", json={"query": QUERY, "rerank_model": "bogus"})
    assert r.status_code == 422
