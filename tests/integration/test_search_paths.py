"""Integration: each retrieval channel returns the expected chunk; hybrid + rerank fire."""

import httpx
import pytest

from tests.integration.conftest import ingest

TARGET_DOC = (
    "BGE-M3 is a multilingual embedding model produced by BAAI. "
    "It outputs 1024-dimensional vectors and supports 8192-token context. "
    "Unlike most embedders, it can produce dense, sparse, and multi-vector outputs simultaneously.\n\n"
    "Snowflake arctic-embed is another option, also at 1024 dimensions but English-leaning."
)
DISTRACTOR_DOC = (
    "PostgreSQL is a relational database management system. "
    "It supports advanced indexing including HNSW for vector search via the pgvector extension."
)


async def _ingest_target_and_distractor(api: httpx.AsyncClient, created_docs: list[int]) -> tuple[int, int]:
    target = await ingest(api, "Embedding models reference", TARGET_DOC, metadata={"source": "test"})
    distractor = await ingest(api, "Postgres reference", DISTRACTOR_DOC, metadata={"source": "test"})
    created_docs.extend([target["id"], distractor["id"]])
    return target["id"], distractor["id"]


@pytest.mark.asyncio
async def test_semantic_search_returns_target(api: httpx.AsyncClient, created_docs: list[int]):
    """Pure semantic path returns the embedding-related doc for an embedding-related query."""
    target_id, _ = await _ingest_target_and_distractor(api, created_docs)
    r = await api.post(
        "/v1/search/semantic",
        json={"query": "What multilingual embedding model produces 1024-dim vectors", "top_k": 5, "min_score": 0.0},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["retrievers_used"] == ["semantic"]
    doc_ids = [r["document_id"] for r in body["results"]]
    assert target_id in doc_ids, f"target doc not in semantic results: {body}"
    assert all(r["sources"] == ["semantic"] for r in body["results"])


@pytest.mark.asyncio
async def test_keyword_search_returns_target(api: httpx.AsyncClient, created_docs: list[int]):
    """Pure keyword (BM25 via pg_search) path returns the doc with literal term match."""
    target_id, _ = await _ingest_target_and_distractor(api, created_docs)
    r = await api.post(
        "/v1/search/keyword",
        json={"query": "BGE-M3 multilingual", "top_k": 5},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["retrievers_used"] == ["keyword"]
    doc_ids = [r["document_id"] for r in body["results"]]
    assert target_id in doc_ids, f"target doc not in keyword results: {body}"


@pytest.mark.asyncio
async def test_hybrid_search_fires_both_channels(api: httpx.AsyncClient, created_docs: list[int]):
    """Hybrid search runs semantic + keyword, fuses with RRF, and reports both retrievers used."""
    target_id, _ = await _ingest_target_and_distractor(api, created_docs)
    r = await api.post(
        "/v1/search",
        json={"query": "BGE-M3 multilingual embedding", "top_k": 5, "rerank": False, "include_graph": False},
    )
    assert r.status_code == 200
    body = r.json()
    assert "semantic" in body["retrievers_used"]
    assert "keyword" in body["retrievers_used"]
    assert "rerank" not in body["retrievers_used"], "rerank disabled in this test"

    target_results = [r for r in body["results"] if r["document_id"] == target_id]
    assert target_results, f"target not in hybrid results: {body}"
    target_top = target_results[0]
    assert set(target_top["sources"]) >= {"semantic", "keyword"}, target_top


@pytest.mark.asyncio
async def test_rerank_reorders_results(api: httpx.AsyncClient, created_docs: list[int]):
    """With rerank enabled, retrievers_used includes rerank and order MAY differ from RRF-only."""
    target_id, _ = await _ingest_target_and_distractor(api, created_docs)
    common = {"query": "BGE-M3 multilingual embedding", "top_k": 5, "include_graph": False}

    r_no_rerank = await api.post("/v1/search", json={**common, "rerank": False})
    r_rerank = await api.post("/v1/search", json={**common, "rerank": True, "rerank_candidates": 10})
    assert r_no_rerank.status_code == 200
    assert r_rerank.status_code == 200

    body_rerank = r_rerank.json()
    assert "rerank" in body_rerank["retrievers_used"], body_rerank
    target_results = [r for r in body_rerank["results"] if r["document_id"] == target_id]
    assert target_results, f"target not in reranked results: {body_rerank}"

    rerank_score = target_results[0]["score"]
    assert 0.0 <= rerank_score <= 1.0, f"reranker score should be in [0,1], got {rerank_score}"
