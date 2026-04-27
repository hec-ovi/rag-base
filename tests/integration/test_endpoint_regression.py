"""Phase 2 close gate: every Phase 1 endpoint exercised live with a representative
request. Per the phase-test-gates rule (full regression of every previous-phase
endpoint, do not trust prior tests still pass).

Endpoint inventory (matches /openapi.json from the Phase 1 close):
  GET    /health
  GET    /health/models
  POST   /v1/documents             (creates)
  GET    /v1/documents             (lists, paginated)
  GET    /v1/documents/{id}        (detail with chunks)
  DELETE /v1/documents/{id}        (cascade)
  POST   /v1/search                (hybrid + RRF + optional rerank + optional graph)
  POST   /v1/search/semantic       (vector only)
  POST   /v1/search/keyword        (BM25 via pg_search only)
  POST   /v1/embed                 (TEI passthrough)
  POST   /v1/rerank                (TEI passthrough)
  POST   /v1/concepts              (graph node CRUD)
  GET    /v1/concepts/{id}
  DELETE /v1/concepts/{id}
  POST   /v1/relations             (graph edge CRUD)
  GET    /v1/relations
  DELETE /v1/relations/{id}
  GET    /v1/graph/neighbors/{id}
  GET    /v1/graph/path/{from}/{to}
  GET    /v1/graph/communities
  GET    /v1/graph/stats

Each test exercises one endpoint with a representative request, asserts the
response shape and a key invariant. Goal: catch any regression introduced by
Phase 2 wiring at the API surface, not just at the test-of-the-day level.
"""

import httpx
import pytest

from tests.integration.conftest import ingest


# ----- /health -----

async def test_health_returns_ok(api: httpx.AsyncClient):
    r = await api.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    for k in ("postgres", "embedding", "reranker", "memgraph"):
        assert k in body


async def test_health_models_returns_loaded_ids(api: httpx.AsyncClient):
    r = await api.get("/health/models")
    assert r.status_code == 200
    body = r.json()
    assert "embedding" in body and "reranker" in body
    assert body["embedding"]["model_id"] == "BAAI/bge-m3"


# ----- /v1/documents lifecycle (create, list, get, delete) -----

async def test_documents_full_lifecycle(api: httpx.AsyncClient):
    # CREATE
    doc = await ingest(api, "Lifecycle test", "Short body content for lifecycle test.",
                       metadata={"source": "regression"})
    doc_id = doc["id"]
    assert doc["chunk_count"] >= 1

    # LIST (paginated)
    r = await api.get("/v1/documents", params={"limit": 50})
    assert r.status_code == 200
    docs = r.json()
    assert any(d["id"] == doc_id for d in docs)

    # GET DETAIL
    r = await api.get(f"/v1/documents/{doc_id}")
    assert r.status_code == 200
    detail = r.json()
    assert detail["id"] == doc_id
    assert detail["title"] == "Lifecycle test"
    assert len(detail["chunks"]) == doc["chunk_count"]

    # DELETE CASCADE
    r = await api.delete(f"/v1/documents/{doc_id}")
    assert r.status_code == 204
    r = await api.get(f"/v1/documents/{doc_id}")
    assert r.status_code == 404


# ----- /v1/search (3 flavors) -----

async def test_search_semantic_endpoint(api: httpx.AsyncClient, created_docs: list[int]):
    doc = await ingest(api, "Semantic regression", "BGE-M3 produces 1024 dim vectors.",
                       metadata={"source": "regression"})
    created_docs.append(doc["id"])
    r = await api.post(
        "/v1/search/semantic",
        json={"query": "embedding model 1024 dimensional", "top_k": 5, "min_score": 0.0},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["retrievers_used"] == ["semantic"]
    assert isinstance(body["results"], list)


async def test_search_keyword_endpoint(api: httpx.AsyncClient, created_docs: list[int]):
    doc = await ingest(api, "Keyword regression", "ParadeDB pg_search BM25.",
                       metadata={"source": "regression"})
    created_docs.append(doc["id"])
    r = await api.post(
        "/v1/search/keyword",
        json={"query": "ParadeDB pg_search", "top_k": 5},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["retrievers_used"] == ["keyword"]


async def test_search_hybrid_endpoint(api: httpx.AsyncClient, created_docs: list[int]):
    doc = await ingest(api, "Hybrid regression", "RRF k=60 fuses semantic and keyword.",
                       metadata={"source": "regression"})
    created_docs.append(doc["id"])
    r = await api.post(
        "/v1/search",
        json={"query": "RRF reciprocal rank fusion", "top_k": 5, "rerank": False, "include_graph": False},
    )
    assert r.status_code == 200
    body = r.json()
    assert "semantic" in body["retrievers_used"]


# ----- /v1/embed and /v1/rerank passthroughs -----

async def test_embed_passthrough(api: httpx.AsyncClient):
    r = await api.post("/v1/embed", json={"inputs": ["hello world", "second text"]})
    assert r.status_code == 200
    body = r.json()
    assert "embeddings" in body
    assert len(body["embeddings"]) == 2
    assert body["dimensions"] == 1024
    assert all(len(v) == 1024 for v in body["embeddings"])


async def test_rerank_passthrough(api: httpx.AsyncClient):
    r = await api.post(
        "/v1/rerank",
        json={"query": "fast database", "texts": ["postgres is fast", "the cat is fluffy"]},
    )
    assert r.status_code == 200
    body = r.json()
    assert "results" in body
    assert len(body["results"]) == 2
    # Postgres-relevant text should outscore the cat one
    by_index = {r["index"]: r["score"] for r in body["results"]}
    assert by_index[0] > by_index[1], by_index


# ----- /v1/concepts CRUD -----

async def test_concept_lifecycle(api: httpx.AsyncClient):
    r = await api.post(
        "/v1/concepts",
        json={"name": "RegressionConcept_X", "type": "TestType", "description": "test", "metadata": {}},
    )
    assert r.status_code == 201
    concept = r.json()
    cid = concept["id"]

    r = await api.get(f"/v1/concepts/{cid}")
    assert r.status_code == 200
    assert r.json()["name"] == "RegressionConcept_X"

    r = await api.delete(f"/v1/concepts/{cid}")
    assert r.status_code == 204


# ----- /v1/relations CRUD -----

async def test_relation_lifecycle(api: httpx.AsyncClient):
    s = await api.post(
        "/v1/concepts",
        json={"name": "RegSrc_X", "type": "TestType", "description": "", "metadata": {}},
    )
    t = await api.post(
        "/v1/concepts",
        json={"name": "RegTgt_X", "type": "TestType", "description": "", "metadata": {}},
    )
    assert s.status_code == 201 and t.status_code == 201

    r = await api.post(
        "/v1/relations",
        json={
            "source_name": "RegSrc_X",
            "target_name": "RegTgt_X",
            "relation_type": "TEST_RELATION",
            "metadata": {},
        },
    )
    assert r.status_code == 201
    relation = r.json()
    rid = relation["id"]

    r = await api.get("/v1/relations", params={"concept_name": "RegSrc_X"})
    assert r.status_code == 200
    assert any(rel["id"] == rid for rel in r.json())

    r = await api.delete(f"/v1/relations/{rid}")
    assert r.status_code == 204

    # cleanup created concepts
    await api.delete(f"/v1/concepts/{s.json()['id']}")
    await api.delete(f"/v1/concepts/{t.json()['id']}")


# ----- /v1/graph endpoints -----

async def test_graph_stats_returns_counts(api: httpx.AsyncClient):
    r = await api.get("/v1/graph/stats")
    assert r.status_code == 200
    body = r.json()
    assert "concepts" in body and "relations" in body
    assert isinstance(body["concepts"], int) and isinstance(body["relations"], int)


async def test_graph_neighbors(api: httpx.AsyncClient):
    s = await api.post(
        "/v1/concepts",
        json={"name": "RegN_A", "type": "T", "description": "", "metadata": {}},
    )
    t = await api.post(
        "/v1/concepts",
        json={"name": "RegN_B", "type": "T", "description": "", "metadata": {}},
    )
    sid, tid = s.json()["id"], t.json()["id"]
    rel = await api.post(
        "/v1/relations",
        json={"source_name": "RegN_A", "target_name": "RegN_B",
              "relation_type": "REGN", "metadata": {}},
    )
    try:
        r = await api.get(f"/v1/graph/neighbors/{sid}", params={"depth": 1})
        assert r.status_code == 200
        neighbors = r.json()
        assert any(n["id"] == tid for n in neighbors)
    finally:
        await api.delete(f"/v1/relations/{rel.json()['id']}")
        await api.delete(f"/v1/concepts/{sid}")
        await api.delete(f"/v1/concepts/{tid}")


async def test_graph_path_endpoint(api: httpx.AsyncClient):
    s = await api.post(
        "/v1/concepts",
        json={"name": "RegP_A", "type": "T", "description": "", "metadata": {}},
    )
    t = await api.post(
        "/v1/concepts",
        json={"name": "RegP_B", "type": "T", "description": "", "metadata": {}},
    )
    sid, tid = s.json()["id"], t.json()["id"]
    rel = await api.post(
        "/v1/relations",
        json={"source_name": "RegP_A", "target_name": "RegP_B",
              "relation_type": "REGP", "metadata": {}},
    )
    try:
        r = await api.get(f"/v1/graph/path/{sid}/{tid}")
        assert r.status_code == 200
        path = r.json()
        assert len(path) >= 2
        assert path[0]["id"] == sid
        assert path[-1]["id"] == tid
    finally:
        await api.delete(f"/v1/relations/{rel.json()['id']}")
        await api.delete(f"/v1/concepts/{sid}")
        await api.delete(f"/v1/concepts/{tid}")


async def test_graph_communities_returns_list(api: httpx.AsyncClient):
    r = await api.get("/v1/graph/communities")
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body, list)
    # Items should have id and community_id; we don't assert non-empty since it
    # depends on cluster algorithm + current graph state.
    for item in body[:5]:
        assert "id" in item
        assert "community_id" in item
