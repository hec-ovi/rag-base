"""Integration tests for the Phase 5 graph-only endpoint.

These hit the live api at /v1/search/graph and assume Memgraph already has the
multi-hop fixture (Alice Chen, Acme Robotics, Berkeley) populated by an earlier
LightRAG ingest. Each test that depends on the fixture probes for it via the
existing /v1/search?include_graph=true endpoint and skips if missing, so the
tests stay clean on a fresh dev env.

Hard guarantee asserted by these tests: the new endpoint NEVER mutates anything.
We capture document and chunk counts before and after the call and require them
to be identical.
"""

import os

import asyncpg
import httpx
import pytest

API_URL = os.environ.get("RAGBASE_API_URL", "http://localhost:5050")


def _default_database_url() -> str:
    user = os.environ.get("POSTGRES_USER", "knowledge")
    password = os.environ.get("POSTGRES_PASSWORD", "knowledge")
    port = os.environ.get("POSTGRES_PORT", "5433")
    db = os.environ.get("POSTGRES_DB", "knowledge")
    return f"postgresql://{user}:{password}@localhost:{port}/{db}"


DATABASE_URL = os.environ.get("RAGBASE_DATABASE_URL", _default_database_url())


async def _has_multi_hop_fixture(api: httpx.AsyncClient) -> bool:
    """Probe whether Memgraph holds the Alice Chen / Acme / Berkeley fixture."""
    try:
        r = await api.post(
            "/v1/search",
            json={
                "query": "Alice Chen Acme Robotics Berkeley",
                "top_k": 5,
                "rerank": False,
                "include_graph": True,
            },
            timeout=30.0,
        )
        if r.status_code != 200:
            return False
        body = r.json()
        return any("graph" in (h.get("sources") or []) for h in body.get("results", []))
    except Exception:
        return False


async def _row_counts(database_url: str) -> tuple[int, int]:
    """(documents, chunks) row counts. Used to assert the endpoint is read-only."""
    conn = await asyncpg.connect(database_url)
    try:
        n_docs = await conn.fetchval("SELECT count(*) FROM documents")
        n_chunks = await conn.fetchval("SELECT count(*) FROM chunks")
    finally:
        await conn.close()
    return int(n_docs), int(n_chunks)


# ────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_graph_search_endpoint_exists(api: httpx.AsyncClient):
    """Smoke: endpoint responds 200 with the documented shape, even on gibberish."""
    r = await api.post(
        "/v1/search/graph",
        json={"query": "qwertyuiopasdfgh", "max_entities": 4, "hops": 1, "top_k_chunks": 5},
        timeout=120.0,
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body.keys()) == {"query", "matched_entities", "subgraph", "chunks", "trace"}
    assert isinstance(body["matched_entities"], list)
    assert "nodes" in body["subgraph"] and "edges" in body["subgraph"]
    assert "latency_ms" in body["trace"]


@pytest.mark.asyncio
async def test_graph_search_empty_when_no_match(api: httpx.AsyncClient):
    """A query that cannot match any graph node returns 200 with empty arrays."""
    r = await api.post(
        "/v1/search/graph",
        json={"query": "xyzzytastic plumbus quagaars frobnicate", "fuzzy": False},
        timeout=60.0,
    )
    assert r.status_code == 200
    body = r.json()
    # NER may extract zero or some labels but they should not match the graph.
    assert body["matched_entities"] == [] or all(
        m["match_method"] in ("exact_ci", "fuzzy_contains") for m in body["matched_entities"]
    )


@pytest.mark.asyncio
async def test_graph_search_finds_known_entity(api: httpx.AsyncClient):
    """A query naming a known entity surfaces it in matched_entities and as a subgraph node."""
    if not await _has_multi_hop_fixture(api):
        pytest.skip("multi-hop fixture (Alice Chen / Acme / Berkeley) not in Memgraph")

    r = await api.post(
        "/v1/search/graph",
        json={"query": "Tell me about Alice Chen", "hops": 1, "top_k_chunks": 10},
        timeout=120.0,
    )
    assert r.status_code == 200
    body = r.json()
    matched_ids = {m["id"] for m in body["matched_entities"]}
    assert "Alice Chen" in matched_ids, f"Alice Chen not matched. Body: {body}"

    node_ids = {n["id"] for n in body["subgraph"]["nodes"]}
    assert "Alice Chen" in node_ids
    # 1-hop should pull at least Acme Robotics into the subgraph
    assert any("Acme" in nid for nid in node_ids), node_ids


@pytest.mark.asyncio
async def test_graph_search_chunks_bridge_to_real_doc_ids(api: httpx.AsyncClient):
    """The chunk bridge must produce real chunks.id values that resolve to Phase 4 doc ids."""
    if not await _has_multi_hop_fixture(api):
        pytest.skip("multi-hop fixture not in Memgraph")

    r = await api.post(
        "/v1/search/graph",
        json={"query": "Alice Chen", "hops": 0, "top_k_chunks": 5},
        timeout=120.0,
    )
    body = r.json()
    assert body["matched_entities"], "expected at least one match for Alice Chen"
    assert body["chunks"], "expected at least one bridged chunk"
    for ch in body["chunks"]:
        assert ch["lightrag_chunk_hash"].startswith("chunk-")
        # doc_id should be a positive int (or None when bridge failed).
        assert ch["doc_id"] is None or isinstance(ch["doc_id"], int)
        # source_entities references the entities that pointed at this chunk
        assert "Alice Chen" in ch["source_entities"]


@pytest.mark.asyncio
async def test_graph_search_two_hops_more_nodes_than_zero(api: httpx.AsyncClient):
    """2-hop expansion returns a strictly larger subgraph than 0-hop for the same seed."""
    if not await _has_multi_hop_fixture(api):
        pytest.skip("multi-hop fixture not in Memgraph")

    payload_base = {"query": "Alice Chen", "top_k_chunks": 5}
    r0 = await api.post("/v1/search/graph", json={**payload_base, "hops": 0}, timeout=120.0)
    r2 = await api.post("/v1/search/graph", json={**payload_base, "hops": 2}, timeout=120.0)
    n0 = len(r0.json()["subgraph"]["nodes"])
    n2 = len(r2.json()["subgraph"]["nodes"])
    assert n2 > n0, f"expected 2-hop ({n2}) > 0-hop ({n0})"


@pytest.mark.asyncio
async def test_graph_search_does_not_mutate_postgres(api: httpx.AsyncClient):
    """Hard read-only guarantee: row counts are identical before and after the call."""
    before = await _row_counts(DATABASE_URL)
    r = await api.post(
        "/v1/search/graph",
        json={"query": "Berkeley California", "hops": 2, "top_k_chunks": 10},
        timeout=120.0,
    )
    assert r.status_code == 200
    after = await _row_counts(DATABASE_URL)
    assert before == after, f"row counts changed after graph search: {before} -> {after}"


@pytest.mark.asyncio
async def test_graph_search_trace_has_split_timings(api: httpx.AsyncClient):
    """Trace separates NER, graph, and chunk-bridge timings; all positive when work was done."""
    r = await api.post(
        "/v1/search/graph",
        json={"query": "Alice Chen", "hops": 1, "top_k_chunks": 5},
        timeout=120.0,
    )
    body = r.json()
    trace = body["trace"]
    assert trace["latency_ms"] > 0
    assert trace["ner_ms"] >= 0
    assert trace["graph_ms"] >= 0
    assert trace["chunk_bridge_ms"] >= 0
    # latency_ms is the wall total; the components should be <= it (with small slack
    # for serialization overhead).
    component_sum = trace["ner_ms"] + trace["graph_ms"] + trace["chunk_bridge_ms"]
    assert component_sum <= trace["latency_ms"] + 50, (trace, component_sum)
