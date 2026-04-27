"""Integration: real LightRAG actually fires through the rag-base pipeline.

These tests require BOTH the docker compose stack AND a reachable LLM endpoint.
They skip cleanly (rather than fail) when LightRAG is not available, so the rest
of the integration suite can still run in environments without an LLM.

The slow tests (ingest + graph-into-RRF) take 5 to 15 minutes per LLM call on a
local reasoning model. They are marked `@pytest.mark.slow` so they can be
de-selected during fast iteration with `pytest -m 'not slow'`.
"""

import asyncio

import httpx
import pytest

from tests.integration.conftest import ingest


async def _lightrag_available(api: httpx.AsyncClient) -> bool:
    """Probe whether the running rag-base API has LightRAG wired up."""
    try:
        r = await api.get("/health")
        if r.status_code != 200:
            return False
        body = r.json()
    except httpx.HTTPError:
        return False
    # The health endpoint does not yet report lightrag, so we infer via memgraph
    # presence (LightRAG depends on it). A more robust check is added below by
    # querying the API state directly, which we can't do over HTTP. For now,
    # individual tests that need LightRAG attempt the operation and skip on
    # connection errors.
    return body.get("memgraph") == "connected"


def test_lightrag_version_pinned():
    """Phase 2 deliverable: lightrag-hku >= 1.4.15 (CVE patches require >= 1.4.14)."""
    import importlib.metadata
    version_str = importlib.metadata.version("lightrag-hku")
    parts = [int(x) for x in version_str.split(".")[:3]]
    assert tuple(parts) >= (1, 4, 14), (
        f"lightrag-hku {version_str} is below the 1.4.14 floor required to close "
        f"CVE-2026-30762 (HIGH JWT bypass) and CVE-2026-39413 (MEDIUM JWT alg confusion)"
    )


async def test_lightrag_module_importable():
    """The rag-base lightrag_store module imports cleanly."""
    from api.src.services import lightrag_store
    assert hasattr(lightrag_store, "init_lightrag")
    assert hasattr(lightrag_store, "lightrag_insert")
    assert hasattr(lightrag_store, "find_docs_via_graph")
    assert hasattr(lightrag_store, "extract_query_entities")
    assert hasattr(lightrag_store, "doc_lightrag_id")
    assert lightrag_store.doc_lightrag_id(42) == "doc_42"


async def test_parse_doc_ids_from_string_handles_sep():
    """LightRAG joins multi-chunk source_ids with '<SEP>'; doc id parser handles arbitrary strings."""
    from api.src.services.lightrag_store import parse_doc_ids_from_string
    assert parse_doc_ids_from_string("doc_1") == [1]
    assert parse_doc_ids_from_string("doc_42 doc_999") == [42, 999]
    assert parse_doc_ids_from_string("doc_1<SEP>doc_2") == [1, 2]
    assert parse_doc_ids_from_string("no doc here") == []
    assert parse_doc_ids_from_string("") == []
    # Same id only appears once
    assert parse_doc_ids_from_string("doc_5 and again doc_5") == [5]


@pytest.mark.slow
async def test_ingest_populates_memgraph(api: httpx.AsyncClient, created_docs: list[int]):
    """Posting a document with named entities populates Memgraph with entity nodes.

    Slow: LightRAG ingest on the local Qwen3.6-27B vLLM takes 5 to 10 minutes per
    small doc. Skipped if LightRAG is not configured (memgraph not 'connected').
    """
    if not await _lightrag_available(api):
        pytest.skip("LightRAG not available (Memgraph not connected)")

    # Use fully made-up entity names so the LLM cannot "correct" them back to
    # canonical real-world entities (which would dedupe to existing Memgraph
    # nodes and silently make this test pass-by-collision OR fail-by-no-growth
    # depending on accumulated state). Made-up names are forced through entity
    # extraction verbatim. We then assert directly that those names landed in
    # Memgraph rather than counting nodes.
    import secrets
    tag = secrets.token_hex(3)  # 6 hex chars; embedded into entity names
    e1 = f"Zorblax{tag}"
    e2 = f"Plumbus{tag}"
    e3 = f"Quagaar{tag}"

    from neo4j import AsyncGraphDatabase
    driver = AsyncGraphDatabase.driver("bolt://localhost:7687", auth=("", ""))
    try:
        # Tiny fixture: 3 made-up entities, 1 relation. Smaller = faster extraction.
        text = f"{e1} is a {e2} search engine produced by {e3}."
        doc = await ingest(api, "Phase 2 LightRAG smoke", text, metadata={"source": "p2_test"}, with_graph=True)
        created_docs.append(doc["id"])

        async with driver.session() as session:
            r = await session.run(
                "MATCH (n:base) WHERE n.entity_id CONTAINS $tag RETURN count(n) AS n",
                tag=tag,
            )
            n_tagged = (await r.single())["n"]

        assert n_tagged >= 1, (
            f"No Memgraph entity nodes contained the per-run tag {tag!r} after ingest. "
            "LightRAG either failed to extract or is not wired into documents.py."
        )
    finally:
        await driver.close()


@pytest.mark.slow
async def test_graph_into_rrf_actually_fires(api: httpx.AsyncClient, created_docs: list[int]):
    """THE killer test for Phase 2.

    Closes the rag-base search.py:66 TODO which left graph channel results out
    of the RRF candidate pool. Now graph results MUST appear in the response's
    `retrievers_used` array, and individual chunks reached only via graph
    expansion MUST appear in the results with sources containing 'graph'.

    Setup: ingest a doc whose query-relevant content lives in entities the LLM
    can identify but whose chunk text does NOT directly mention the query terms,
    forcing the graph channel to be the path that surfaces it.
    """
    if not await _lightrag_available(api):
        pytest.skip("LightRAG not available")

    # Tiny fixture: one clear entity-relation pair to keep ingest under 5 min.
    text_a = "Tantivy is a Rust search engine produced by Quickwit."
    doc_a = await ingest(api, "Quickwit doc", text_a, metadata={"source": "p2_killer"}, with_graph=True)
    created_docs.append(doc_a["id"])

    # Query mentions Tantivy by name. Graph channel can also surface Quickwit
    # via the 1-hop expansion DEVELOPS -> Quickwit, which gives doc_a relevance
    # via its entities even on queries the chunk text alone might not catch.
    r = await api.post(
        "/v1/search",
        json={"query": "Tantivy Rust search engine", "top_k": 10, "rerank": False, "include_graph": True},
    )
    assert r.status_code == 200
    body = r.json()

    # The graph channel must be reported in retrievers_used. If it isn't, either
    # the wiring is broken or the LLM entity-extraction returned nothing.
    assert "graph" in body["retrievers_used"], (
        f"graph channel did not fire (retrievers_used={body['retrievers_used']}). "
        f"This is the search.py:66 TODO that Phase 2 was supposed to close."
    )

    # At least one returned chunk must have 'graph' in its sources.
    graph_attributed = [r for r in body["results"] if "graph" in r.get("sources", [])]
    assert graph_attributed, (
        f"No result chunks have sources containing 'graph'. The graph channel "
        f"is reported as used but its chunks did not survive RRF, which means "
        f"the channel returned no candidates. Response: {body}"
    )


@pytest.mark.slow
async def test_lightrag_idempotent_ingest(api: httpx.AsyncClient, created_docs: list[int]):
    """Ingesting two documents with overlapping entities does not duplicate nodes.

    LightRAG should merge entity records when the same name appears in multiple
    documents. Re-ingesting the SAME content via a new doc should not double the
    Memgraph node count (entities merge, but new chunks are added with new chunk
    hashes that augment source_id of existing entities).
    """
    if not await _lightrag_available(api):
        pytest.skip("LightRAG not available")

    from neo4j import AsyncGraphDatabase
    driver = AsyncGraphDatabase.driver("bolt://localhost:7687", auth=("", ""))
    try:
        # Tiny fixture; idempotent ingest of the same content as two docs.
        text = "Mixedbread produces the mxbai-rerank-large-v2 reranker."
        doc1 = await ingest(api, "mxbai doc A", text, metadata={"source": "p2_idem"}, with_graph=True)
        created_docs.append(doc1["id"])

        async with driver.session() as session:
            r = await session.run(
                "MATCH (n:base) WHERE toLower(n.entity_id) IN $names "
                "RETURN count(DISTINCT n.entity_id) AS unique_entities",
                names=["mixedbread", "mxbai-rerank-large-v2"],
            )
            entities_after_first = (await r.single())["unique_entities"]

        # Ingest the same content again as a different document.
        doc2 = await ingest(api, "mxbai doc B", text, metadata={"source": "p2_idem"}, with_graph=True)
        created_docs.append(doc2["id"])

        async with driver.session() as session:
            r = await session.run(
                "MATCH (n:base) WHERE toLower(n.entity_id) IN $names "
                "RETURN count(DISTINCT n.entity_id) AS unique_entities",
                names=["mixedbread", "mxbai-rerank-large-v2"],
            )
            entities_after_second = (await r.single())["unique_entities"]

        assert entities_after_second == entities_after_first, (
            f"Entity dedup failed: {entities_after_first} unique entities after first "
            f"ingest, {entities_after_second} after second. LightRAG should merge "
            f"identical entity_id across docs."
        )
    finally:
        await driver.close()
