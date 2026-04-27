"""Integration: Contextual Chunk Headers (CCH) actually change retrieval ranking.

CCH (the rag-base built-in: title + metadata prefix prepended to chunk text before
embedding) is the cheap version of contextual retrieval. Phase 4 will add full
Anthropic Contextual Retrieval. This Phase 1 test asserts the existing mechanism
is end-to-end wired: the title prefix actually alters the embedding and therefore
the ranking.

Method: ingest the same chunk content under two different titles, then issue a
query that matches one of the titles. The doc whose title matches must rank above
the other, even though the chunk content is identical.
"""

import httpx
import pytest

from tests.integration.conftest import ingest


CHUNK_CONTENT = "the answer to that question is 42 according to recent analysis."


@pytest.mark.asyncio
async def test_cch_title_changes_ranking(api: httpx.AsyncClient, created_docs: list[int]):
    """Identical chunk content + different titles -> title-matching doc wins on title-targeted query."""
    matching_title = "Voyage AI rerank-2.5 reranker model documentation"
    decoy_title = "Postgres pgvector HNSW index configuration"

    doc_match = await ingest(api, matching_title, CHUNK_CONTENT, metadata={"source": "cch_test"})
    doc_decoy = await ingest(api, decoy_title, CHUNK_CONTENT, metadata={"source": "cch_test"})
    created_docs.extend([doc_match["id"], doc_decoy["id"]])

    r = await api.post(
        "/v1/search/semantic",
        json={"query": "Voyage rerank-2.5 reranker model", "top_k": 10, "min_score": 0.0},
    )
    assert r.status_code == 200
    results = r.json()["results"]

    cch_test_results = [r for r in results if r["document_id"] in (doc_match["id"], doc_decoy["id"])]
    assert len(cch_test_results) >= 2, f"expected both test docs in top-10: {results}"

    match_pos = next((i for i, r in enumerate(cch_test_results) if r["document_id"] == doc_match["id"]), None)
    decoy_pos = next((i for i, r in enumerate(cch_test_results) if r["document_id"] == doc_decoy["id"]), None)
    assert match_pos is not None and decoy_pos is not None
    assert match_pos < decoy_pos, (
        f"CCH not effective: matching-title doc at rank {match_pos}, decoy at {decoy_pos}. "
        f"If chunks were embedded without title prefix the two would tie. "
        f"Results: {cch_test_results}"
    )


@pytest.mark.asyncio
async def test_cch_metadata_changes_ranking(api: httpx.AsyncClient, created_docs: list[int]):
    """Identical chunk + identical title + different metadata -> metadata-matching doc wins."""
    title = "Reference notes"
    matching_meta = {"source": "lightrag-graphrag-comparison"}
    decoy_meta = {"source": "react-frontend-styling-guide"}

    doc_match = await ingest(api, title + " A", CHUNK_CONTENT, metadata=matching_meta)
    doc_decoy = await ingest(api, title + " B", CHUNK_CONTENT, metadata=decoy_meta)
    created_docs.extend([doc_match["id"], doc_decoy["id"]])

    r = await api.post(
        "/v1/search/semantic",
        json={"query": "lightrag graphrag comparison", "top_k": 10, "min_score": 0.0},
    )
    assert r.status_code == 200
    results = r.json()["results"]

    test_results = [r for r in results if r["document_id"] in (doc_match["id"], doc_decoy["id"])]
    assert len(test_results) >= 2, f"expected both test docs in top-10: {results}"

    match_pos = next((i for i, r in enumerate(test_results) if r["document_id"] == doc_match["id"]), None)
    decoy_pos = next((i for i, r in enumerate(test_results) if r["document_id"] == doc_decoy["id"]), None)
    assert match_pos < decoy_pos, (
        f"CCH metadata not effective: matching-metadata doc at rank {match_pos}, decoy at {decoy_pos}. "
        f"Results: {test_results}"
    )
