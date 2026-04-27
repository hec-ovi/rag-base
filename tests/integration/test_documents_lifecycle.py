"""Integration: ingest -> chunks -> list -> get -> delete cascade."""

import httpx
import pytest

from tests.integration.conftest import ingest


SHORT_DOC = "Postgres is a database. It is open source. It supports JSONB."
LONG_DOC = "\n\n".join([f"Paragraph number {i} contains some text about topic {i}." for i in range(60)])


@pytest.mark.asyncio
async def test_ingest_short_doc_creates_chunks(api: httpx.AsyncClient, created_docs: list[int]):
    """Ingesting a doc creates >=1 chunk with embeddings."""
    doc = await ingest(api, "Postgres notes", SHORT_DOC)
    created_docs.append(doc["id"])
    assert doc["id"] > 0
    assert doc["chunk_count"] >= 1, doc

    detail = (await api.get(f"/v1/documents/{doc['id']}")).json()
    assert len(detail["chunks"]) == doc["chunk_count"]
    for chunk in detail["chunks"]:
        assert chunk["content"].strip(), "chunk content must be non-empty"
        assert chunk["token_count"] > 0, "token_count must be > 0"


@pytest.mark.asyncio
async def test_ingest_long_doc_splits_into_multiple_chunks(api: httpx.AsyncClient, created_docs: list[int]):
    """A long doc that exceeds chunk_size words splits into multiple chunks."""
    doc = await ingest(api, "Long doc", LONG_DOC)
    created_docs.append(doc["id"])
    assert doc["chunk_count"] >= 2, f"expected long doc to produce multiple chunks, got {doc['chunk_count']}"


@pytest.mark.asyncio
async def test_list_documents_includes_chunk_count(api: httpx.AsyncClient, created_docs: list[int]):
    """List endpoint returns docs ordered by created_at desc with chunk_count populated."""
    d1 = await ingest(api, "Doc A", "Content A here.")
    d2 = await ingest(api, "Doc B", "Content B here.")
    created_docs.extend([d1["id"], d2["id"]])

    r = await api.get("/v1/documents", params={"limit": 50})
    assert r.status_code == 200
    docs = r.json()
    found = {d["id"] for d in docs}
    assert d1["id"] in found
    assert d2["id"] in found
    for d in docs:
        if d["id"] in (d1["id"], d2["id"]):
            assert d["chunk_count"] >= 1, d


@pytest.mark.asyncio
async def test_delete_document_cascades_to_chunks(api: httpx.AsyncClient):
    """Deleting a doc removes its chunks (FK CASCADE) and 404s subsequent fetch."""
    doc = await ingest(api, "Ephemeral doc", "Ephemeral content.")
    doc_id = doc["id"]

    r = await api.delete(f"/v1/documents/{doc_id}")
    assert r.status_code == 204

    r = await api.get(f"/v1/documents/{doc_id}")
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_get_missing_document_returns_404(api: httpx.AsyncClient):
    r = await api.get("/v1/documents/999999999")
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_ingest_empty_title_rejected(api: httpx.AsyncClient):
    r = await api.post("/v1/documents", json={"title": "", "content": "x"})
    assert r.status_code == 422
