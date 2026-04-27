"""Phase 3c: markdown header-path prefix on chunks at ingest.

These tests exercise two contracts:

1. Storage: indexed_content is "[<title | meta>] [<header_path>] <chunk content>"
   and the raw chunks.content stays unchanged.

2. Retrieval: BM25 (pg_search) ranks chunks by indexed_content, so chunks whose
   raw content does NOT mention a heading word still match a query for that
   word, because the breadcrumb in indexed_content carries it.

We connect to Postgres directly to read indexed_content (it is intentionally
not exposed via the API; it is an index-only field).
"""

import os

import asyncpg
import httpx
import pytest

from tests.integration.conftest import ingest

def _default_database_url() -> str:
    user = os.environ.get("POSTGRES_USER", "knowledge")
    password = os.environ.get("POSTGRES_PASSWORD", "knowledge")
    port = os.environ.get("POSTGRES_PORT", "5433")
    db = os.environ.get("POSTGRES_DB", "knowledge")
    return f"postgresql://{user}:{password}@localhost:{port}/{db}"


DATABASE_URL = os.environ.get("RAGBASE_DATABASE_URL", _default_database_url())


def _filler(phrase: str, repeats: int = 100) -> str:
    """Build a filler paragraph long enough to force a chunk split at 512 words."""
    return (phrase + " ") * repeats


@pytest.mark.asyncio
async def test_indexed_content_carries_title_and_header_path(
    api: httpx.AsyncClient, created_docs: list[int]
):
    """A chunk's indexed_content prefixes [title|meta] [header_path] <raw>; raw content is unchanged."""
    title = "Phase3c smoke title"
    content = (
        "# Setup\n\n"
        "## Linux\n\n"
        "Install the package via apt for Linux systems."
    )
    res = await ingest(api, title, content, metadata={"source": "phase3c-test"})
    created_docs.append(res["id"])

    conn = await asyncpg.connect(DATABASE_URL)
    try:
        rows = await conn.fetch(
            "SELECT chunk_index, content, indexed_content FROM chunks WHERE document_id = $1 ORDER BY chunk_index",
            res["id"],
        )
    finally:
        await conn.close()

    assert len(rows) >= 1, "no chunks created for ingested doc"
    for row in rows:
        # raw content stays untouched
        assert row["content"] in content
        # indexed_content carries the title-meta bracket
        assert row["indexed_content"].startswith(f"[{title} | source: phase3c-test]"), (
            f"indexed_content missing title-meta prefix: {row['indexed_content'][:120]!r}"
        )
        # and the header path bracket (Setup is the H1 active at the first paragraph)
        assert "[Setup]" in row["indexed_content"], (
            f"indexed_content missing [Setup] breadcrumb: {row['indexed_content'][:200]!r}"
        )
        # and the raw chunk content sits at the tail
        assert row["indexed_content"].rstrip().endswith(row["content"].rstrip())


@pytest.mark.asyncio
async def test_indexed_content_omits_breadcrumb_when_no_headers(
    api: httpx.AsyncClient, created_docs: list[int]
):
    """A doc with no markdown headers gets [title|meta] only; no empty breadcrumb bracket."""
    title = "Plain doc no headers"
    content = (
        "First paragraph of plain prose.\n\n"
        "Second paragraph of plain prose."
    )
    res = await ingest(api, title, content)
    created_docs.append(res["id"])

    conn = await asyncpg.connect(DATABASE_URL)
    try:
        rows = await conn.fetch(
            "SELECT indexed_content FROM chunks WHERE document_id = $1",
            res["id"],
        )
    finally:
        await conn.close()

    assert rows
    for row in rows:
        # Title-meta prefix present
        assert row["indexed_content"].startswith(f"[{title}] "), row["indexed_content"][:120]
        # No empty breadcrumb bracket like "[] "
        assert "[] " not in row["indexed_content"]


@pytest.mark.asyncio
async def test_breadcrumb_makes_chunk_findable_by_section_word(
    api: httpx.AsyncClient, created_docs: list[int]
):
    """BM25 finds a chunk by a section-name word that lives only in the breadcrumb.

    Builds a long doc whose Python and Java sections each overflow chunk_size,
    so the chunker emits multiple chunks per section. The trailing chunks of a
    section contain ONLY the filler text (the heading was already absorbed
    into the first chunk of the section). Without the breadcrumb a query for
    'Python' could not match those trailing chunks; with the breadcrumb their
    indexed_content carries '[... > Python]' so BM25 still sees the term.
    """
    py_filler = _filler("the widget process orchestrates synchronized stages reliably")
    java_filler = _filler("the gadget routine handles aligned segments precisely")
    content = (
        "# Programming\n\n"
        "## Python\n\n"
        f"{py_filler}\n\n"
        "## Java\n\n"
        f"{java_filler}"
    )
    res = await ingest(api, "Long programming guide", content)
    doc_id = res["id"]
    created_docs.append(doc_id)

    # Sanity: the doc must have produced multiple chunks (otherwise no split).
    detail = (await api.get(f"/v1/documents/{doc_id}")).json()
    assert detail["chunk_count"] >= 3, (
        f"expected multi-chunk split, got {detail['chunk_count']}"
    )

    # Find chunks whose RAW content does not mention "Python" but whose indexed_content does
    # (i.e. the breadcrumb is the only place the word lives).
    conn = await asyncpg.connect(DATABASE_URL)
    try:
        rows = await conn.fetch(
            """
            SELECT id, chunk_index, content, indexed_content
            FROM chunks
            WHERE document_id = $1
            ORDER BY chunk_index
            """,
            doc_id,
        )
    finally:
        await conn.close()

    breadcrumb_only = [
        r for r in rows
        if "Python" not in r["content"] and "Python" in r["indexed_content"]
    ]
    assert breadcrumb_only, (
        "expected at least one chunk where 'Python' lives only in the breadcrumb; "
        "if this fails the chunker did not split mid-section"
    )

    # Now BM25 search via the keyword endpoint and verify at least one of those
    # breadcrumb-only chunks is returned.
    r = await api.post(
        "/v1/search/keyword",
        json={"query": "widget Python", "top_k": 20},
    )
    r.raise_for_status()
    body = r.json()
    returned_ids = {hit["chunk_id"] for hit in body["results"]}
    breadcrumb_only_ids = {r["id"] for r in breadcrumb_only}
    assert returned_ids & breadcrumb_only_ids, (
        f"BM25 did not surface any breadcrumb-only Python chunk. "
        f"returned={returned_ids}, breadcrumb_only={breadcrumb_only_ids}"
    )
