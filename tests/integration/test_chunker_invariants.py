"""Integration: chunker invariants on real ingestion.

The unit tests cover the chunker as a pure function. These tests assert the
ingest pipeline (chunker + storage) preserves character coverage and respects
the configured chunk size on real document content.
"""

import os

import httpx
import pytest

from tests.integration.conftest import ingest


CONFIGURED_CHUNK_SIZE_WORDS = int(os.environ.get("CHUNK_SIZE", "512"))
SOFT_UPPER_BOUND_WORDS = int(CONFIGURED_CHUNK_SIZE_WORDS * 1.2)


def _build_long_doc() -> str:
    """~2400-word doc with diverse paragraph lengths to exercise the chunker."""
    paragraphs = []
    for i in range(50):
        line_count = (i % 5) + 2
        body = " ".join(f"sentence {i}-{j} contains words about topic alpha and beta gamma." for j in range(line_count))
        paragraphs.append(f"Section {i}: {body}")
    return "\n\n".join(paragraphs)


@pytest.mark.asyncio
async def test_chunk_size_within_bound(api: httpx.AsyncClient, created_docs: list[int]):
    """Every chunk's token_count is <= chunk_size * 1.2 (soft upper).

    Soft upper accounts for the chunker keeping a paragraph atomic if it fits.
    """
    doc_text = _build_long_doc()
    doc = await ingest(api, "Chunk size bound test", doc_text)
    created_docs.append(doc["id"])

    detail = (await api.get(f"/v1/documents/{doc['id']}")).json()
    for chunk in detail["chunks"]:
        assert 1 <= chunk["token_count"] <= SOFT_UPPER_BOUND_WORDS, (
            f"chunk {chunk['chunk_index']} token_count={chunk['token_count']} "
            f"out of bound [1, {SOFT_UPPER_BOUND_WORDS}]"
        )


@pytest.mark.asyncio
async def test_chunker_char_coverage(api: httpx.AsyncClient, created_docs: list[int]):
    """Sum of chunk char lengths covers >=80% of source content.

    80% (not 95%) accounts for paragraph separator collapse and inter-chunk overlap
    which is added not removed; the chunker dedups paragraph boundaries.
    """
    doc_text = _build_long_doc()
    source_len_no_sep = sum(len(p) for p in doc_text.split("\n\n") if p.strip())

    doc = await ingest(api, "Chunk coverage test", doc_text)
    created_docs.append(doc["id"])

    detail = (await api.get(f"/v1/documents/{doc['id']}")).json()
    chunk_chars = sum(len(c["content"]) for c in detail["chunks"])

    coverage = chunk_chars / source_len_no_sep if source_len_no_sep else 0
    assert coverage >= 0.80, f"chunk char coverage {coverage:.2%} below 80% floor"


@pytest.mark.asyncio
async def test_short_doc_single_chunk(api: httpx.AsyncClient, created_docs: list[int]):
    """A doc with content shorter than chunk_size produces exactly 1 chunk preserving the text."""
    text = "A very short document with one paragraph only."
    doc = await ingest(api, "Short doc", text)
    created_docs.append(doc["id"])

    detail = (await api.get(f"/v1/documents/{doc['id']}")).json()
    assert detail["chunk_count"] == 1
    assert detail["chunks"][0]["content"] == text


@pytest.mark.asyncio
async def test_chunk_indices_are_sequential(api: httpx.AsyncClient, created_docs: list[int]):
    """chunk_index values form 0..N-1 with no gaps."""
    doc = await ingest(api, "Index sequence test", _build_long_doc())
    created_docs.append(doc["id"])
    detail = (await api.get(f"/v1/documents/{doc['id']}")).json()
    indices = [c["chunk_index"] for c in detail["chunks"]]
    assert indices == list(range(len(indices))), f"chunk indices not sequential: {indices}"
