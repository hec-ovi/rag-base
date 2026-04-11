"""Tests for chunking logic."""

from api.src.services.chunking import chunk_text


def test_short_text_single_chunk():
    """Text shorter than chunk_size returns one chunk."""
    text = "This is a short text."
    chunks = chunk_text(text, chunk_size=100, overlap=10)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_paragraph_splitting():
    """Text with paragraphs splits on paragraph boundaries."""
    text = "First paragraph here.\n\nSecond paragraph here.\n\nThird paragraph here."
    chunks = chunk_text(text, chunk_size=5, overlap=0)
    assert len(chunks) >= 2


def test_overlap():
    """Chunks with overlap share content at boundaries."""
    # Build text with distinct paragraphs
    paras = [f"Paragraph {i} with some content words." for i in range(10)]
    text = "\n\n".join(paras)
    chunks_no_overlap = chunk_text(text, chunk_size=10, overlap=0)
    chunks_with_overlap = chunk_text(text, chunk_size=10, overlap=5)
    # With overlap, we should have at least as many chunks
    assert len(chunks_with_overlap) >= len(chunks_no_overlap)


def test_empty_text():
    """Empty text returns empty list or single empty-ish chunk."""
    chunks = chunk_text("", chunk_size=100, overlap=10)
    # Either empty list or list with empty string filtered
    assert len(chunks) <= 1


def test_long_paragraph_splits():
    """A single very long paragraph gets word-level split."""
    words = ["word"] * 200
    text = " ".join(words)
    chunks = chunk_text(text, chunk_size=50, overlap=10)
    assert len(chunks) > 1
    # Each chunk should be roughly chunk_size words or less
    for chunk in chunks:
        assert len(chunk.split()) <= 55  # small tolerance


def test_chunk_content_preserved():
    """All original content is present across chunks (no data loss)."""
    text = "Alpha bravo charlie.\n\nDelta echo foxtrot.\n\nGolf hotel india."
    chunks = chunk_text(text, chunk_size=5, overlap=0)
    combined = " ".join(chunks)
    for word in ["Alpha", "bravo", "charlie", "Delta", "echo", "foxtrot", "Golf", "hotel", "india"]:
        assert word in combined
