"""Tests for chunking logic."""

from api.src.services.chunking import chunk_text, chunk_text_with_headers


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


# Phase 3c: chunk_text_with_headers


def test_headers_no_markdown_returns_empty_path():
    """Plain text with no markdown headers yields empty header_path on every chunk."""
    text = "Alpha bravo charlie.\n\nDelta echo foxtrot."
    chunks = chunk_text_with_headers(text, chunk_size=5, overlap=0)
    assert len(chunks) >= 1
    for c in chunks:
        assert c["header_path"] == ""
        assert "content" in c


def test_headers_single_h1_propagates():
    """A doc with one H1 attaches that title to every subsequent chunk's path."""
    text = "# Setup Guide\n\nFirst paragraph of guide.\n\nSecond paragraph of guide."
    chunks = chunk_text_with_headers(text, chunk_size=5, overlap=0)
    assert all(c["header_path"] == "Setup Guide" for c in chunks)


def test_headers_nested_path_grows_and_resets():
    """Heading stack tracks H1 > H2 > H3, resets deeper levels when a higher level reappears."""
    text = (
        "# Guide\n\nintro paragraph.\n\n"
        "## Setup\n\nsetup paragraph.\n\n"
        "### Linux\n\nlinux specific paragraph.\n\n"
        "## Usage\n\nusage paragraph."
    )
    chunks = chunk_text_with_headers(text, chunk_size=3, overlap=0)
    paths = [c["header_path"] for c in chunks]
    # Some chunk must have the deepest path active
    assert any(p == "Guide > Setup > Linux" for p in paths), paths
    # After H2 "Usage" appears, H3 "Linux" must NOT still be in any later path
    last_linux_idx = max((i for i, p in enumerate(paths) if "Linux" in p), default=-1)
    first_usage_idx = next((i for i, p in enumerate(paths) if "Usage" in p), -1)
    assert first_usage_idx > last_linux_idx
    assert all("Linux" not in p for p in paths[first_usage_idx:])


def test_headers_skip_levels_handled():
    """A doc that skips levels (H1 then H3) does not crash and produces a coherent path."""
    text = "# Top\n\ntop paragraph.\n\n### Deep\n\ndeep paragraph."
    # chunk_size=2 forces each ~2-word paragraph into its own chunk so we can
    # see the path active at each one
    chunks = chunk_text_with_headers(text, chunk_size=2, overlap=0)
    paths = [c["header_path"] for c in chunks]
    # The deep paragraph (or its containing chunk) should have Top and Deep in its path
    # (H2 slot stays None and is skipped in the " > " join)
    assert any(p == "Top > Deep" for p in paths), paths


def test_headers_chunk_starting_with_heading_includes_itself():
    """A chunk whose first paragraph IS a heading carries that heading in its path."""
    text = "## Section A\n\nsection a content here words."
    chunks = chunk_text_with_headers(text, chunk_size=20, overlap=0)
    # All chunks for this doc should have "Section A" in path
    assert all(c["header_path"] == "Section A" for c in chunks)


def test_headers_long_paragraph_inherits_active_path():
    """A long paragraph that gets word-split inherits the active heading path on all sub-chunks."""
    text = "# Big\n\n" + " ".join(["word"] * 200)
    chunks = chunk_text_with_headers(text, chunk_size=50, overlap=10)
    # The long paragraph yields multiple sub-chunks; every one should carry "Big"
    long_chunks = [c for c in chunks if c["content"].startswith("word")]
    assert len(long_chunks) > 1
    for c in long_chunks:
        assert c["header_path"] == "Big"


def test_headers_empty_text():
    """Empty input returns an empty list, no crash."""
    assert chunk_text_with_headers("", chunk_size=100, overlap=10) == []


def test_headers_hash_inside_text_not_treated_as_heading():
    """A hash that is not at the start of a paragraph (e.g. a # in code or sentence) does not update the stack."""
    text = "# Real Heading\n\nThe number is # 5 in this sentence.\n\nAnother paragraph."
    chunks = chunk_text_with_headers(text, chunk_size=20, overlap=0)
    for c in chunks:
        assert c["header_path"] == "Real Heading"
