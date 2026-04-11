"""Document chunking — structure-aware text splitting."""


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 50) -> list[str]:
    """Split text into chunks at paragraph/sentence boundaries.

    Strategy: split on double newlines (paragraphs) first, then accumulate
    paragraphs into chunks up to chunk_size tokens (approximated as words).
    Falls back to word-level splitting for long paragraphs.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    if not paragraphs:
        # If the entire text is whitespace/empty after stripping, return nothing
        stripped = text.strip()
        if not stripped:
            return []
        paragraphs = [stripped]

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for para in paragraphs:
        para_len = len(para.split())

        # If single paragraph exceeds chunk_size, split it by sentences
        if para_len > chunk_size:
            # Flush current buffer first
            if current:
                chunks.append("\n\n".join(current))
                current = []
                current_len = 0
            # Split long paragraph into sentence-level chunks
            chunks.extend(_split_long_text(para, chunk_size, overlap))
            continue

        # Would adding this paragraph exceed the limit?
        if current_len + para_len > chunk_size and current:
            chunks.append("\n\n".join(current))
            # Overlap: keep last paragraph if it fits
            if overlap > 0 and current:
                last = current[-1]
                last_len = len(last.split())
                if last_len <= overlap:
                    current = [last]
                    current_len = last_len
                else:
                    current = []
                    current_len = 0
            else:
                current = []
                current_len = 0

        current.append(para)
        current_len += para_len

    if current:
        chunks.append("\n\n".join(current))

    return chunks


def _split_long_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split a long block of text by words with overlap."""
    words = text.split()
    chunks: list[str] = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start = end - overlap if overlap > 0 else end

    return chunks
