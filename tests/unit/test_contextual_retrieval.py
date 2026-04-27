"""Unit tests for the CR blurb generator (api/src/services/contextual_retrieval.py).

The generator is the only piece of CR-specific logic in the engine. It must:

- Return one blurb per chunk in input order.
- Return an empty list when called with an empty chunk list (no LLM calls).
- Survive per-chunk LLM failure: that chunk gets "" and the rest still complete.
- Pass document text + chunk text through the prompt template correctly.

We mock llm_complete so the tests stay fast, deterministic, and free of the
docker stack.
"""

import asyncio
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "api"))

from src.services.contextual_retrieval import generate_blurbs  # noqa: E402


@pytest.mark.asyncio
async def test_returns_one_blurb_per_chunk_in_order():
    """Each chunk slot is filled with exactly the LLM's response for that chunk."""
    seen_prompts: list[str] = []

    async def fake_llm(prompt: str, **kwargs) -> str:
        seen_prompts.append(prompt)
        # Blurb derives from the chunk so we can assert ordering downstream.
        if "chunk_a" in prompt:
            return "blurb for A"
        if "chunk_b" in prompt:
            return "blurb for B"
        if "chunk_c" in prompt:
            return "blurb for C"
        return "unknown"

    out = await generate_blurbs(
        fake_llm,
        document_text="full document text",
        chunk_texts=["chunk_a", "chunk_b", "chunk_c"],
    )
    assert out == ["blurb for A", "blurb for B", "blurb for C"]
    assert len(seen_prompts) == 3
    # Each prompt carried the document text (cache prefix) and its specific chunk.
    for p in seen_prompts:
        assert "full document text" in p


@pytest.mark.asyncio
async def test_empty_chunk_list_short_circuits_without_calling_llm():
    """Empty list in -> empty list out, zero LLM calls."""
    calls = 0

    async def fake_llm(prompt: str, **kwargs) -> str:
        nonlocal calls
        calls += 1
        return "should not be called"

    out = await generate_blurbs(fake_llm, document_text="anything", chunk_texts=[])
    assert out == []
    assert calls == 0


@pytest.mark.asyncio
async def test_per_chunk_failure_returns_empty_blurb_and_rest_succeed():
    """One chunk's LLM call raises -> that slot is "", others still get blurbs."""

    async def flaky_llm(prompt: str, **kwargs) -> str:
        if "FAIL_ME" in prompt:
            raise RuntimeError("LLM transient failure")
        if "chunk_a" in prompt:
            return "blurb for A"
        if "chunk_c" in prompt:
            return "blurb for C"
        return "unknown"

    out = await generate_blurbs(
        flaky_llm,
        document_text="doc",
        chunk_texts=["chunk_a", "FAIL_ME chunk_b", "chunk_c"],
    )
    assert out == ["blurb for A", "", "blurb for C"]


@pytest.mark.asyncio
async def test_blurb_is_stripped():
    """LLM may return trailing/leading whitespace; the service strips it."""

    async def whitespace_llm(prompt: str, **kwargs) -> str:
        return "   blurb with whitespace  \n\n"

    out = await generate_blurbs(whitespace_llm, document_text="d", chunk_texts=["x"])
    assert out == ["blurb with whitespace"]


@pytest.mark.asyncio
async def test_concise_flag_passed_through():
    """The blurb call must request concise mode so vLLM goes direct without thinking."""
    seen_kwargs: list[dict] = []

    async def recording_llm(prompt: str, **kwargs) -> str:
        seen_kwargs.append(dict(kwargs))
        return "ok"

    await generate_blurbs(recording_llm, document_text="d", chunk_texts=["x", "y"])
    assert all(kw.get("concise") is True for kw in seen_kwargs)


@pytest.mark.asyncio
async def test_concurrency_capped_by_semaphore():
    """max_concurrent=2 means at most 2 in-flight LLM calls at any time."""
    in_flight = 0
    peak = 0

    async def slow_llm(prompt: str, **kwargs) -> str:
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.05)
        in_flight -= 1
        return "blurb"

    await generate_blurbs(
        slow_llm,
        document_text="d",
        chunk_texts=[f"c{i}" for i in range(8)],
        max_concurrent=2,
    )
    assert peak <= 2


@pytest.mark.asyncio
async def test_total_failure_returns_all_empty_strings_no_raise():
    """If every chunk's call raises, the function still returns and the caller
    can decide what to do with an all-empty result."""

    async def always_fails(prompt: str, **kwargs) -> str:
        raise RuntimeError("LLM is down")

    out = await generate_blurbs(
        always_fails,
        document_text="doc",
        chunk_texts=["a", "b", "c"],
    )
    assert out == ["", "", ""]
