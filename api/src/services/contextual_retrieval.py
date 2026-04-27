"""Anthropic Contextual Retrieval blurb generation at ingest time.

When `POST /v1/documents` is called with `contextual_retrieval: true`, the
ingest pipeline asks the configured LLM for a 50-100 token "situating"
blurb per chunk. The blurb is prepended to chunks.indexed_content (alongside
the existing title + header-path prefixes); chunks.content stays untouched.

Design notes:

- Prompt body lives in `api/prompts/contextual_retrieval.md`. Editing the prompt
  requires no code change, and no test re-run on the prompt itself; only the
  service code is tested. Restart the api container to pick up edits.

- vLLM has automatic prefix caching (`--enable-prefix-caching`, on by default
  in v0.20.0+). It hashes prefix tokens in 16-token blocks; subsequent requests
  whose prefix matches a cached block sequence skip prefill on that prefix
  entirely. Anthropic reports ~10x speedup on cached portions; cost drops from
  ~$10/MTok to ~$1/MTok.

- The cache only helps subsequent requests, NOT concurrent ones with the same
  prefix. If we fan out N concurrent calls all sharing a fresh document prefix,
  vLLM has to compute the prefix N times in parallel before any of them
  populate the cache, which jams the engine on long prefixes (a 5000-word doc
  is ~6500 prefill tokens; doing that 4-way concurrent on a single iGPU is
  catastrophic). To dodge this we issue the first call solo: it pays the full
  prefill cost but populates the cache. Calls 2..N then fan out in parallel,
  each one's prefix-cache lookup hits the warmed entry, and they pay only the
  cost of prefilling the per-chunk suffix (~1000 tokens) plus decoding the
  short blurb (~100 tokens). On a single-chunk document this degrades to a
  single solo call, which is just the natural shape.

- Concurrency is bounded by `max_concurrent` (default 1, fully serial). With
  the prefix cache warmed by the solo first call, subsequent calls are fast,
  and serializing them keeps the engine calm on a shared LLM (no risk of
  starving LightRAG ingest or query-time entity extraction). Bump above 1
  only when you have a dedicated LLM and the small wall-time win matters.

- Per-chunk failure -> empty blurb for that chunk only. The aggregate call
  never raises: callers see a list of strings the same length as the input,
  with possibly some empty entries. This matches the LightRAG-failure pattern
  in `routers/documents.py` where one failed augmentation does not block ingest.
"""

import asyncio
import logging
from typing import Awaitable, Callable

from src.services import prompts

logger = logging.getLogger(__name__)

LLMComplete = Callable[..., Awaitable[str]]


async def _one_blurb(llm_complete: LLMComplete, document_text: str, chunk_text: str) -> str:
    """Single CR LLM call. Returns blurb (stripped) or "" on failure."""
    try:
        blurb = await llm_complete(
            prompts.render(
                "contextual_retrieval",
                document=document_text,
                chunk=chunk_text,
            ),
            concise=True,
        )
        return (blurb or "").strip()
    except Exception as e:
        logger.warning("CR blurb generation failed for one chunk: %s", e)
        return ""


async def generate_blurbs(
    llm_complete: LLMComplete,
    document_text: str,
    chunk_texts: list[str],
    *,
    max_concurrent: int = 1,
) -> list[str]:
    """Return one CR blurb per chunk, in input order.

    First call runs solo so the document prefix populates vLLM's KV cache;
    remaining calls fan out at `max_concurrent` and benefit from the warm
    cache. On per-chunk failure that slot becomes "" and the rest proceed;
    the function as a whole does not raise.
    """
    if not chunk_texts:
        return []

    # Warm the cache with a solo first call.
    blurbs: list[str] = [""] * len(chunk_texts)
    blurbs[0] = await _one_blurb(llm_complete, document_text, chunk_texts[0])
    if len(chunk_texts) == 1:
        return blurbs

    # Subsequent calls run in parallel; the prefix cache is warm now.
    sem = asyncio.Semaphore(max_concurrent)

    async def one_capped(idx: int, chunk_text: str) -> tuple[int, str]:
        async with sem:
            return idx, await _one_blurb(llm_complete, document_text, chunk_text)

    rest = await asyncio.gather(
        *[one_capped(i, c) for i, c in enumerate(chunk_texts[1:], start=1)]
    )
    for idx, blurb in rest:
        blurbs[idx] = blurb
    return blurbs
