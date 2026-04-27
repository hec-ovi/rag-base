"""LLM client for the OpenAI Responses API (used by LightRAG).

Designed for the local vLLM running Qwen3.6-27B-AWQ4. Four key behaviors:

1. Streaming-only. The vLLM Responses backend has bugs in non-streaming mode
   (per user). We always send `stream: true`, parse SSE deltas as they arrive,
   and reconstruct the final text + usage from the `response.completed` event.

2. Never sends max_output_tokens. Reasoning models burn tokens on the thinking
   chain; capping the output mid-thought leaves the response incomplete.

3. Reasoning is OFF by default via `chat_template_kwargs.enable_thinking: false`.
   Measurements on Qwen3.6-27B-AWQ4 showed `reasoning.effort=minimal` still let
   the model emit ~684 output tokens of preamble for a 3-word answer (33 s wall);
   `enable_thinking=false` cuts that to 3 tokens (0.7 s wall). Pass
   `enable_thinking=True` to opt back in.

4. When `concise=True` (signaled by LightRAG via `keyword_extraction=True`),
   prepends an explicit anti-deliberation system instruction. Belt-and-braces
   alongside `enable_thinking=false`.

Factory pattern: the public surface is `make_llm_complete(...)` and
`make_lightrag_llm_func(...)` which close over config values.
"""

import asyncio
import json
import logging
import time
from typing import Awaitable, Callable

import httpx

logger = logging.getLogger(__name__)

LLMComplete = Callable[..., Awaitable[str]]

CONCISE_SYSTEM_INSTRUCTION = (
    "Output ONLY the final answer. Do not think out loud. Do not deliberate. "
    "Do not list alternatives. Do not self-correct. Respond directly and concisely. "
    "Do not include any preamble, explanation, or commentary unless explicitly requested."
)


def _extract_message_text(response: dict) -> str:
    """Pull final message text from a /v1/responses payload, ignoring reasoning blocks.

    Used as a fallback when streaming yielded no `output_text.delta` events but the
    final `response.completed` payload still contains assembled output.
    """
    pieces: list[str] = []
    for item in response.get("output", []):
        if item.get("type") == "message":
            for content in item.get("content") or []:
                text = content.get("text") or ""
                if text:
                    pieces.append(text)
    return "".join(pieces).strip()


def _build_input(prompt: str, system_prompt: str | None, history: list[dict], concise: bool) -> list[dict]:
    """Convert prompt + history into Responses API `input` message list."""
    messages: list[dict] = []
    sys_parts: list[str] = []
    if concise:
        sys_parts.append(CONCISE_SYSTEM_INSTRUCTION)
    if system_prompt:
        sys_parts.append(system_prompt.strip())
    if sys_parts:
        messages.append({"role": "system", "content": "\n\n".join(sys_parts)})
    for msg in history or []:
        if "role" in msg and "content" in msg:
            messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": prompt})
    return messages


async def _consume_stream(response: httpx.Response) -> tuple[str, str, dict]:
    """Read SSE stream from a /v1/responses streaming call.

    Returns (assembled_text, final_status, final_usage). Defensive against servers
    that omit `event:` lines (in which case we read `type` from the JSON data),
    and falls back to extracting message text from the final payload if no deltas
    arrived.
    """
    text_pieces: list[str] = []
    final_status = "?"
    final_usage: dict = {}
    final_response: dict = {}
    current_event: str | None = None

    async for raw_line in response.aiter_lines():
        line = raw_line.rstrip("\r")
        if not line:
            current_event = None
            continue
        if line.startswith(":"):
            continue  # SSE comment / keepalive
        if line.startswith("event:"):
            current_event = line[len("event:"):].strip()
            continue
        if not line.startswith("data:"):
            continue
        data_str = line[len("data:"):].strip()
        if not data_str or data_str == "[DONE]":
            continue
        try:
            event_data = json.loads(data_str)
        except json.JSONDecodeError:
            logger.debug("SSE non-JSON data line ignored: %r", data_str[:120])
            continue

        ev = current_event or event_data.get("type", "")

        if ev == "response.output_text.delta":
            delta = event_data.get("delta", "")
            if isinstance(delta, str) and delta:
                text_pieces.append(delta)
        elif ev == "response.completed":
            resp = event_data.get("response", {}) or {}
            final_response = resp
            final_status = resp.get("status", "?")
            final_usage = resp.get("usage", {}) or {}
        elif ev in ("response.failed", "response.incomplete", "error"):
            err = event_data.get("error") or (event_data.get("response") or {}).get("error")
            raise RuntimeError(f"LLM stream returned {ev}: {err}")

    text = "".join(text_pieces).strip()
    if not text and final_response:
        text = _extract_message_text(final_response)
    return text, final_status, final_usage


def make_llm_complete(
    base_url: str,
    model: str,
    api_key: str = "",
    enable_thinking: bool = False,
    timeout: float = 600.0,
    max_retries: int = 1,
    retry_backoff: float = 2.0,
) -> LLMComplete:
    """Build a closure that sends one streaming /v1/responses call and returns assistant text.

    `enable_thinking` defaults to False, which sends
    `chat_template_kwargs: {enable_thinking: false}` to the vLLM Responses
    endpoint. On Qwen3.6-27B-AWQ4 this cuts a 3-word answer from ~684 output
    tokens / 33 s to 3 tokens / 0.7 s. Pass True only if a future test wants
    to compare with thinking on.

    `timeout` is per-attempt wall time. Streaming keeps the connection active so this
    is a true ceiling, not a between-byte timeout. 600s comfortably covers entity
    extraction over multi-paragraph chunks.

    `max_retries` defaults to 1 (up to 2 total attempts) since hangs typically clear
    on retry. Backoff doubles starting at `retry_backoff` seconds.
    """

    headers: dict[str, str] = {
        "content-type": "application/json",
        "accept": "text/event-stream",
    }
    if api_key:
        headers["authorization"] = f"Bearer {api_key}"

    async def llm_complete(
        prompt: str,
        system_prompt: str | None = None,
        history_messages: list[dict] | None = None,
        *,
        concise: bool = False,
        **_unused,
    ) -> str:
        payload: dict = {
            "model": model,
            "input": _build_input(prompt, system_prompt, history_messages or [], concise),
            "stream": True,
            "chat_template_kwargs": {"enable_thinking": enable_thinking},
        }

        last_err: Exception | None = None
        for attempt in range(max_retries + 1):
            t0 = time.perf_counter()
            try:
                async with httpx.AsyncClient(base_url=base_url, timeout=timeout) as client:
                    async with client.stream(
                        "POST", "/v1/responses", json=payload, headers=headers
                    ) as r:
                        r.raise_for_status()
                        text, final_status, final_usage = await _consume_stream(r)
                dt = time.perf_counter() - t0
                in_tok = final_usage.get("input_tokens", 0)
                out_tok = final_usage.get("output_tokens", 0)
                logger.info(
                    "LLM stream %s: %.1fs (concise=%s, in=%d, out=%d, chars=%d)",
                    final_status, dt, concise, in_tok, out_tok, len(text),
                )
                if final_status not in ("completed", "?"):
                    logger.warning("LLM stream non-completed status: %s usage=%s", final_status, final_usage)
                return text
            except (httpx.TimeoutException, httpx.HTTPError, RuntimeError) as e:
                dt = time.perf_counter() - t0
                logger.warning(
                    "LLM stream failed after %.1fs (attempt %d/%d, concise=%s): %s",
                    dt, attempt + 1, max_retries + 1, concise, e,
                )
                last_err = e
                if attempt < max_retries:
                    await asyncio.sleep(retry_backoff * (2 ** attempt))
                else:
                    raise
        raise last_err if last_err else RuntimeError("llm_complete exhausted retries")

    return llm_complete
