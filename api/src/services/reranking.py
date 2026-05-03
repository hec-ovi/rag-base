"""TEI reranker client."""

import logging
from typing import Any

import httpx

from src.models.search import RerankModel

logger = logging.getLogger(__name__)


def select_rerank_client(state: Any, mode: RerankModel | None):
    """Pick the rerank client for the requested mode, with silent fallback to
    the default CPU TEI client when the requested sidecar is unavailable.

    Returns ``(client, fallback_used)``. ``client`` is ``None`` only when no
    reranker at all is wired (the router treats this as "skip rerank").
    """
    default = getattr(state, "rerank_client", None)
    if mode is None or mode == "default":
        return default, False
    if mode == "bge-gpu":
        c = getattr(state, "bge_gpu_rerank_client", None)
        if c is not None:
            return c, False
        if default is not None:
            logger.warning(
                "rerank_model=bge-gpu requested but unavailable; falling back to default"
            )
        return default, default is not None
    if mode == "qwen-4b":
        c = getattr(state, "qwen_rerank_client", None)
        if c is not None:
            return c, False
        if default is not None:
            logger.warning(
                "rerank_model=qwen-4b requested but unavailable; falling back to default"
            )
        return default, default is not None
    if mode == "qwen-8b":
        c = getattr(state, "qwen_8b_rerank_client", None)
        if c is not None:
            return c, False
        if default is not None:
            logger.warning(
                "rerank_model=qwen-8b requested but unavailable; falling back to default"
            )
        return default, default is not None
    return default, False


async def rerank(
    client: httpx.AsyncClient,
    query: str,
    texts: list[str],
    return_text: bool = False,
) -> list[dict]:
    """Rerank texts against a query via TEI /rerank endpoint.

    Returns list of {index, score, text?} sorted by score descending.
    """
    resp = await client.post(
        "/rerank",
        json={
            "query": query,
            "texts": texts,
            "raw_scores": False,
            "return_text": return_text,
        },
    )
    resp.raise_for_status()
    results = resp.json()
    return sorted(results, key=lambda r: r["score"], reverse=True)


async def get_model_info(client: httpx.AsyncClient) -> dict:
    """Get loaded reranker model info from TEI."""
    resp = await client.get("/info")
    resp.raise_for_status()
    return resp.json()
