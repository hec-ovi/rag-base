"""TEI reranker client."""

import httpx


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
