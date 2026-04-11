"""TEI embedding client."""

import httpx


async def embed_texts(client: httpx.AsyncClient, texts: list[str]) -> list[list[float]]:
    """Embed a list of texts via TEI /embed endpoint."""
    resp = await client.post("/embed", json={"inputs": texts})
    resp.raise_for_status()
    return resp.json()


async def embed_single(client: httpx.AsyncClient, text: str) -> list[float]:
    """Embed a single text, return one vector."""
    vectors = await embed_texts(client, [text])
    return vectors[0]


async def get_model_info(client: httpx.AsyncClient) -> dict:
    """Get loaded model info from TEI."""
    resp = await client.get("/info")
    resp.raise_for_status()
    return resp.json()
