"""Embedding passthrough endpoint."""

from fastapi import APIRouter, Request

from src.models.search import EmbedRequest, EmbedResponse
from src.services.embedding import embed_texts, get_model_info

router = APIRouter(tags=["embed"])


@router.post("/embed", response_model=EmbedResponse)
async def embed(body: EmbedRequest, request: Request):
    """Embed text(s) via TEI. Returns vector(s)."""
    client = request.app.state.embed_client
    vectors = await embed_texts(client, body.inputs)
    info = await get_model_info(client)
    return EmbedResponse(
        embeddings=vectors,
        model=info.get("model_id", "unknown"),
        dimensions=len(vectors[0]) if vectors else 0,
    )
