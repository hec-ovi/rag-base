"""Rerank passthrough endpoint."""

from fastapi import APIRouter, HTTPException, Request

from src.models.search import RerankRequest, RerankResponse, RerankResult
from src.services.reranking import get_model_info
from src.services.reranking import rerank as rerank_texts

router = APIRouter(tags=["rerank"])


@router.post("/rerank", response_model=RerankResponse)
async def rerank(body: RerankRequest, request: Request):
    """Rerank candidates against a query via TEI cross-encoder."""
    client = request.app.state.rerank_client
    if not client:
        raise HTTPException(503, "Reranker is disabled")

    results = await rerank_texts(client, body.query, body.texts, return_text=body.return_text)
    info = await get_model_info(client)

    return RerankResponse(
        results=[RerankResult(**r) for r in results],
        model=info.get("model_id", "unknown"),
    )
