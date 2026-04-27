"""Phase 5 graph-only retrieval endpoint.

Exposes `POST /v1/search/graph` as the fast-mode sibling of `/v1/search`.
Bypasses pgvector, BM25, RRF, and the cross-encoder reranker. Drives the
graph via GLiNER-based NER (no LLM at query time) and returns a structured
{matched_entities, subgraph, chunks, trace} payload.

Strict guarantees:
- Read-only against Memgraph, LightRAG KV, and Postgres.
- Empty 200 (matched_entities=[], chunks=[]) when NER finds nothing or no
  graph nodes match. Callers can decide whether to retry against /v1/search.
- 503 (and a clear error body) only when a hard infra dep is unreachable
  (Memgraph driver missing, NER service not initialized).
"""

import logging

from fastapi import APIRouter, HTTPException, Request

from src.models.graph_search import GraphSearchRequest, GraphSearchResponse
from src.services.graph_only_search import graph_only_search

router = APIRouter(tags=["search"])
logger = logging.getLogger(__name__)


@router.post("/search/graph", response_model=GraphSearchResponse)
async def graph_search(body: GraphSearchRequest, request: Request):
    """Graph-only retrieval (NER -> Memgraph -> chunks). No embedding, no LLM."""
    pool = request.app.state.db_pool
    graph_driver = getattr(request.app.state, "graph_driver", None)
    lightrag = getattr(request.app.state, "lightrag", None)
    ner_service = getattr(request.app.state, "ner_service", None)

    if graph_driver is None:
        raise HTTPException(
            status_code=503,
            detail="Graph driver unavailable: Memgraph not connected at startup.",
        )
    if ner_service is None:
        raise HTTPException(
            status_code=503,
            detail="NER service unavailable: gliner not loaded at startup.",
        )

    payload = await graph_only_search(
        ner_service=ner_service,
        graph_driver=graph_driver,
        lightrag=lightrag,
        pool=pool,
        query=body.query,
        max_entities=body.max_entities,
        hops=body.hops,
        ranking=body.ranking,
        top_k_chunks=body.top_k_chunks,
        fuzzy=body.fuzzy,
        ner_labels=body.ner_labels,
    )
    return payload
