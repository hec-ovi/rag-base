"""Hybrid search endpoint — semantic + keyword + graph → RRF → rerank."""

import asyncio

from fastapi import APIRouter, Request

from src.config import settings
from src.models.search import SearchRequest, SearchResponse, SearchResult
from src.services.embedding import embed_single
from src.services.fusion import reciprocal_rank_fusion
from src.services.keyword_search import search_keyword
from src.services.reranking import rerank
from src.services.vector_store import search_semantic

router = APIRouter(tags=["search"])


@router.post("/search", response_model=SearchResponse)
async def hybrid_search(body: SearchRequest, request: Request):
    """Hybrid search: parallel fan-out → RRF merge → optional rerank."""
    pool = request.app.state.db_pool
    embed_client = request.app.state.embed_client
    rerank_client = request.app.state.rerank_client

    candidates = body.rerank_candidates if body.rerank and rerank_client else body.top_k
    retrievers_used = []

    # 1. Embed the query
    query_vector = await embed_single(embed_client, body.query)

    # 2. Fan out: semantic + keyword (+ graph) in parallel
    tasks = [
        search_semantic(pool, query_vector, top_k=candidates, min_score=body.min_score),
        search_keyword(pool, body.query, top_k=candidates),
    ]

    # Graph expansion (if enabled and requested)
    graph_results = []
    if body.include_graph and request.app.state.graph_driver:
        # Simple entity extraction: look for capitalized multi-word terms
        # A production system would use NER here
        words = body.query.split()
        entity_candidates = [w for w in words if len(w) > 2]
        if entity_candidates:
            from src.services.graph_store import graph_search_expansion
            tasks.append(graph_search_expansion(request.app.state.graph_driver, entity_candidates))

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Collect successful results
    ranked_lists = []
    semantic_results = results[0] if not isinstance(results[0], Exception) else []
    keyword_results = results[1] if not isinstance(results[1], Exception) else []

    if semantic_results:
        ranked_lists.append(semantic_results)
        retrievers_used.append("semantic")
    if keyword_results:
        ranked_lists.append(keyword_results)
        retrievers_used.append("keyword")

    if len(results) > 2 and not isinstance(results[2], Exception) and results[2]:
        graph_results = results[2]
        retrievers_used.append("graph")
        # Graph results don't have chunk_ids directly — skip for RRF if no chunk mapping
        # TODO: link concepts to chunks for graph→RRF integration

    # 3. Merge with RRF
    if not ranked_lists:
        return SearchResponse(query=body.query, results=[], total=0, retrievers_used=[])

    merged = reciprocal_rank_fusion(*ranked_lists)

    # 4. Optional rerank
    if body.rerank and rerank_client and merged:
        texts_to_rerank = [item["content"] for item in merged[:candidates]]
        reranked = await rerank(rerank_client, body.query, texts_to_rerank)
        retrievers_used.append("rerank")

        # Rebuild results in reranked order
        reranked_results = []
        for r in reranked[:body.top_k]:
            item = merged[r["index"]]
            reranked_results.append(
                SearchResult(
                    chunk_id=item["chunk_id"],
                    document_id=item["document_id"],
                    document_title=item.get("document_title", ""),
                    content=item["content"],
                    score=r["score"],
                    sources=item.get("sources", []),
                )
            )
        return SearchResponse(
            query=body.query,
            results=reranked_results,
            total=len(reranked_results),
            retrievers_used=retrievers_used,
        )

    # Without rerank, return top_k from RRF
    final = [
        SearchResult(
            chunk_id=item["chunk_id"],
            document_id=item["document_id"],
            document_title=item.get("document_title", ""),
            content=item["content"],
            score=item["score"],
            sources=item.get("sources", []),
        )
        for item in merged[:body.top_k]
    ]

    return SearchResponse(
        query=body.query,
        results=final,
        total=len(final),
        retrievers_used=retrievers_used,
    )


@router.post("/search/semantic", response_model=SearchResponse)
async def semantic_search(body: SearchRequest, request: Request):
    """Vector-only search (no keyword, no graph, no rerank)."""
    pool = request.app.state.db_pool
    embed_client = request.app.state.embed_client

    query_vector = await embed_single(embed_client, body.query)
    results = await search_semantic(pool, query_vector, top_k=body.top_k, min_score=body.min_score)

    return SearchResponse(
        query=body.query,
        results=[
            SearchResult(
                chunk_id=r["chunk_id"],
                document_id=r["document_id"],
                document_title=r.get("document_title", ""),
                content=r["content"],
                score=r["score"],
                sources=["semantic"],
            )
            for r in results
        ],
        total=len(results),
        retrievers_used=["semantic"],
    )


@router.post("/search/keyword", response_model=SearchResponse)
async def keyword_search_endpoint(body: SearchRequest, request: Request):
    """Keyword-only search (no vector, no graph, no rerank)."""
    pool = request.app.state.db_pool

    results = await search_keyword(pool, body.query, top_k=body.top_k)

    return SearchResponse(
        query=body.query,
        results=[
            SearchResult(
                chunk_id=r["chunk_id"],
                document_id=r["document_id"],
                document_title=r.get("document_title", ""),
                content=r["content"],
                score=r["score"],
                sources=["keyword"],
            )
            for r in results
        ],
        total=len(results),
        retrievers_used=["keyword"],
    )
