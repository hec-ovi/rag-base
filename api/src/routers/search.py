"""Hybrid search endpoint — semantic + keyword + graph → RRF → rerank."""

import asyncio
import logging

from fastapi import APIRouter, Request

from src.models.search import SearchRequest, SearchResponse, SearchResult
from src.services.embedding import embed_single
from src.services.fusion import reciprocal_rank_fusion
from src.services.keyword_search import search_keyword
from src.services.lightrag_store import extract_query_entities, find_docs_via_graph
from src.services.reranking import rerank, select_rerank_client
from src.services.vector_store import search_semantic, search_semantic_in_docs

router = APIRouter(tags=["search"])
logger = logging.getLogger(__name__)


async def _graph_channel(
    rag,
    graph_driver,
    llm_complete,
    pool,
    query: str,
    query_vector: list[float],
    top_k: int,
    min_score: float,
) -> list[dict]:
    """Run the LightRAG-backed graph retrieval channel.

    Pipeline:
      1. LLM extracts entity names from the query (concise mode).
      2. Cypher into Memgraph: find entities matching those names plus 1-hop
         neighbors, return their LightRAG chunk hashes.
      3. Bridge those chunk hashes via LightRAG's text_chunks KV to our doc ids.
      4. Semantic search restricted to those doc ids; results tagged source="graph".

    Returns [] on any failure or if no graph hits found. The other retrieval
    channels are unaffected.
    """
    if rag is None or graph_driver is None or llm_complete is None:
        return []
    try:
        names = await extract_query_entities(llm_complete, query)
        if not names:
            return []
        doc_ids = await find_docs_via_graph(rag, graph_driver, names)
        if not doc_ids:
            return []
        return await search_semantic_in_docs(
            pool, query_vector, doc_ids, top_k=top_k, min_score=min_score, source_label="graph"
        )
    except Exception as e:
        logger.warning("Graph channel failed: %s", e)
        return []


@router.post("/search", response_model=SearchResponse)
async def hybrid_search(body: SearchRequest, request: Request):
    """Hybrid search: parallel fan-out → RRF merge → optional rerank."""
    pool = request.app.state.db_pool
    embed_client = request.app.state.embed_client
    rerank_client, _fallback_used = select_rerank_client(request.app.state, body.rerank_model)
    graph_driver = getattr(request.app.state, "graph_driver", None)
    lightrag = getattr(request.app.state, "lightrag", None)
    llm_complete = getattr(request.app.state, "llm_complete", None)

    candidates = body.rerank_candidates if body.rerank and rerank_client else body.top_k
    retrievers_used = []

    # 1. Embed the query
    query_vector = await embed_single(embed_client, body.query)

    # 2. Fan out: semantic + keyword + graph in parallel
    tasks: list = [
        search_semantic(pool, query_vector, top_k=candidates, min_score=body.min_score),
        search_keyword(pool, body.query, top_k=candidates),
    ]
    graph_enabled = bool(body.include_graph and lightrag and graph_driver and llm_complete)
    if graph_enabled:
        tasks.append(
            _graph_channel(
                lightrag, graph_driver, llm_complete, pool,
                body.query, query_vector, candidates, body.min_score,
            )
        )

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

    if graph_enabled:
        graph_results = results[2] if not isinstance(results[2], Exception) else []
        if graph_results:
            ranked_lists.append(graph_results)
            retrievers_used.append("graph")

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
