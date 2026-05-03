"""FastAPI application entry point."""

import json
import logging
import os
from contextlib import asynccontextmanager

import asyncpg
import httpx
from fastapi import FastAPI
from pgvector.asyncpg import register_vector

from src.config import settings
from src.routers import concepts, documents, embed, graph, graph_search, health, rerank, relations, search

logger = logging.getLogger("rag-base")


async def _init_connection(conn):
    """Initialize each connection in the pool: vector type, JSON codec, HNSW tuning."""
    await register_vector(conn)
    await conn.set_type_codec(
        "jsonb", encoder=json.dumps, decoder=json.loads, schema="pg_catalog"
    )
    ef_search = max(1, min(1000, int(settings.hnsw_ef_search)))
    await conn.execute(f"SET hnsw.ef_search = {ef_search}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ──────────────────────────────────────────────

    # 1. Postgres (required — fail hard if unavailable)
    app.state.db_pool = await asyncpg.create_pool(
        dsn=settings.database_url,
        min_size=2,
        max_size=10,
        init=_init_connection,
    )

    # 2. TEI embedding (required — fail hard if unavailable)
    app.state.embed_client = httpx.AsyncClient(
        base_url=settings.embedding_url, timeout=30.0
    )

    # 3. TEI reranker (optional — degrade gracefully)
    app.state.rerank_client = None
    if settings.rerank_enabled:
        client = httpx.AsyncClient(
            base_url=settings.reranker_url, timeout=60.0
        )
        try:
            resp = await client.get("/health")
            if resp.status_code == 200:
                app.state.rerank_client = client
                logger.info("Reranker connected: %s", settings.reranker_url)
            else:
                await client.aclose()
                logger.warning("Reranker unhealthy (status %d), disabled", resp.status_code)
        except Exception:
            await client.aclose()
            logger.warning("Reranker unreachable at %s, disabled", settings.reranker_url)

    # 3b. Optional GPU reranker sidecars (purely additive; default request path
    # still uses app.state.rerank_client). Each one is probed independently and
    # degrades to None on failure; the search router silently falls back to the
    # default reranker when a requested sidecar is unavailable.
    app.state.bge_gpu_rerank_client = None
    if settings.bge_gpu_reranker_url:
        c = httpx.AsyncClient(base_url=settings.bge_gpu_reranker_url, timeout=60.0)
        try:
            resp = await c.get("/health")
            if resp.status_code == 200:
                app.state.bge_gpu_rerank_client = c
                logger.info("BGE-GPU reranker connected: %s", settings.bge_gpu_reranker_url)
            else:
                await c.aclose()
                logger.warning("BGE-GPU reranker unhealthy (status %d), disabled", resp.status_code)
        except Exception:
            await c.aclose()
            logger.warning("BGE-GPU reranker unreachable at %s, disabled", settings.bge_gpu_reranker_url)

    app.state.qwen_rerank_client = None
    if settings.qwen_reranker_url:
        c = httpx.AsyncClient(base_url=settings.qwen_reranker_url, timeout=120.0)
        try:
            resp = await c.get("/health")
            if resp.status_code == 200:
                app.state.qwen_rerank_client = c
                logger.info("Qwen reranker connected: %s", settings.qwen_reranker_url)
            else:
                await c.aclose()
                logger.warning("Qwen reranker unhealthy (status %d), disabled", resp.status_code)
        except Exception:
            await c.aclose()
            logger.warning("Qwen reranker unreachable at %s, disabled", settings.qwen_reranker_url)

    app.state.qwen_8b_rerank_client = None
    if settings.qwen_8b_reranker_url:
        c = httpx.AsyncClient(base_url=settings.qwen_8b_reranker_url, timeout=180.0)
        try:
            resp = await c.get("/health")
            if resp.status_code == 200:
                app.state.qwen_8b_rerank_client = c
                logger.info("Qwen-8B reranker connected: %s", settings.qwen_8b_reranker_url)
            else:
                await c.aclose()
                logger.warning("Qwen-8B reranker unhealthy (status %d), disabled", resp.status_code)
        except Exception:
            await c.aclose()
            logger.warning("Qwen-8B reranker unreachable at %s, disabled", settings.qwen_8b_reranker_url)

    # 4. Memgraph (optional — degrade gracefully)
    app.state.graph_driver = None
    if settings.memgraph_enabled:
        try:
            from neo4j import AsyncGraphDatabase
            driver = AsyncGraphDatabase.driver(
                settings.memgraph_url, auth=("", "")
            )
            await driver.verify_connectivity()
            app.state.graph_driver = driver
            logger.info("Memgraph connected: %s", settings.memgraph_url)
            # Ensure indexes/constraints exist
            from src.services.graph_store import ensure_indexes
            try:
                await ensure_indexes(driver)
            except Exception:
                pass  # constraint may already exist
        except Exception:
            logger.warning("Memgraph unreachable at %s, disabled", settings.memgraph_url)

    # 5. LightRAG (optional — degrade gracefully)
    # Requires Memgraph (graph storage) and a reachable LLM. If either is missing,
    # the search graph channel is silently disabled; ingest still stores docs but
    # skips entity extraction.
    app.state.lightrag = None
    app.state.llm_complete = None
    if app.state.graph_driver and settings.llm_base_url:
        try:
            from src.services.llm_responses import make_llm_complete
            from src.services.lightrag_store import init_lightrag
            llm_complete = make_llm_complete(
                base_url=settings.llm_base_url,
                model=settings.llm_model,
                api_key=settings.llm_api_key,
                enable_thinking=settings.llm_enable_thinking,
            )
            # Ping the LLM with a tiny call so a misconfigured endpoint fails fast.
            try:
                await llm_complete("ping", concise=True)
                app.state.llm_complete = llm_complete
                logger.info("LLM endpoint reachable: %s", settings.llm_base_url)
            except Exception as e:
                logger.warning("LLM ping failed (%s); LightRAG disabled", e)

            if app.state.llm_complete:
                lightrag_dir = os.environ.get("LIGHTRAG_WORKING_DIR", "/app/lightrag_data")
                rag = await init_lightrag(
                    working_dir=lightrag_dir,
                    embed_client=app.state.embed_client,
                    llm_complete=app.state.llm_complete,
                    memgraph_url=settings.memgraph_url,
                )
                app.state.lightrag = rag
                logger.info("LightRAG ready (working_dir=%s)", lightrag_dir)
        except Exception as e:
            logger.warning("LightRAG init failed (%s); graph channel disabled", e)

    # 6. NER service for the Phase 5 graph-only endpoint (optional - degrade gracefully).
    # Lazy-loaded: the GLiNER model is fetched from disk on the first /v1/search/graph
    # call, not here. Construction is cheap; we only build the wrapper now so the
    # router can find it on app.state.
    app.state.ner_service = None
    try:
        from src.services.ner import NERService
        app.state.ner_service = NERService()
        logger.info("NER service constructed (model loads on first /v1/search/graph call)")
    except Exception as e:
        logger.warning("NER service unavailable (%s); /v1/search/graph will return 503", e)

    yield

    # ── Shutdown ─────────────────────────────────────────────
    if app.state.lightrag:
        try:
            await app.state.lightrag.finalize_storages()
        except Exception:
            pass
    await app.state.db_pool.close()
    await app.state.embed_client.aclose()
    if app.state.rerank_client:
        await app.state.rerank_client.aclose()
    if getattr(app.state, "bge_gpu_rerank_client", None):
        await app.state.bge_gpu_rerank_client.aclose()
    if getattr(app.state, "qwen_rerank_client", None):
        await app.state.qwen_rerank_client.aclose()
    if getattr(app.state, "qwen_8b_rerank_client", None):
        await app.state.qwen_8b_rerank_client.aclose()
    if app.state.graph_driver:
        await app.state.graph_driver.close()


app = FastAPI(
    title="rag-base",
    description="RAG backend — hybrid search with embeddings, reranking, and knowledge graph.",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(health.router)
app.include_router(documents.router, prefix="/v1")
app.include_router(search.router, prefix="/v1")
app.include_router(graph_search.router, prefix="/v1")
app.include_router(concepts.router, prefix="/v1")
app.include_router(relations.router, prefix="/v1")
app.include_router(graph.router, prefix="/v1")
app.include_router(embed.router, prefix="/v1")
app.include_router(rerank.router, prefix="/v1")
