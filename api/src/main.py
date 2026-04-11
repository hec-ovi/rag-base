"""FastAPI application entry point."""

import json
import logging
from contextlib import asynccontextmanager

import asyncpg
import httpx
from fastapi import FastAPI
from pgvector.asyncpg import register_vector

from src.config import settings
from src.routers import concepts, documents, embed, graph, health, rerank, relations, search

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

    yield

    # ── Shutdown ─────────────────────────────────────────────
    await app.state.db_pool.close()
    await app.state.embed_client.aclose()
    if app.state.rerank_client:
        await app.state.rerank_client.aclose()
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
app.include_router(concepts.router, prefix="/v1")
app.include_router(relations.router, prefix="/v1")
app.include_router(graph.router, prefix="/v1")
app.include_router(embed.router, prefix="/v1")
app.include_router(rerank.router, prefix="/v1")
