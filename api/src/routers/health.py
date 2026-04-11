"""Health check endpoints."""

from fastapi import APIRouter, Request

router = APIRouter(tags=["health"])


@router.get("/health")
async def health(request: Request):
    """Overall service health — checks Postgres, TEI, and optionally Memgraph."""
    checks = {}

    # Postgres
    try:
        async with request.app.state.db_pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        checks["postgres"] = "connected"
    except Exception as e:
        checks["postgres"] = f"error: {e}"

    # TEI Embedding
    try:
        resp = await request.app.state.embed_client.get("/health")
        checks["embedding"] = "healthy" if resp.status_code == 200 else f"status {resp.status_code}"
    except Exception as e:
        checks["embedding"] = f"error: {e}"

    # TEI Reranker
    if request.app.state.rerank_client:
        try:
            resp = await request.app.state.rerank_client.get("/health")
            checks["reranker"] = "healthy" if resp.status_code == 200 else f"status {resp.status_code}"
        except Exception as e:
            checks["reranker"] = f"error: {e}"
    else:
        checks["reranker"] = "disabled"

    # Memgraph
    if request.app.state.graph_driver:
        try:
            await request.app.state.graph_driver.verify_connectivity()
            checks["memgraph"] = "connected"
        except Exception as e:
            checks["memgraph"] = f"error: {e}"
    else:
        checks["memgraph"] = "disabled"

    all_ok = all(
        v in ("connected", "healthy", "disabled")
        for v in checks.values()
    )

    return {"status": "ok" if all_ok else "degraded", **checks}


@router.get("/health/models")
async def health_models(request: Request):
    """Info about loaded embedding and reranker models."""
    result = {}

    try:
        resp = await request.app.state.embed_client.get("/info")
        result["embedding"] = resp.json() if resp.status_code == 200 else None
    except Exception:
        result["embedding"] = None

    if request.app.state.rerank_client:
        try:
            resp = await request.app.state.rerank_client.get("/info")
            result["reranker"] = resp.json() if resp.status_code == 200 else None
        except Exception:
            result["reranker"] = None
    else:
        result["reranker"] = "disabled"

    return result
