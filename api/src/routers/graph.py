"""Graph traversal and algorithm endpoints."""

from fastapi import APIRouter, HTTPException, Query, Request

router = APIRouter(tags=["graph"])


@router.get("/graph/neighbors/{concept_id}")
async def get_neighbors(concept_id: int, request: Request, depth: int = Query(2, ge=1, le=5)):
    """Multi-hop neighborhood traversal from a concept."""
    driver = request.app.state.graph_driver
    if not driver:
        raise HTTPException(503, "Graph engine is disabled")

    from src.services.graph_store import get_neighbors as _get
    return await _get(driver, concept_id, depth)


@router.get("/graph/path/{from_id}/{to_id}")
async def get_path(from_id: int, to_id: int, request: Request):
    """Shortest path between two concepts."""
    driver = request.app.state.graph_driver
    if not driver:
        raise HTTPException(503, "Graph engine is disabled")

    from src.services.graph_store import get_shortest_path as _get
    path = await _get(driver, from_id, to_id)
    if not path:
        raise HTTPException(404, "No path found")
    return path


@router.get("/graph/communities")
async def get_communities(request: Request):
    """Community detection via Louvain algorithm."""
    driver = request.app.state.graph_driver
    if not driver:
        raise HTTPException(503, "Graph engine is disabled")

    from src.services.graph_store import get_communities as _get
    return await _get(driver)


@router.get("/graph/stats")
async def get_stats(request: Request):
    """Graph statistics: node and edge counts."""
    driver = request.app.state.graph_driver
    if not driver:
        raise HTTPException(503, "Graph engine is disabled")

    from src.services.graph_store import get_stats as _get
    return await _get(driver)
