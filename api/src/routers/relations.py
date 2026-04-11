"""Relation (graph edge) CRUD."""

from fastapi import APIRouter, HTTPException, Query, Request

from src.models.relation import RelationCreate, RelationOut

router = APIRouter(tags=["relations"])


@router.post("/relations", response_model=RelationOut, status_code=201)
async def create_relation(body: RelationCreate, request: Request):
    """Create a typed directed edge between two concepts."""
    driver = request.app.state.graph_driver
    if not driver:
        raise HTTPException(503, "Graph engine is disabled")
    if body.source_name == body.target_name:
        raise HTTPException(400, "Self-referencing relations are not allowed")

    from src.services.graph_store import create_relation as _create
    result = await _create(driver, body.source_name, body.target_name, body.relation_type, body.metadata)
    if not result:
        raise HTTPException(404, "One or both concepts not found")
    return result


@router.get("/relations", response_model=list[RelationOut])
async def get_relations(request: Request, concept_name: str = Query(..., min_length=1)):
    """Get all relations for a concept."""
    driver = request.app.state.graph_driver
    if not driver:
        raise HTTPException(503, "Graph engine is disabled")

    from src.services.graph_store import get_relations as _get
    return await _get(driver, concept_name)


@router.delete("/relations/{relation_id}", status_code=204)
async def delete_relation(relation_id: int, request: Request):
    """Delete a relation by ID."""
    driver = request.app.state.graph_driver
    if not driver:
        raise HTTPException(503, "Graph engine is disabled")

    from src.services.graph_store import delete_relation as _delete
    deleted = await _delete(driver, relation_id)
    if not deleted:
        raise HTTPException(404, "Relation not found")
