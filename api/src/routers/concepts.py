"""Concept (graph node) CRUD."""

from fastapi import APIRouter, HTTPException, Request

from src.models.concept import ConceptCreate, ConceptDetail, ConceptOut

router = APIRouter(tags=["concepts"])


@router.post("/concepts", response_model=ConceptOut, status_code=201)
async def create_concept(body: ConceptCreate, request: Request):
    """Create or update a concept node in the knowledge graph (upserts by name)."""
    driver = request.app.state.graph_driver
    if not driver:
        raise HTTPException(503, "Graph engine is disabled")

    from src.services.graph_store import create_concept as _create
    result = await _create(driver, body.name, body.type, body.description, body.metadata)
    return result


@router.get("/concepts/{concept_id}", response_model=ConceptDetail)
async def get_concept(concept_id: int, request: Request):
    """Get a concept with its relations."""
    driver = request.app.state.graph_driver
    if not driver:
        raise HTTPException(503, "Graph engine is disabled")

    from src.services.graph_store import get_concept as _get
    result = await _get(driver, concept_id)
    if not result:
        raise HTTPException(404, "Concept not found")
    return result


@router.delete("/concepts/{concept_id}", status_code=204)
async def delete_concept(concept_id: int, request: Request):
    """Delete a concept and all its edges."""
    driver = request.app.state.graph_driver
    if not driver:
        raise HTTPException(503, "Graph engine is disabled")

    from src.services.graph_store import delete_concept as _delete
    deleted = await _delete(driver, concept_id)
    if not deleted:
        raise HTTPException(404, "Concept not found")
