"""Relation (graph edge) schemas."""

from pydantic import BaseModel, Field


class RelationCreate(BaseModel):
    source_name: str = Field(..., min_length=1)
    target_name: str = Field(..., min_length=1)
    relation_type: str = Field(..., min_length=1)
    metadata: dict = Field(default_factory=dict)


class RelationOut(BaseModel):
    id: int
    source: str
    target: str
    relation_type: str
    metadata: dict
