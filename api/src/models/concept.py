"""Concept (graph node) schemas."""

from pydantic import BaseModel, Field


class ConceptCreate(BaseModel):
    name: str = Field(..., min_length=1)
    type: str = Field(default="Entity")
    description: str = Field(default="")
    metadata: dict = Field(default_factory=dict)


class ConceptOut(BaseModel):
    id: int
    name: str
    type: str
    description: str
    metadata: dict


class ConceptDetail(ConceptOut):
    relations: list[dict] = []
