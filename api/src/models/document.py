"""Document schemas."""

from datetime import datetime

from pydantic import BaseModel, Field


class DocumentCreate(BaseModel):
    title: str = Field(..., min_length=1)
    content: str = Field(..., min_length=1)
    metadata: dict = Field(default_factory=dict)


class ChunkOut(BaseModel):
    id: int
    chunk_index: int
    content: str
    token_count: int


class DocumentOut(BaseModel):
    id: int
    title: str
    content: str
    metadata: dict
    created_at: datetime
    updated_at: datetime
    chunk_count: int = 0


class DocumentDetail(DocumentOut):
    chunks: list[ChunkOut] = []
