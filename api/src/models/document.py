"""Document schemas."""

from datetime import datetime

from pydantic import BaseModel, Field


class DocumentCreate(BaseModel):
    title: str = Field(..., min_length=1)
    content: str = Field(..., min_length=1)
    metadata: dict = Field(default_factory=dict)
    contextual_retrieval: bool = Field(
        default=False,
        description=(
            "When true, ingest runs Anthropic Contextual Retrieval per chunk: "
            "the LLM (LLM_BASE_URL) generates a 50-100 token blurb situating the "
            "chunk in the document, prepended to chunks.indexed_content. Default "
            "false so existing callers see no behavior change. Requires the LLM "
            "endpoint to be reachable; if not, blurbs are silently skipped per "
            "chunk and ingest still succeeds."
        ),
    )


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
