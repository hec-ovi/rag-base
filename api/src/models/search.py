"""Search schemas."""

from typing import Literal

from pydantic import BaseModel, Field


RerankModel = Literal["default", "bge-gpu", "qwen-4b", "qwen-8b"]


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=20, ge=1, le=100)
    rerank: bool = True
    rerank_candidates: int = Field(default=50, ge=1, le=200)
    min_score: float = Field(default=0.0, ge=0.0, le=1.0)
    include_graph: bool = True
    rerank_model: RerankModel | None = None  # None == today's behavior (default CPU TEI)


class SearchResult(BaseModel):
    chunk_id: int
    document_id: int
    document_title: str
    content: str
    score: float
    sources: list[str] = []  # which retrievers contributed: "semantic", "keyword", "graph"


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]
    total: int
    retrievers_used: list[str]


class EmbedRequest(BaseModel):
    inputs: list[str] = Field(..., min_length=1)


class EmbedResponse(BaseModel):
    embeddings: list[list[float]]
    model: str
    dimensions: int


class RerankRequest(BaseModel):
    query: str = Field(..., min_length=1)
    texts: list[str] = Field(..., min_length=1)
    return_text: bool = False


class RerankResult(BaseModel):
    index: int
    score: float
    text: str | None = None


class RerankResponse(BaseModel):
    results: list[RerankResult]
    model: str
