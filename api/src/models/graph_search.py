"""Pydantic models for the Phase 5 graph-only endpoint.

API shape follows the research finding at
`.research/graph-only-retrieval-no-embedding-fast-mode/FINDINGS.md` (section
"Recommended API shape"): structured, machine-consumable payload with the full
matched-entity / subgraph / chunks triplet plus a per-query trace block.
"""

from typing import Literal

from pydantic import BaseModel, Field

# Match methods that the matcher can attribute to a hit. Stable enum so callers
# can branch on it (e.g. only trust exact for high-stakes flows).
MatchMethod = Literal["exact_ci", "fuzzy_contains"]


class GraphSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Raw user query, used for NER and as a logging key.")
    max_entities: int = Field(default=8, ge=1, le=32, description="Cap on entity strings forwarded from NER to the graph match step.")
    hops: int = Field(default=1, ge=0, le=2, description="Graph traversal radius from each matched entity. 0 returns just the matched entities and their attached chunks.")
    ranking: Literal["none", "degree"] = Field(default="degree", description="Subgraph node ordering. 'degree' sorts by node degree (in+out), 'none' preserves match order.")
    top_k_chunks: int = Field(default=20, ge=1, le=200, description="Maximum number of source chunks returned. Chunks are ordered by association strength (number of matched/expanded entities pointing at them).")
    fuzzy: bool = Field(default=True, description="If True, also match graph nodes whose entity_id case-insensitively CONTAINS an extracted name. False forces exact-only.")
    ner_labels: list[str] | None = Field(default=None, description="Custom GLiNER label inventory. Falls back to the service's default vocabulary when omitted.")


class MatchedEntity(BaseModel):
    id: str
    name: str
    type: str
    match_score: float = Field(..., description="NER confidence for the source phrase that matched this node. Equal across all nodes that matched the same phrase.")
    match_method: MatchMethod
    source_phrase: str = Field(..., description="The phrase NER extracted from the query that produced this match.")


class SubgraphNode(BaseModel):
    id: str
    name: str
    type: str
    description: str | None = None
    degree: int


class SubgraphEdge(BaseModel):
    source: str
    target: str
    relation: str = Field(..., description="The natural-language description of the edge (LightRAG stores semantic relations as descriptions, not typed edges).")
    weight: float = 1.0
    keywords: str | None = None


class GraphSearchChunk(BaseModel):
    chunk_id: int | None = Field(default=None, description="rag-base chunks.id when the LightRAG hash bridged successfully via (full_doc_id, chunk_order_index). None if no bridge could be made.")
    doc_id: int | None
    lightrag_chunk_hash: str = Field(..., description="The chunk-<hash> id LightRAG attached to the entity's source_id.")
    text: str
    source_entities: list[str] = Field(..., description="entity_ids from the subgraph whose source_id contained this chunk hash.")


class GraphSearchTrace(BaseModel):
    latency_ms: float
    ner_ms: float
    graph_ms: float
    chunk_bridge_ms: float
    n_ner_entities: int
    n_matched_nodes: int
    n_subgraph_nodes: int
    n_subgraph_edges: int


class GraphSearchResponse(BaseModel):
    query: str
    matched_entities: list[MatchedEntity]
    subgraph: dict = Field(..., description="{'nodes': [SubgraphNode], 'edges': [SubgraphEdge]}")
    chunks: list[GraphSearchChunk]
    trace: GraphSearchTrace
