"""Application configuration loaded from environment variables."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Postgres
    database_url: str = "postgresql://knowledge:changeme@postgres:5432/knowledge"

    # TEI embedding
    embedding_url: str = "http://embedding:80"

    # TEI reranker
    reranker_url: str = "http://reranker:80"
    rerank_enabled: bool = True

    # Memgraph
    memgraph_url: str = "bolt://memgraph:7687"
    memgraph_enabled: bool = True

    # Search tuning
    hnsw_ef_search: int = 100
    default_search_top_k: int = 20
    default_rerank_candidates: int = 50
    default_min_score: float = 0.40

    # Chunking
    chunk_size: int = 512
    chunk_overlap: int = 50

    # Server
    api_workers: int = 1
    api_log_level: str = "info"


settings = Settings()
