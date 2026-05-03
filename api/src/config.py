"""Application configuration loaded from environment variables."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Postgres
    database_url: str = "postgresql://knowledge:changeme@postgres:5432/knowledge"

    # TEI embedding
    embedding_url: str = "http://embedding:80"

    # TEI reranker (default, CPU)
    reranker_url: str = "http://reranker:80"
    rerank_enabled: bool = True

    # Optional GPU reranker sidecars (additive; default behavior unchanged when unset)
    bge_gpu_reranker_url: str | None = None
    qwen_reranker_url: str | None = None
    qwen_8b_reranker_url: str | None = None

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

    # LLM (used by LightRAG for entity extraction + query-key generation)
    llm_base_url: str = "http://host.docker.internal:8000"
    llm_model: str = "Qwen3.6-27B-AWQ4"
    llm_api_key: str = ""
    # Disabled by default: see api/src/services/llm_responses.py docstring for the
    # 50x latency measurement that motivated this default.
    llm_enable_thinking: bool = False


settings = Settings()
