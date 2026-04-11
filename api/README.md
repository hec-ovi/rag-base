# api

FastAPI orchestrator - the only custom code in this stack.

**Image:** Built from `Dockerfile` (python:3.12-slim-bookworm)
**Port:** 5050 host → 8000 internal (configurable via `API_PORT`)

## What it does

Thin REST API that ties together Postgres, TEI embedding, TEI reranker, and Memgraph. Handles:

- Document ingestion (chunk, embed, store atomically)
- Hybrid search (semantic + keyword + graph, merged with RRF, reranked)
- Graph operations (concepts, relations, traversal, algorithms)
- Passthrough endpoints for embedding and reranking

## Startup behavior

Requires Postgres and TEI embedding to be healthy before starting (`depends_on` in compose).

Reranker and Memgraph are **optional** - the API probes them at startup. If unreachable, it logs a warning and disables them. The API never crashes due to missing optional services.

## Stack

- FastAPI with async throughout
- asyncpg for Postgres (connection pool, JSONB codec, pgvector registered)
- httpx for TEI calls (reusable async client)
- neo4j async driver for Memgraph (Bolt protocol)
- uvicorn + uvloop
