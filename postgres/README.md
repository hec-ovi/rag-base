# postgres

PostgreSQL 17 with pgvector 0.8.2 extension.

**Image:** `pgvector/pgvector:0.8.2-pg17`
**Port:** 5433 host → 5432 internal (configurable via `POSTGRES_PORT`)

## What it does

Stores documents, chunks, and their vector embeddings. Handles both semantic search (vector similarity via HNSW index) and keyword search (full-text via tsvector + GIN index).

## Schema

- `documents` - source documents with JSONB metadata
- `chunks` - document fragments with `embedding vector(1024)` and auto-generated `tsvector` columns
- HNSW index on embeddings (cosine distance)
- GIN index on tsvector (full-text search)
- GIN index on document metadata (JSONB)
- Unique constraint on (document_id, chunk_index) to prevent duplicate chunks

## Persistence

Data is stored in a named Docker volume (`ragbase_pgdata`). Survives `docker compose down`. Only destroyed by `docker compose down -v`.

## Init

`init.sql` runs automatically on first boot via `/docker-entrypoint-initdb.d/`. It will not re-run if the data volume already exists.
