# postgres

PostgreSQL 17 with pgvector + ParadeDB pg_search.

**Image:** `paradedb/paradedb:0.23.1-pg17` (bundles `pgvector` + `pg_search`)
**Port:** 5433 host -> 5432 internal (configurable via `POSTGRES_PORT`)

## What it does

Stores documents, chunks, and their vector embeddings. Handles both semantic search (vector similarity via HNSW index) and keyword search (real BM25 via pg_search / Tantivy).

## Schema

- `documents` - source documents with JSONB metadata
- `chunks` - document fragments with `content` (raw chunk text), `indexed_content` (augmented `[title | meta] [header > path] <chunk>` form fed to embedding + BM25), and `embedding vector(1024)`
- HNSW index on `chunks.embedding` (cosine distance)
- BM25 index on `chunks.indexed_content` via pg_search (`USING bm25 (id, indexed_content) WITH (key_field = 'id')`)
- GIN index on document metadata (JSONB)
- Unique constraint on (document_id, chunk_index) to prevent duplicate chunks

## Persistence

Data is bind-mounted to `./data/postgres` in the rag-base repo (with a `pgdata/` subdir to satisfy PG's empty-dir requirement). Survives `docker compose down`. Wipe by deleting `./data/postgres`. The `./data/` tree is gitignored.

## Init

`init.sql` runs automatically on first boot via `/docker-entrypoint-initdb.d/`. It will not re-run if the data directory already has a populated `pgdata/`.
