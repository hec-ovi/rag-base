-- ============================================================
-- rag-base schema
-- Runs once on first boot (via /docker-entrypoint-initdb.d/)
-- ============================================================

-- Extensions
CREATE EXTENSION IF NOT EXISTS vector;        -- pgvector: vector type + HNSW/IVFFlat
CREATE EXTENSION IF NOT EXISTS pg_trgm;       -- trigram similarity for fuzzy matching

-- ── Documents ───────────────────────────────────────────────
-- Source documents as ingested.
CREATE TABLE documents (
    id          BIGSERIAL PRIMARY KEY,
    title       TEXT NOT NULL,
    content     TEXT NOT NULL,
    metadata    JSONB NOT NULL DEFAULT '{}',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_documents_metadata ON documents USING gin (metadata);

-- ── Chunks ──────────────────────────────────────────────────
-- Chunks are fragments of documents, each with a vector and tsvector.
CREATE TABLE chunks (
    id           BIGSERIAL PRIMARY KEY,
    document_id  BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index  INT NOT NULL,
    content      TEXT NOT NULL,
    token_count  INT NOT NULL DEFAULT 0,
    embedding    vector(1024),
    tsv          tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    metadata     JSONB NOT NULL DEFAULT '{}',
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Foreign key lookup
CREATE INDEX idx_chunks_document_id ON chunks (document_id);

-- Full-text search (keyword retrieval)
CREATE INDEX idx_chunks_tsv ON chunks USING gin (tsv);

-- Vector similarity search (semantic retrieval)
-- HNSW index — created after first data load is more efficient,
-- but we create it here so it's ready from the start.
CREATE INDEX idx_chunks_embedding ON chunks
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Prevent duplicate chunks per document
CREATE UNIQUE INDEX idx_chunks_doc_index ON chunks (document_id, chunk_index);

-- ── Helper: update updated_at on documents ──────────────────
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_documents_updated_at
    BEFORE UPDATE ON documents
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();
