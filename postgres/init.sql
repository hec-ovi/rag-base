-- ============================================================
-- rag-base schema
-- Runs once on first boot (via /docker-entrypoint-initdb.d/)
-- ============================================================

-- Extensions
CREATE EXTENSION IF NOT EXISTS vector;        -- pgvector: vector type + HNSW/IVFFlat
CREATE EXTENSION IF NOT EXISTS pg_search;     -- ParadeDB pg_search: real BM25 via Tantivy
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
-- Chunks are fragments of documents, each with a vector and a BM25 index.
-- Phase 3a swapped tsvector + ts_rank for ParadeDB pg_search BM25 (real
-- IDF + length normalization + term-frequency saturation), no DB migration.
-- Phase 3c added indexed_content: the augmented text fed to embedding +
-- BM25 = "[title | metadata] [header > path] <chunk>". The raw chunk lives
-- in content unchanged so display + reconstruction stay clean.
CREATE TABLE chunks (
    id              BIGSERIAL PRIMARY KEY,
    document_id     BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index     INT NOT NULL,
    content         TEXT NOT NULL,
    indexed_content TEXT NOT NULL,
    token_count     INT NOT NULL DEFAULT 0,
    embedding       vector(1024),
    metadata        JSONB NOT NULL DEFAULT '{}',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Foreign key lookup
CREATE INDEX idx_chunks_document_id ON chunks (document_id);

-- Full-text search (keyword retrieval), real BM25 via pg_search.
-- Indexes indexed_content (title + header path + chunk) so BM25 sees the
-- same disambiguating prefix the embedder sees.
CREATE INDEX idx_chunks_bm25 ON chunks
    USING bm25 (id, indexed_content)
    WITH (key_field = 'id');

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
