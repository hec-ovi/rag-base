"""pgvector semantic search queries."""

import asyncpg


async def search_semantic(
    pool: asyncpg.Pool,
    query_embedding: list[float],
    top_k: int = 50,
    min_score: float = 0.0,
) -> list[dict]:
    """Cosine similarity search over chunk embeddings.

    Returns list of {chunk_id, document_id, document_title, content, score, source}.
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT
                c.id AS chunk_id,
                c.document_id,
                d.title AS document_title,
                c.content,
                1 - (c.embedding <=> $1::vector) AS score
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE c.embedding IS NOT NULL
            ORDER BY c.embedding <=> $1::vector
            LIMIT $2
            """,
            query_embedding,
            top_k,
        )
    return [
        {**dict(row), "source": "semantic"}
        for row in rows
        if row["score"] >= min_score
    ]


async def search_semantic_in_docs(
    pool: asyncpg.Pool,
    query_embedding: list[float],
    document_ids: list[int],
    top_k: int = 50,
    min_score: float = 0.0,
    source_label: str = "graph",
) -> list[dict]:
    """Cosine similarity over chunks RESTRICTED to a list of documents.

    Used by the graph channel: LightRAG identifies relevant docs, then we pick
    the most query-relevant chunks within them. Result tagged source="graph"
    (default) so RRF in search.py can credit the channel correctly.
    """
    if not document_ids:
        return []
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT
                c.id AS chunk_id,
                c.document_id,
                d.title AS document_title,
                c.content,
                1 - (c.embedding <=> $1::vector) AS score
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE c.embedding IS NOT NULL
              AND c.document_id = ANY($3::bigint[])
            ORDER BY c.embedding <=> $1::vector
            LIMIT $2
            """,
            query_embedding,
            top_k,
            document_ids,
        )
    return [
        {**dict(row), "source": source_label}
        for row in rows
        if row["score"] >= min_score
    ]


async def insert_document_with_chunks(
    pool: asyncpg.Pool,
    title: str,
    content: str,
    metadata: dict,
    chunks: list[dict],
) -> dict:
    """Insert a document and all its chunks atomically in a single transaction.

    Each chunk dict has: content, indexed_content, chunk_index, token_count, embedding.
    indexed_content is the title + header-path + chunk text fed to the embedder
    and indexed by BM25; content is the raw chunk text for display.
    Returns the document row as a dict with chunk_count.
    """
    async with pool.acquire() as conn:
        async with conn.transaction():
            row = await conn.fetchrow(
                """
                INSERT INTO documents (title, content, metadata)
                VALUES ($1, $2, $3)
                RETURNING id, title, content, metadata, created_at, updated_at
                """,
                title,
                content,
                metadata,
            )
            doc_id = row["id"]

            if chunks:
                await conn.executemany(
                    """
                    INSERT INTO chunks (document_id, chunk_index, content, indexed_content, token_count, embedding)
                    VALUES ($1, $2, $3, $4, $5, $6::vector)
                    """,
                    [
                        (
                            doc_id,
                            chunk["chunk_index"],
                            chunk["content"],
                            chunk["indexed_content"],
                            chunk["token_count"],
                            chunk["embedding"],
                        )
                        for chunk in chunks
                    ],
                )

    return {**dict(row), "chunk_count": len(chunks)}
