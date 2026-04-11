"""PostgreSQL full-text search (tsvector/BM25-style keyword retrieval)."""

import asyncpg


async def search_keyword(
    pool: asyncpg.Pool,
    query: str,
    top_k: int = 50,
) -> list[dict]:
    """Full-text search over chunk content using tsvector + ts_rank.

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
                ts_rank(c.tsv, websearch_to_tsquery('english', $1)) AS score
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE c.tsv @@ websearch_to_tsquery('english', $1)
            ORDER BY score DESC
            LIMIT $2
            """,
            query,
            top_k,
        )
    return [{**dict(row), "source": "keyword"} for row in rows]
