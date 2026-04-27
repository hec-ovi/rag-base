"""Real BM25 keyword retrieval via ParadeDB pg_search.

Phase 3a swap. Replaced the previous tsvector + ts_rank implementation, which
lacks IDF, term-frequency saturation, and document-length normalization. The
new pg_search backend embeds Tantivy (Lucene-class BM25) inside Postgres, so
no separate cluster or migration is needed; coexists with pgvector in the
same database.

Phase 3c: BM25 now matches against indexed_content (title + header path +
chunk), not the raw chunk. This mirrors what the embedder sees so both
retrieval channels benefit from the same structural disambiguation.

The `|||` operator runs a match-disjunction against the BM25-indexed column.
`pdb.score(id)` produces the BM25 score for the matching row; rows that do
not match return null (filtered out by the `WHERE` clause anyway).
"""

import asyncpg


async def search_keyword(
    pool: asyncpg.Pool,
    query: str,
    top_k: int = 50,
) -> list[dict]:
    """BM25 keyword search over chunk indexed_content via pg_search.

    Matches against indexed_content (title + header-path + chunk) but returns
    the raw c.content for display.

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
                pdb.score(c.id) AS score
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE c.indexed_content ||| $1
            ORDER BY score DESC
            LIMIT $2
            """,
            query,
            top_k,
        )
    return [{**dict(row), "source": "keyword"} for row in rows]
