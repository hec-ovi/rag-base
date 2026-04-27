"""Document CRUD + ingest pipeline."""

import logging

from fastapi import APIRouter, HTTPException, Query, Request

from src.config import settings
from src.models.document import DocumentCreate, DocumentDetail, DocumentOut
from src.services.chunking import chunk_text_with_headers
from src.services.embedding import embed_texts
from src.services.lightrag_store import lightrag_insert
from src.services.vector_store import insert_document_with_chunks

router = APIRouter(tags=["documents"])
logger = logging.getLogger(__name__)


@router.post("/documents", response_model=DocumentOut, status_code=201)
async def create_document(body: DocumentCreate, request: Request):
    """Ingest a document: chunk it, embed each chunk, store everything atomically.

    If LightRAG is configured (Memgraph + LLM both reachable), the document is
    also fed to LightRAG for entity-and-relation extraction into the graph. That
    call is awaited inline (so retrieval after ingest sees the graph) but failure
    does not block document creation: rag-base remains useful with hybrid +
    rerank even when graph extraction is unavailable or hangs.

    Test-only escape hatch: header `X-LightRAG-Ingest: false` skips the LightRAG
    step. Production traffic should never send this; tests that don't assert on
    graph behavior use it to avoid paying ~9 minutes of LLM time per ingest.
    """
    pool = request.app.state.db_pool
    embed_client = request.app.state.embed_client
    lightrag = getattr(request.app.state, "lightrag", None)
    skip_lightrag = request.headers.get("x-lightrag-ingest", "true").lower() == "false"

    # 1. Chunk the content with markdown header path tracking (Phase 3c).
    # Each chunk carries the breadcrumb (e.g. "Guide > Setup > Linux") active at
    # its first paragraph, so retrieval sees the doc structure that put the chunk
    # there. Free disambiguation lift; zero LLM cost.
    chunk_records = chunk_text_with_headers(
        body.content, settings.chunk_size, settings.chunk_overlap
    )

    # 2. Build contextual chunk headers (CCH) for embedding + BM25.
    # Augmented form fed to the embedder and stored as indexed_content:
    #   [title | meta_k: meta_v | ...] [Header > Path] <chunk text>
    # The header-path bracket is omitted when the chunk lives under no heading.
    # Original raw chunk text is stored separately in chunks.content.
    meta_parts = [body.title]
    for k, v in body.metadata.items():
        meta_parts.append(f"{k}: {v}")
    title_meta = " | ".join(meta_parts)

    indexed_texts: list[str] = []
    for record in chunk_records:
        if record["header_path"]:
            indexed_texts.append(
                f"[{title_meta}] [{record['header_path']}] {record['content']}"
            )
        else:
            indexed_texts.append(f"[{title_meta}] {record['content']}")

    # 3. Embed all augmented chunks in one batch (if any)
    vectors = await embed_texts(embed_client, indexed_texts) if indexed_texts else []

    # 4. Build chunk records (raw content + augmented indexed_content)
    chunks = [
        {
            "content": record["content"],
            "indexed_content": indexed,
            "chunk_index": i,
            "token_count": len(record["content"].split()),
            "embedding": vector,
        }
        for i, (record, indexed, vector) in enumerate(
            zip(chunk_records, indexed_texts, vectors)
        )
    ]

    # 5. Store document + chunks in a single transaction
    result = await insert_document_with_chunks(
        pool, body.title, body.content, body.metadata, chunks
    )

    # 6. Feed to LightRAG for entity/relation extraction (best-effort).
    # Slow on the local reasoning vLLM (1-5 min per chunk) but bounded by
    # lightrag_insert's internal timeout. Failure is logged, not propagated.
    if lightrag is not None and not skip_lightrag:
        try:
            ok = await lightrag_insert(lightrag, body.content, result["id"])
            if not ok:
                logger.warning("LightRAG ingest did not complete cleanly for doc %d", result["id"])
        except Exception as e:
            logger.warning("LightRAG ingest raised for doc %d: %s", result["id"], e)

    return result


@router.get("/documents", response_model=list[DocumentOut])
async def list_documents(
    request: Request,
    offset: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
):
    """List documents, paginated."""
    pool = request.app.state.db_pool
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT d.*, (SELECT count(*) FROM chunks c WHERE c.document_id = d.id) AS chunk_count
            FROM documents d
            ORDER BY d.created_at DESC
            OFFSET $1 LIMIT $2
            """,
            offset,
            limit,
        )
    return [dict(row) for row in rows]


@router.get("/documents/{doc_id}", response_model=DocumentDetail)
async def get_document(doc_id: int, request: Request):
    """Get a document with all its chunks."""
    pool = request.app.state.db_pool
    async with pool.acquire() as conn:
        doc = await conn.fetchrow("SELECT * FROM documents WHERE id = $1", doc_id)
        if not doc:
            raise HTTPException(404, "Document not found")
        chunks = await conn.fetch(
            """
            SELECT id, chunk_index, content, token_count
            FROM chunks WHERE document_id = $1 ORDER BY chunk_index
            """,
            doc_id,
        )
    return {
        **dict(doc),
        "chunk_count": len(chunks),
        "chunks": [dict(c) for c in chunks],
    }


@router.delete("/documents/{doc_id}", status_code=204)
async def delete_document(doc_id: int, request: Request):
    """Delete a document and all its chunks (CASCADE)."""
    pool = request.app.state.db_pool
    async with pool.acquire() as conn:
        result = await conn.execute("DELETE FROM documents WHERE id = $1", doc_id)
    if result == "DELETE 0":
        raise HTTPException(404, "Document not found")
