"""Document CRUD + ingest pipeline."""

from fastapi import APIRouter, HTTPException, Query, Request

from src.config import settings
from src.models.document import DocumentCreate, DocumentDetail, DocumentOut
from src.services.chunking import chunk_text
from src.services.embedding import embed_texts
from src.services.vector_store import insert_document_with_chunks

router = APIRouter(tags=["documents"])


@router.post("/documents", response_model=DocumentOut, status_code=201)
async def create_document(body: DocumentCreate, request: Request):
    """Ingest a document: chunk it, embed each chunk, store everything atomically."""
    pool = request.app.state.db_pool
    embed_client = request.app.state.embed_client

    # 1. Chunk the content
    texts = chunk_text(body.content, settings.chunk_size, settings.chunk_overlap)

    # 2. Embed all chunks in one batch (if any)
    vectors = await embed_texts(embed_client, texts) if texts else []

    # 3. Build chunk records
    chunks = [
        {
            "content": text,
            "chunk_index": i,
            "token_count": len(text.split()),
            "embedding": vector,
        }
        for i, (text, vector) in enumerate(zip(texts, vectors))
    ]

    # 4. Store document + chunks in a single transaction
    result = await insert_document_with_chunks(
        pool, body.title, body.content, body.metadata, chunks
    )

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
