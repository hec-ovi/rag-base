"""LightRAG integration for rag-base.

Phase 2 design: LightRAG owns ingest-time entity-and-relation extraction; the
extracted graph lives in Memgraph. At query time we do NOT call LightRAG's full
retrieval pipeline (too slow on the local LLM). Instead:

  1. Use the LLM to extract entity names from the query (concise mode, ~5-30s).
  2. Cypher-query Memgraph for concepts matching those names plus their 1-hop
     neighbors. Each concept node carries a source_doc_ids array (LightRAG sets
     this at ingest).
  3. Aggregate the source doc_ids and run semantic search RESTRICTED to those
     documents. Those chunks become the "graph" channel for RRF.

This is the integration that closes the rag-base search.py:66 TODO. The TODO
was "link concepts to chunks for graph -> RRF integration"; we link by going
graph -> doc_id -> top semantic chunks of that doc.

Why not LightRAG's own aquery: the local Qwen3.6-27B is a reasoning model; a full
LightRAG aquery makes multiple LLM calls per query and would push p95 latency
into multi-minute territory. Using LightRAG ONLY for ingest (where slow is
acceptable, runs in background) keeps query latency dominated by the existing
hybrid+rerank path.

LightRAG storage backends:
  - Graph: MemgraphStorage (we already run Memgraph)
  - Vector / KV / DocStatus: file-based (NanoVectorDB / JsonKV / JsonDocStatus)
    in working_dir. File-based is fine for a single-instance backend; migrating
    to Postgres-backed storage is a future optimization.
"""

import asyncio
import json
import logging
import os
import re
import time
from typing import Awaitable, Callable

import httpx
import numpy as np

logger = logging.getLogger(__name__)

LLMComplete = Callable[..., Awaitable[str]]


def make_lightrag_embedding_func(embed_client: httpx.AsyncClient, embedding_dim: int = 1024):
    """Build the LightRAG embedding callable backed by our TEI embedding service."""
    from lightrag.utils import wrap_embedding_func_with_attrs

    @wrap_embedding_func_with_attrs(embedding_dim=embedding_dim, max_token_size=8192)
    async def embed(texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, embedding_dim), dtype=np.float32)
        r = await embed_client.post("/embed", json={"inputs": texts}, timeout=120.0)
        r.raise_for_status()
        vectors = r.json()
        return np.array(vectors, dtype=np.float32)

    return embed


def make_lightrag_llm_func(llm_complete: LLMComplete):
    """Adapt our llm_complete closure to LightRAG's llm_model_func signature.

    LightRAG passes keyword_extraction=True for short query-key calls and False for
    long extraction calls. We map that to our `concise` flag.
    """

    async def lightrag_llm(
        prompt: str,
        system_prompt: str | None = None,
        history_messages: list[dict] | None = None,
        keyword_extraction: bool = False,
        **kwargs,
    ) -> str:
        return await llm_complete(
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            concise=keyword_extraction,
        )

    return lightrag_llm


async def init_lightrag(
    working_dir: str,
    embed_client: httpx.AsyncClient,
    llm_complete: LLMComplete,
    memgraph_url: str,
    *,
    default_llm_timeout: int = 900,
    llm_model_max_async: int = 2,
    chunk_token_size: int = 600,
    entity_extract_max_gleaning: int = 0,
):
    """Construct, initialize, and return a LightRAG instance configured for rag-base.

    Caller must `await rag.finalize_storages()` at shutdown.

    Defaults are tuned for the local Qwen3.6-27B reasoning vLLM:
      - default_llm_timeout=900: give each LLM call 15 minutes. Reasoning models
        can take 5+ minutes on substantive extraction; LightRAG's worker timeout
        is roughly 2x this value so the overall ceiling lands around 30 minutes
        per chunk. Slow but bounded.
      - llm_model_max_async=2: at most 2 concurrent extraction calls. Higher
        concurrency causes vLLM queue contention with this model.
      - chunk_token_size=600: smaller LightRAG chunks => simpler extraction =>
        faster per-chunk LLM call. Independent of our rag-base chunks.
      - entity_extract_max_gleaning=0: skip the follow-up cleanup pass. With a
        slow LLM the cleanup doubles ingest time for marginal quality gain.
    """
    from lightrag import LightRAG
    from lightrag.kg.shared_storage import initialize_pipeline_status

    os.makedirs(working_dir, exist_ok=True)

    # MemgraphStorage reads connection from env vars at construction.
    os.environ.setdefault("MEMGRAPH_URI", memgraph_url)
    os.environ.setdefault("MEMGRAPH_USER", "")
    os.environ.setdefault("MEMGRAPH_PASSWORD", "")

    rag = LightRAG(
        working_dir=working_dir,
        embedding_func=make_lightrag_embedding_func(embed_client),
        llm_model_func=make_lightrag_llm_func(llm_complete),
        graph_storage="MemgraphStorage",
        vector_storage="NanoVectorDBStorage",
        kv_storage="JsonKVStorage",
        doc_status_storage="JsonDocStatusStorage",
        default_llm_timeout=default_llm_timeout,
        llm_model_max_async=llm_model_max_async,
        chunk_token_size=chunk_token_size,
        entity_extract_max_gleaning=entity_extract_max_gleaning,
    )
    await rag.initialize_storages()
    await initialize_pipeline_status()
    logger.info("LightRAG initialized at working_dir=%s", working_dir)
    return rag


def doc_lightrag_id(document_id: int) -> str:
    """Stable LightRAG-side ID for our document."""
    return f"doc_{document_id}"


_DOC_ID_RE = re.compile(r"\bdoc_(\d+)\b")


def parse_doc_ids_from_string(s: str) -> list[int]:
    """Best-effort extraction of doc_<int> markers from a LightRAG output string."""
    seen: dict[int, None] = {}
    for match in _DOC_ID_RE.finditer(s or ""):
        try:
            seen.setdefault(int(match.group(1)), None)
        except ValueError:
            continue
    return list(seen.keys())


async def lightrag_insert(
    rag,
    content: str,
    document_id: int,
    *,
    timeout: float = 1800.0,
    poll_interval: float = 5.0,
) -> bool:
    """Ingest content into LightRAG, tagged with our document_id.

    Returns True only after verifying the doc's final status is "processed".

    LightRAG's `ainsert` is non-blocking when its internal pipeline queue is
    busy: it returns immediately after enqueueing, even though extraction has
    not started. The doc sits in PENDING state until the worker picks it up.
    A naive "check status right after ainsert" pattern always sees PENDING and
    falsely reports failure. We must POLL doc_status until it transitions to a
    terminal state (processed / failed) or we hit the outer timeout.

    `timeout` is the wall-clock cap (default 30 min). Real-world bound per chunk
    is roughly `default_llm_timeout * 2` (LightRAG worker overhead).
    """
    t0 = time.perf_counter()
    lightrag_id = doc_lightrag_id(document_id)
    try:
        await asyncio.wait_for(
            rag.ainsert(content, ids=[lightrag_id]),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        logger.warning("LightRAG ainsert timed out after %.0fs for doc %d", timeout, document_id)
        return False
    except Exception as e:
        logger.warning("LightRAG ainsert raised for doc %d: %s", document_id, e)
        return False

    # Poll until the doc reaches a terminal status or we exhaust the wall-clock budget.
    # Terminal statuses (LightRAG DocStatus enum): "processed" (good), "failed" (bad).
    # Non-terminal: "pending", "processing", and any other unknown status.
    deadline = t0 + timeout
    last_status = "unknown"
    while time.perf_counter() < deadline:
        try:
            statuses = await rag.doc_status.get_by_ids([lightrag_id])
        except Exception as e:
            logger.warning("LightRAG doc_status lookup failed for doc %d: %s", document_id, e)
            return False

        if not statuses or not statuses[0]:
            await asyncio.sleep(poll_interval)
            continue

        status_obj = statuses[0].get("status")
        # status_obj may be a string OR a DocStatus enum (.value attr); normalize
        status = getattr(status_obj, "value", str(status_obj)).lower()
        last_status = status

        if status == "processed":
            logger.info(
                "LightRAG ingest verified processed for doc %d in %.1fs",
                document_id, time.perf_counter() - t0,
            )
            return True
        if status == "failed":
            logger.warning(
                "LightRAG doc %d ended in status=failed after %.1fs",
                document_id, time.perf_counter() - t0,
            )
            return False

        await asyncio.sleep(poll_interval)

    logger.warning(
        "LightRAG doc %d still status=%s after %.0fs (wall-clock timeout)",
        document_id, last_status, time.perf_counter() - t0,
    )
    return False


# ---------- Query-side: query -> entity names -> graph -> doc_ids ----------

_ENTITY_EXTRACTION_PROMPT = """Extract the named entities from the following question.
Output ONLY a JSON list of strings (entity names), nothing else.
If there are no clear entities, output [].

Question: {query}"""


async def extract_query_entities(llm_complete: LLMComplete, query: str) -> list[str]:
    """Ask the LLM for entity names mentioned in the query.

    Uses concise mode so the model goes direct to JSON without deliberation.
    Returns empty list on parse failure.
    """
    try:
        text = await llm_complete(
            _ENTITY_EXTRACTION_PROMPT.format(query=query),
            concise=True,
        )
    except Exception as e:
        logger.warning("Entity extraction LLM call failed: %s", e)
        return []

    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()

    try:
        names = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if not match:
            return []
        try:
            names = json.loads(match.group(0))
        except json.JSONDecodeError:
            return []

    if not isinstance(names, list):
        return []
    return [str(n).strip() for n in names if str(n).strip()][:20]


async def find_docs_via_graph(rag, graph_driver, entity_names: list[str]) -> list[int]:
    """Find document IDs whose entities match `entity_names` (or are 1-hop neighbors of matches).

    Schema verified empirically against a real LightRAG ingest:
      Entity nodes (label `:base`): {entity_id, entity_type, description, source_id}
        where source_id = "chunk-<hash>" (LightRAG's internal chunk id; LightRAG
        joins multiple with "<SEP>" if the entity appears in multiple chunks)
      Edges (type `DIRECTED`): {description, keywords, source_id, weight}

    The rag-base doc id we want is NOT on the entity node directly. We have to
    bridge via LightRAG's text_chunks KV store: each chunk record holds a
    `full_doc_id` field equal to the id we passed to `ainsert(ids=[...])`,
    which in our case is `doc_<int>` (see doc_lightrag_id above).

    Pipeline:
      1. Cypher: match entities by entity_id case-insensitive (plus 1-hop neighbors),
         collect distinct source_id strings.
      2. Split each source_id on "<SEP>" -> list of chunk hashes.
      3. Batch lookup chunk hashes in rag.text_chunks.
      4. Parse `full_doc_id` of each chunk -> our doc id.
    """
    if not entity_names or not graph_driver or rag is None:
        return []

    cypher = """
    MATCH (e:base)
    WHERE any(name IN $names WHERE toLower(e.entity_id) = toLower(name))
    OPTIONAL MATCH (e)-[r]-(n:base)
    WITH collect(DISTINCT e) + collect(DISTINCT n) AS nodes
    UNWIND nodes AS x
    WITH DISTINCT x
    WHERE x IS NOT NULL AND x.source_id IS NOT NULL
    RETURN x.source_id AS source_id
    """

    chunk_ids: set[str] = set()
    try:
        async with graph_driver.session() as session:
            result = await session.run(cypher, names=entity_names)
            async for record in result:
                source_id = record.get("source_id") or ""
                if not source_id:
                    continue
                for cid in str(source_id).split("<SEP>"):
                    cid = cid.strip()
                    if cid:
                        chunk_ids.add(cid)
    except Exception as e:
        logger.warning("find_docs_via_graph cypher failed: %s", e)
        return []

    if not chunk_ids:
        return []

    try:
        chunk_records = await rag.text_chunks.get_by_ids(list(chunk_ids))
    except Exception as e:
        logger.warning("text_chunks.get_by_ids failed: %s", e)
        return []

    doc_ids: dict[int, None] = {}
    for record in chunk_records or []:
        if not record:
            continue
        full_doc_id = record.get("full_doc_id") if isinstance(record, dict) else None
        for did in parse_doc_ids_from_string(str(full_doc_id or "")):
            doc_ids.setdefault(did, None)

    return list(doc_ids.keys())
