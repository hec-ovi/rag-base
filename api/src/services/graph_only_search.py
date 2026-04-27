"""Graph-only retrieval orchestration for Phase 5.

Pipeline (per `.research/graph-only-retrieval-no-embedding-fast-mode/FINDINGS.md`):

  raw user query
    -> NER (GLiNER, no LLM, no embedding)
    -> entity-string -> :base node match in Memgraph (exact case-insensitive,
       optionally fuzzy CONTAINS)
    -> N-hop traversal (0..2)
    -> optional ranking (degree-sort or none)
    -> source_id bridge: entity.source_id ("chunk-<hash><SEP>...") split, look up
       LightRAG text_chunks by hash, take (full_doc_id, chunk_order_index)
    -> map to rag-base chunks.id via the (document_id, chunk_index) unique pair
    -> structured response

What this module DOES NOT do (deliberate):
- No pgvector lookup, no BM25, no RRF, no cross-encoder rerank, no LLM call.
- No writes to any store.
- No mutation of LightRAG state.

The ingest path is untouched. Every function here only reads from Memgraph,
the LightRAG KV stores, and Postgres.
"""

import logging
import re
import time
from typing import Any

import asyncpg

logger = logging.getLogger(__name__)

# Memgraph entity-id values are case-sensitive Python strings, but real-world
# input from the NER side is messy: "berkeley" should still match "Berkeley".
# We compare in lowercase. Fuzzy mode additionally matches CONTAINS.
_EXACT_CYPHER = """
MATCH (e:base)
WHERE toLower(e.entity_id) = toLower($name)
RETURN e.entity_id AS id, e.entity_type AS type, e.description AS description, e.source_id AS source_id
LIMIT 8
"""

_FUZZY_CYPHER = """
MATCH (e:base)
WHERE toLower(e.entity_id) CONTAINS toLower($name)
RETURN e.entity_id AS id, e.entity_type AS type, e.description AS description, e.source_id AS source_id
LIMIT 8
"""

# Subgraph expansion runs as TWO queries (nodes first, then edges among those
# nodes). Earlier we tried a single Cypher that built node + edge lists via
# list comprehension over `collect(DISTINCT r)`; Memgraph collapsed `r.description`
# to a single value across all returned edges, producing identical relations on
# distinct edges. Splitting into two simple queries dodges that and is easier
# to reason about. Cost is one extra round trip; latency budget unaffected.

# Step A: collect node entity_ids reachable within `hops` steps of the seeds.
# OPTIONAL MATCH so seeds with no neighbors still come back.
_SUBGRAPH_NODES_CYPHER_0HOP = """
MATCH (n:base)
WHERE n.entity_id IN $seed_ids
RETURN n.entity_id AS id, n.entity_type AS type, n.description AS description, n.source_id AS source_id
"""

# For 1+ hops, we union the seeds with their N-hop neighbors. UNWIND + DISTINCT
# does the dedup; LIMIT bounds the blast radius for hub nodes.
_SUBGRAPH_NODES_CYPHER_NHOP = """
MATCH (seed:base)
WHERE seed.entity_id IN $seed_ids
OPTIONAL MATCH (seed)-[*1..%d]-(neighbor:base)
WITH collect(DISTINCT seed) + collect(DISTINCT neighbor) AS all_nodes
UNWIND all_nodes AS n
WITH DISTINCT n
WHERE n IS NOT NULL
RETURN n.entity_id AS id, n.entity_type AS type, n.description AS description, n.source_id AS source_id
LIMIT 200
"""

# Step B: edges WHERE both endpoints are in the node set we just built. This
# is the safe place to read relationship properties: each row is a real bound
# relationship, no list-comprehension trickery.
_SUBGRAPH_EDGES_CYPHER = """
MATCH (a:base)-[r]-(b:base)
WHERE a.entity_id IN $node_ids
  AND b.entity_id IN $node_ids
  AND id(a) < id(b)
RETURN
    a.entity_id AS source,
    b.entity_id AS target,
    r.description AS description,
    r.keywords AS keywords,
    coalesce(r.weight, 1.0) AS weight
LIMIT 500
"""


def _split_source_id(source_id: str) -> list[str]:
    """LightRAG packs multiple chunk hashes per node with `<SEP>`."""
    if not source_id:
        return []
    return [s.strip() for s in str(source_id).split("<SEP>") if s.strip()]


_DOC_ID_RE = re.compile(r"^doc_(\d+)$")


def _parse_doc_id(full_doc_id: str | None) -> int | None:
    """LightRAG stores our doc_id as 'doc_<int>' in full_doc_id."""
    if not full_doc_id:
        return None
    m = _DOC_ID_RE.match(str(full_doc_id))
    if not m:
        return None
    try:
        return int(m.group(1))
    except (TypeError, ValueError):
        return None


async def _match_entities(
    graph_driver,
    ner_results: list[dict],
    fuzzy: bool,
) -> list[dict]:
    """For each NER hit, find the :base node(s) it matches in Memgraph.

    Returns a flat list of {id, type, description, source_id, source_phrase, match_score, match_method}.
    Multiple matches for the same source_phrase are kept (a fuzzy CONTAINS may
    legitimately hit "Berkeley" and "Berkeley, California" both).
    """
    matched: list[dict] = []
    seen: set[tuple[str, str]] = set()  # (entity_id, source_phrase)

    async with graph_driver.session() as session:
        for hit in ner_results:
            phrase = hit["text"].strip()
            if not phrase:
                continue
            score = float(hit.get("score", 0.0))

            # Try exact case-insensitive first
            try:
                exact_result = await session.run(_EXACT_CYPHER, name=phrase)
                exact_rows = [dict(r) async for r in exact_result]
            except Exception as e:
                logger.warning("graph match (exact) failed for %r: %s", phrase, e)
                exact_rows = []

            if exact_rows:
                for row in exact_rows:
                    key = (row["id"], phrase)
                    if key in seen:
                        continue
                    seen.add(key)
                    matched.append({
                        "id": row["id"],
                        "type": row.get("type") or "",
                        "description": row.get("description") or "",
                        "source_id": row.get("source_id") or "",
                        "source_phrase": phrase,
                        "match_score": score,
                        "match_method": "exact_ci",
                    })
                continue

            if not fuzzy:
                continue

            # Fall back to CONTAINS
            try:
                fuzzy_result = await session.run(_FUZZY_CYPHER, name=phrase)
                fuzzy_rows = [dict(r) async for r in fuzzy_result]
            except Exception as e:
                logger.warning("graph match (fuzzy) failed for %r: %s", phrase, e)
                fuzzy_rows = []

            for row in fuzzy_rows:
                key = (row["id"], phrase)
                if key in seen:
                    continue
                seen.add(key)
                matched.append({
                    "id": row["id"],
                    "type": row.get("type") or "",
                    "description": row.get("description") or "",
                    "source_id": row.get("source_id") or "",
                    "source_phrase": phrase,
                    "match_score": score,
                    "match_method": "fuzzy_contains",
                })

    return matched


async def _expand_subgraph(
    graph_driver,
    seed_entity_ids: list[str],
    hops: int,
) -> tuple[list[dict], list[dict]]:
    """Pull the subgraph centered on the seed nodes out to `hops` steps.

    Two queries:
      A. Collect node entity_ids reachable within `hops` from the seeds.
      B. Pull edges whose endpoints are both in that node set.

    Splitting like this avoids a Memgraph quirk where collecting relationships
    via path comprehension and then reading `r.description` in a list
    comprehension returned the same description for every edge.

    Returns (nodes, edges). Each node dict has {id, type, description, source_id};
    each edge dict has {source, target, description, keywords, weight}.
    Both lists are deduped.
    """
    if not seed_entity_ids:
        return [], []

    nodes_cypher = (
        _SUBGRAPH_NODES_CYPHER_0HOP if hops <= 0
        else _SUBGRAPH_NODES_CYPHER_NHOP % hops
    )

    nodes_out: list[dict] = []
    seen_nodes: set[str] = set()

    async with graph_driver.session() as session:
        # Step A: node set
        try:
            result = await session.run(nodes_cypher, seed_ids=seed_entity_ids)
            async for row in result:
                nid = row.get("id")
                if not nid or nid in seen_nodes:
                    continue
                seen_nodes.add(nid)
                nodes_out.append({
                    "id": nid,
                    "type": row.get("type") or "",
                    "description": row.get("description") or "",
                    "source_id": row.get("source_id") or "",
                })
        except Exception as e:
            logger.warning("subgraph nodes query failed (hops=%d): %s", hops, e)
            return [], []

        if not nodes_out:
            return [], []

        # Step B: edges among those nodes (skip when 0-hop with a single seed,
        # there can't be edges within a single-node set anyway)
        node_ids = list(seen_nodes)
        edges_out: list[dict] = []
        if len(node_ids) >= 2:
            try:
                result = await session.run(_SUBGRAPH_EDGES_CYPHER, node_ids=node_ids)
                async for row in result:
                    src = row.get("source")
                    tgt = row.get("target")
                    if not src or not tgt:
                        continue
                    edges_out.append({
                        "source": src,
                        "target": tgt,
                        "description": row.get("description") or "",
                        "keywords": row.get("keywords") or "",
                        "weight": float(row.get("weight") or 1.0),
                    })
            except Exception as e:
                logger.warning("subgraph edges query failed: %s", e)
                # nodes are already populated; return them with empty edges

    return nodes_out, edges_out


def _rank_nodes(
    nodes: list[dict],
    edges: list[dict],
    seed_ids: set[str],
    ranking: str,
) -> list[dict]:
    """Apply ranking to nodes. Always anchors seeds to the top regardless of ranking mode."""
    if not nodes:
        return nodes

    # Compute degree from edges (in + out, undirected)
    degree: dict[str, int] = {}
    for e in edges:
        degree[e["source"]] = degree.get(e["source"], 0) + 1
        degree[e["target"]] = degree.get(e["target"], 0) + 1

    # Attach degree to each node
    for n in nodes:
        n["_degree"] = degree.get(n["id"], 0)

    if ranking == "degree":
        # Seeds first (preserved order), then non-seeds sorted by degree desc.
        seeds = [n for n in nodes if n["id"] in seed_ids]
        rest = [n for n in nodes if n["id"] not in seed_ids]
        rest.sort(key=lambda n: (-n["_degree"], n["id"]))
        return seeds + rest

    # ranking == "none": preserve match order; seeds-first by definition since
    # the subgraph cypher returns seeds before neighbors when hops > 0.
    return nodes


async def _bridge_chunks(
    lightrag,
    pool: asyncpg.Pool,
    nodes: list[dict],
    top_k_chunks: int,
) -> tuple[list[dict], float]:
    """Resolve each node's source_id chunk hashes to full chunks via LightRAG + Postgres.

    The graph node carries `source_id = "chunk-<hash><SEP>chunk-<hash2>..."`.
    Each hash is a LightRAG-internal id; LightRAG's text_chunks KV store maps
    it to {full_doc_id="doc_<int>", chunk_order_index, content, ...}. We then
    map (document_id, chunk_index) -> our chunks.id so callers get the same
    chunk_id they'd see from /v1/search.

    Returns (chunks, bridge_ms).
    """
    t0 = time.perf_counter()

    # 1. Collect all chunk hashes mentioned across the subgraph, and remember
    #    which entity_ids point at each hash (for source_entities in the response).
    hash_to_entities: dict[str, set[str]] = {}
    for n in nodes:
        for h in _split_source_id(n.get("source_id") or ""):
            hash_to_entities.setdefault(h, set()).add(n["id"])

    if not hash_to_entities:
        return [], (time.perf_counter() - t0) * 1000

    if lightrag is None:
        # We can still return empty source-attribution payloads if LightRAG is down,
        # but we cannot resolve hash -> doc, so chunks come back empty.
        logger.warning("LightRAG unavailable; skipping chunk bridge")
        return [], (time.perf_counter() - t0) * 1000

    # 2. Lookup hashes in LightRAG text_chunks. The KV store may return None for
    #    unknown hashes; preserve order alignment with our hash list.
    hash_list = list(hash_to_entities.keys())
    try:
        chunk_records = await lightrag.text_chunks.get_by_ids(hash_list)
    except Exception as e:
        logger.warning("text_chunks.get_by_ids failed: %s", e)
        return [], (time.perf_counter() - t0) * 1000

    # 3. For each known record, collect (full_doc_id, chunk_order_index).
    #    Score chunks by the number of matched entities pointing at them so the
    #    rank output reflects "this chunk supports the most retrieved nodes".
    by_score: list[tuple[int, str, dict]] = []  # (-score, hash, lightrag_record)
    for h, record in zip(hash_list, chunk_records or []):
        if not record:
            continue
        n_supporting = len(hash_to_entities[h])
        by_score.append((-n_supporting, h, record))

    by_score.sort(key=lambda t: (t[0], t[1]))

    # 4. Bridge to our chunks.id via Postgres on (document_id, chunk_index).
    pairs: list[tuple[int, int]] = []
    flat: list[tuple[str, dict]] = []  # (hash, record) in chosen order
    for _neg_score, h, record in by_score[:top_k_chunks]:
        doc_id = _parse_doc_id(record.get("full_doc_id") if isinstance(record, dict) else None)
        chunk_index = record.get("chunk_order_index") if isinstance(record, dict) else None
        if doc_id is None or chunk_index is None:
            flat.append((h, record))
            continue
        pairs.append((doc_id, int(chunk_index)))
        flat.append((h, record))

    chunk_id_lookup: dict[tuple[int, int], int] = {}
    if pairs:
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, document_id, chunk_index
                FROM chunks
                WHERE (document_id, chunk_index) IN (
                    SELECT * FROM unnest($1::bigint[], $2::int[])
                )
                """,
                [p[0] for p in pairs],
                [p[1] for p in pairs],
            )
        for row in rows:
            chunk_id_lookup[(int(row["document_id"]), int(row["chunk_index"]))] = int(row["id"])

    # 5. Assemble response chunks
    out: list[dict] = []
    for h, record in flat:
        doc_id = _parse_doc_id(record.get("full_doc_id") if isinstance(record, dict) else None)
        chunk_index = record.get("chunk_order_index") if isinstance(record, dict) else None
        text = record.get("content") if isinstance(record, dict) else ""
        chunk_id = chunk_id_lookup.get((doc_id, int(chunk_index))) if doc_id is not None and chunk_index is not None else None
        out.append({
            "chunk_id": chunk_id,
            "doc_id": doc_id,
            "lightrag_chunk_hash": h,
            "text": str(text or ""),
            "source_entities": sorted(hash_to_entities[h]),
        })

    return out, (time.perf_counter() - t0) * 1000


async def graph_only_search(
    *,
    ner_service,
    graph_driver,
    lightrag,
    pool: asyncpg.Pool,
    query: str,
    max_entities: int,
    hops: int,
    ranking: str,
    top_k_chunks: int,
    fuzzy: bool,
    ner_labels: list[str] | None,
) -> dict:
    """Top-level orchestration. Read-only; never writes to any store.

    Returns a dict matching `GraphSearchResponse`.
    """
    overall_t0 = time.perf_counter()

    # 1. NER
    ner_t0 = time.perf_counter()
    ner_hits = await ner_service.extract(query, labels=ner_labels)
    ner_ms = (time.perf_counter() - ner_t0) * 1000

    # Cap to max_entities, prefer higher-confidence hits
    ner_hits = sorted(ner_hits, key=lambda h: -h["score"])[:max_entities]

    # 2. Graph match
    graph_t0 = time.perf_counter()
    matches = await _match_entities(graph_driver, ner_hits, fuzzy=fuzzy)
    seed_ids = list({m["id"] for m in matches})

    # 3. Subgraph expansion
    nodes, edges = await _expand_subgraph(graph_driver, seed_ids, hops=hops)
    graph_ms = (time.perf_counter() - graph_t0) * 1000

    # 4. Ranking
    seed_set = set(seed_ids)
    ranked_nodes = _rank_nodes(nodes, edges, seed_set, ranking)

    # 5. Chunk bridge
    chunks, bridge_ms = await _bridge_chunks(lightrag, pool, ranked_nodes, top_k_chunks)

    overall_ms = (time.perf_counter() - overall_t0) * 1000

    return {
        "query": query,
        "matched_entities": [
            {
                "id": m["id"],
                "name": m["id"],
                "type": m["type"],
                "match_score": m["match_score"],
                "match_method": m["match_method"],
                "source_phrase": m["source_phrase"],
            }
            for m in matches
        ],
        "subgraph": {
            "nodes": [
                {
                    "id": n["id"],
                    "name": n["id"],
                    "type": n["type"],
                    "description": n.get("description") or None,
                    "degree": int(n.get("_degree", 0)),
                }
                for n in ranked_nodes
            ],
            "edges": [
                {
                    "source": e["source"],
                    "target": e["target"],
                    "relation": e["description"],
                    "weight": float(e["weight"]),
                    "keywords": e.get("keywords") or None,
                }
                for e in edges
            ],
        },
        "chunks": chunks,
        "trace": {
            "latency_ms": round(overall_ms, 2),
            "ner_ms": round(ner_ms, 2),
            "graph_ms": round(graph_ms, 2),
            "chunk_bridge_ms": round(bridge_ms, 2),
            "n_ner_entities": len(ner_hits),
            "n_matched_nodes": len(seed_ids),
            "n_subgraph_nodes": len(ranked_nodes),
            "n_subgraph_edges": len(edges),
        },
    }
