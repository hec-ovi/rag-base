"""Run the Phase 1 smoke set against rag-base and record baseline metrics.

Output: appends one row PER CHANNEL to tests/golden/eval_history.jsonl.

This is NOT the golden-set eval (see tests/golden/CURATION.md). It is a small
smoke set sized to give Phase 2 a baseline to beat.

Channels measured:
  - semantic       (POST /v1/search/semantic)
  - keyword        (POST /v1/search/keyword)
  - hybrid_norerank(POST /v1/search rerank=False include_graph=False)
  - hybrid_rerank  (POST /v1/search rerank=True include_graph=False, candidates=50)

Metrics per channel:
  - hit@1, hit@5, hit@10  (was the expected doc returned at or before this rank)
  - mrr                   (mean reciprocal rank of the expected doc, 0 if not in top 10)
  - latency_p50, p95, mean (milliseconds)
  - n_queries

Usage:
  python tests/baseline/run_baseline.py [--keep-docs]
"""

import argparse
import asyncio
import json
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_SET = REPO_ROOT / "tests" / "baseline" / "smoke_set.json"
HISTORY = REPO_ROOT / "tests" / "golden" / "eval_history.jsonl"
API_URL = "http://localhost:5050"


def percentile(values, p):
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * p / 100
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


async def ingest_docs(client: httpx.AsyncClient, docs: list[dict], with_graph: bool = False) -> dict[str, int]:
    """Ingest each smoke doc, return {key: doc_id}.

    By default sends `X-LightRAG-Ingest: false` so ingest is fast (~1 sec per doc
    instead of ~9 min on the local reasoning LLM). The non-graph channels
    (semantic, keyword, hybrid_norerank, hybrid_rerank) produce real numbers
    either way; only `hybrid_graph_rerank` depends on graph data, and that row
    is irrelevant for the Phase 3a/3b/3c comparisons (none of those swaps
    touch the graph channel). Pass `with_graph=True` when measuring graph
    quality specifically.
    """
    headers = {} if with_graph else {"X-LightRAG-Ingest": "false"}
    key_to_id = {}
    for d in docs:
        r = await client.post(
            "/v1/documents",
            json={"title": d["title"], "content": d["content"], "metadata": d.get("metadata", {})},
            headers=headers,
        )
        r.raise_for_status()
        key_to_id[d["key"]] = r.json()["id"]
    return key_to_id


async def cleanup_docs(client: httpx.AsyncClient, ids: list[int]):
    for doc_id in ids:
        try:
            await client.delete(f"/v1/documents/{doc_id}")
        except httpx.HTTPError:
            pass


async def query_channel(client: httpx.AsyncClient, channel: str, query: str) -> tuple[list[int], float]:
    """Issue a query on the named channel, return (ranked doc_ids, latency_ms)."""
    if channel == "semantic":
        path, body = "/v1/search/semantic", {"query": query, "top_k": 10, "min_score": 0.0}
    elif channel == "keyword":
        path, body = "/v1/search/keyword", {"query": query, "top_k": 10}
    elif channel == "hybrid_norerank":
        path, body = "/v1/search", {"query": query, "top_k": 10, "rerank": False, "include_graph": False}
    elif channel == "hybrid_rerank":
        path, body = "/v1/search", {"query": query, "top_k": 10, "rerank": True, "rerank_candidates": 50, "include_graph": False}
    elif channel == "hybrid_graph_rerank":
        path, body = "/v1/search", {"query": query, "top_k": 10, "rerank": True, "rerank_candidates": 50, "include_graph": True}
    else:
        raise ValueError(channel)

    t0 = time.perf_counter()
    r = await client.post(path, json=body)
    dt_ms = (time.perf_counter() - t0) * 1000.0
    r.raise_for_status()
    body = r.json()
    return [item["document_id"] for item in body["results"]], dt_ms


def score_query(ranked_doc_ids: list[int], expected_id: int) -> tuple[int, int, int, float]:
    """Return (hit_at_1, hit_at_5, hit_at_10, reciprocal_rank)."""
    rank = next((i + 1 for i, did in enumerate(ranked_doc_ids) if did == expected_id), None)
    if rank is None:
        return 0, 0, 0, 0.0
    return (
        1 if rank <= 1 else 0,
        1 if rank <= 5 else 0,
        1 if rank <= 10 else 0,
        1.0 / rank,
    )


async def run_baseline(keep_docs: bool, phase: str, note_override: str | None = None, with_graph: bool = False):
    smoke = json.loads(SMOKE_SET.read_text())
    docs, queries = smoke["docs"], smoke["queries"]

    async with httpx.AsyncClient(base_url=API_URL, timeout=600.0) as client:
        h = (await client.get("/health")).json()
        if h.get("status") != "ok":
            print(f"API not healthy: {h}", file=sys.stderr)
            sys.exit(2)

        mode = "with-graph" if with_graph else "skip-graph (fast)"
        print(f"Ingesting {len(docs)} smoke docs [{mode}]...")
        key_to_id = await ingest_docs(client, docs, with_graph=with_graph)

        try:
            # When --with-graph is off, skip hybrid_graph_rerank entirely.
            # Reasons:
            #   1. The graph is empty (we skipped LightRAG ingest), so the channel
            #      would degenerate to hybrid_rerank with extra noise.
            #   2. The query path still calls the LLM for entity extraction even
            #      with an empty graph, which adds 5-30s per query and can hang.
            channels = ["semantic", "keyword", "hybrid_norerank", "hybrid_rerank"]
            if with_graph:
                channels.append("hybrid_graph_rerank")
            ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
            commit = git_commit()
            rows = []

            for channel in channels:
                hit1 = hit5 = hit10 = 0
                rr_sum = 0.0
                latencies = []
                for q in queries:
                    expected_id = key_to_id[q["expected_key"]]
                    ranked_ids, dt = await query_channel(client, channel, q["query"])
                    h1, h5, h10, rr = score_query(ranked_ids, expected_id)
                    hit1 += h1
                    hit5 += h5
                    hit10 += h10
                    rr_sum += rr
                    latencies.append(dt)

                n = len(queries)
                row = {
                    "ts": ts,
                    "phase": phase,
                    "commit": commit,
                    "channel": channel,
                    "n_queries": n,
                    "hit_at_1": round(hit1 / n, 4),
                    "hit_at_5": round(hit5 / n, 4),
                    "hit_at_10": round(hit10 / n, 4),
                    "mrr": round(rr_sum / n, 4),
                    "latency_ms_mean": round(statistics.mean(latencies), 2),
                    "latency_ms_p50": round(percentile(latencies, 50), 2),
                    "latency_ms_p95": round(percentile(latencies, 95), 2),
                    "embedding_model": "BAAI/bge-m3",
                    "reranker_model": "BAAI/bge-reranker-v2-m3",
                    "smoke_set_version": smoke["schema_version"],
                    "note": note_override or f"phase-{phase} smoke baseline",
                }
                rows.append(row)
                print(
                    f"{channel:18s}  "
                    f"hit@1={row['hit_at_1']:.2f}  hit@5={row['hit_at_5']:.2f}  "
                    f"hit@10={row['hit_at_10']:.2f}  mrr={row['mrr']:.3f}  "
                    f"p50={row['latency_ms_p50']:.0f}ms  p95={row['latency_ms_p95']:.0f}ms"
                )

            HISTORY.parent.mkdir(parents=True, exist_ok=True)
            with HISTORY.open("a") as f:
                for row in rows:
                    f.write(json.dumps(row) + "\n")
            print(f"\nAppended {len(rows)} rows to {HISTORY.relative_to(REPO_ROOT)}")

        finally:
            if not keep_docs:
                await cleanup_docs(client, list(key_to_id.values()))
                print(f"Cleaned up {len(key_to_id)} smoke docs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--keep-docs", action="store_true", help="leave smoke docs in DB after run")
    parser.add_argument("--phase", default="1", help="phase tag for the eval_history row")
    parser.add_argument("--note", default=None, help="override note field on the row")
    parser.add_argument(
        "--with-graph",
        action="store_true",
        help="run real LightRAG ingest (~9 min per doc) so hybrid_graph_rerank gets a meaningful row. Default skips it for ~1 sec ingests.",
    )
    args = parser.parse_args()
    asyncio.run(run_baseline(args.keep_docs, args.phase, args.note, args.with_graph))
