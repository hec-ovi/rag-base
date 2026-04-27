"""Phase 4 integration validation runner.

Step-by-step retrieval evaluation against an adversarial corpus. Each step is
a single CLI invocation, ctrl-C-able, idempotent, and writes a JSONL row plus
a summary table to stdout.

Usage:
  python tests/integration_validation/run.py ingest          # load the corpus, no LightRAG
  python tests/integration_validation/run.py ingest-graph    # also feed multi-hop docs into LightRAG (slow)
  python tests/integration_validation/run.py step lexical    # query each tagged target via /v1/search/keyword
  python tests/integration_validation/run.py step semantic
  python tests/integration_validation/run.py step hybrid_norerank
  python tests/integration_validation/run.py step hybrid_rerank
  python tests/integration_validation/run.py step hybrid_graph_rerank
  python tests/integration_validation/run.py step header_ablation   # custom logic, see below
  python tests/integration_validation/run.py cleanup
  python tests/integration_validation/run.py status

Doc IDs are persisted across invocations in
tests/integration_validation/results/state.json so subsequent steps can hit
the same ingested corpus without re-uploading.
"""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tests.integration_validation.corpus import CORPUS  # noqa: E402
from tests.integration_validation.queries import QUERIES, queries_with_target  # noqa: E402

API_URL = os.environ.get("RAGBASE_API_URL", "http://localhost:5050")
RESULTS_DIR = Path(__file__).resolve().parent / "results"
STATE_FILE = RESULTS_DIR / "state.json"

# Multi-hop docs are the only ones we feed through LightRAG. Keeping the
# slow LLM ingest under three docs caps wall to about 30 minutes.
GRAPH_DOCS: tuple[str, ...] = ("HOP_RESEARCHER", "HOP_COMPANY", "HOP_CITY")


# ──────────────────────────────────────────────────────────────────────
# State (persists ingested doc ids across CLI invocations)
# ──────────────────────────────────────────────────────────────────────


def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"doc_ids": {}, "graph_ingested": []}


def save_state(state: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2))


# ──────────────────────────────────────────────────────────────────────
# Ingest
# ──────────────────────────────────────────────────────────────────────


async def cmd_ingest(args) -> None:
    state = load_state()
    async with httpx.AsyncClient(base_url=API_URL, timeout=60.0) as client:
        for key, spec in CORPUS.items():
            if key in state["doc_ids"]:
                print(f"  skip   {key:<16}  already ingested as id={state['doc_ids'][key]}")
                continue
            t0 = time.perf_counter()
            r = await client.post(
                "/v1/documents",
                json={
                    "title": spec["title"],
                    "content": spec["content"],
                    "metadata": spec["metadata"],
                },
                headers={"X-LightRAG-Ingest": "false"},
            )
            r.raise_for_status()
            doc_id = r.json()["id"]
            state["doc_ids"][key] = doc_id
            save_state(state)
            print(f"  ingest {key:<16}  id={doc_id}  ({time.perf_counter()-t0:.2f}s)")
    print(f"\n{len(state['doc_ids'])} docs in state.")


async def cmd_ingest_graph(args) -> None:
    """Re-ingest the multi-hop docs through LightRAG (slow, LLM-bound)."""
    state = load_state()
    async with httpx.AsyncClient(base_url=API_URL, timeout=1800.0) as client:
        for key in GRAPH_DOCS:
            if key in state["graph_ingested"]:
                print(f"  skip   {key:<16}  already in graph")
                continue
            spec = CORPUS[key]
            print(f"  graph  {key:<16}  posting (LLM entity extraction, ~5-10 min)...", flush=True)
            t0 = time.perf_counter()
            r = await client.post(
                "/v1/documents",
                json={
                    "title": spec["title"] + " (graph)",
                    "content": spec["content"],
                    "metadata": {**spec["metadata"], "graph_variant": True},
                },
                # NO X-LightRAG-Ingest header => LightRAG runs.
            )
            r.raise_for_status()
            doc_id = r.json()["id"]
            state["doc_ids"][f"{key}_GRAPH"] = doc_id
            state["graph_ingested"].append(key)
            save_state(state)
            print(f"  done   {key:<16}  id={doc_id}  ({time.perf_counter()-t0:.1f}s)")
    print(f"\n{len(state['graph_ingested'])}/{len(GRAPH_DOCS)} multi-hop docs in graph.")


async def cmd_cleanup(args) -> None:
    state = load_state()
    async with httpx.AsyncClient(base_url=API_URL, timeout=60.0) as client:
        for key, doc_id in list(state["doc_ids"].items()):
            try:
                await client.delete(f"/v1/documents/{doc_id}")
                print(f"  delete {key:<16}  id={doc_id}")
            except httpx.HTTPError as e:
                print(f"  warn   {key:<16}  id={doc_id} ({e})")
    state["doc_ids"] = {}
    state["graph_ingested"] = []
    save_state(state)


async def cmd_status(args) -> None:
    state = load_state()
    print(f"API: {API_URL}")
    print(f"Docs ingested: {len(state['doc_ids'])}")
    for key, doc_id in state["doc_ids"].items():
        print(f"  {key:<20} -> id={doc_id}")
    print(f"Graph-ingested: {state['graph_ingested']}")


# ──────────────────────────────────────────────────────────────────────
# Search
# ──────────────────────────────────────────────────────────────────────


async def _search(client: httpx.AsyncClient, channel: str, query: str, top_k: int = 10) -> dict:
    """Dispatch a query through the channel, return the parsed body."""
    if channel == "lexical":
        r = await client.post("/v1/search/keyword", json={"query": query, "top_k": top_k})
    elif channel == "semantic":
        r = await client.post("/v1/search/semantic", json={"query": query, "top_k": top_k})
    elif channel == "hybrid_norerank":
        r = await client.post(
            "/v1/search",
            json={"query": query, "top_k": top_k, "rerank": False, "include_graph": False},
        )
    elif channel == "hybrid_rerank":
        r = await client.post(
            "/v1/search",
            json={
                "query": query,
                "top_k": top_k,
                "rerank": True,
                "rerank_candidates": 50,
                "include_graph": False,
            },
        )
    elif channel == "hybrid_graph_rerank":
        r = await client.post(
            "/v1/search",
            json={
                "query": query,
                "top_k": top_k,
                "rerank": True,
                "rerank_candidates": 50,
                "include_graph": True,
            },
        )
    elif channel == "hybrid_graph_norerank":
        r = await client.post(
            "/v1/search",
            json={
                "query": query,
                "top_k": top_k,
                "rerank": False,
                "include_graph": True,
            },
        )
    elif channel == "graph_only":
        # /v1/search/graph returns a different shape (chunks[] with chunk_id + doc_id,
        # no document_id at the top level). Translate to the rank-eval shape so the
        # rest of the runner stays uniform.
        r = await client.post(
            "/v1/search/graph",
            json={"query": query, "top_k_chunks": top_k, "hops": 1, "fuzzy": True},
        )
        r.raise_for_status()
        body = r.json()
        return {
            "results": [
                {"document_id": c.get("doc_id"), "chunk_id": c.get("chunk_id")}
                for c in body.get("chunks", [])
            ],
            "retrievers_used": ["graph"],
            "trace": body.get("trace", {}),
        }
    else:
        raise ValueError(f"unknown channel: {channel}")
    r.raise_for_status()
    return r.json()


def _rank_of_target(results: list[dict], target_doc_id: int, also_accept: list[int] | None = None) -> int | None:
    """Return 1-based rank of target_doc_id in results, or None if not present.

    `also_accept` allows alternate doc ids to count as a hit. Used for graph
    channels where the corpus has both a non-graph copy (target_id) and a
    graph-ingested copy (target_GRAPH id) of the same content; the graph
    pipeline can only surface the variant whose entities were extracted.
    """
    accept = {target_doc_id, *(also_accept or [])}
    for i, hit in enumerate(results, start=1):
        if hit.get("document_id") in accept:
            return i
    return None


async def cmd_step(args) -> None:
    channel = args.channel
    state = load_state()
    if not state["doc_ids"]:
        print("ERROR: no docs ingested. Run `ingest` first.", file=sys.stderr)
        sys.exit(2)

    out_path = RESULTS_DIR / f"step_{channel}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("")  # truncate per run

    if channel == "header_ablation":
        await _step_header_ablation(state, out_path)
        return

    rows: list[dict] = []
    async with httpx.AsyncClient(base_url=API_URL, timeout=120.0) as client:
        for q in QUERIES:
            if not q.get("target"):
                continue
            target_key = q["target"]
            target_id = state["doc_ids"].get(target_key)
            if target_id is None:
                print(f"  skip   {q['text'][:60]:<60}  target {target_key} not ingested")
                continue

            t0 = time.perf_counter()
            body = await _search(client, channel, q["text"], top_k=10)
            dt_ms = (time.perf_counter() - t0) * 1000
            results = body.get("results", [])
            # For graph-driven channels, the graph-ingested copy ("<key>_GRAPH")
            # is the one whose entities are actually in Memgraph; accept it too.
            also_accept: list[int] = []
            if channel in ("graph_only", "hybrid_graph_rerank", "hybrid_graph_norerank"):
                graph_id = state["doc_ids"].get(f"{target_key}_GRAPH")
                if graph_id is not None:
                    also_accept.append(graph_id)
            rank = _rank_of_target(results, target_id, also_accept=also_accept)
            row = {
                "channel": channel,
                "scenario": q["scenario"],
                "query": q["text"],
                "target": target_key,
                "target_id": target_id,
                "rank": rank,
                "favors": q["favors"],
                "latency_ms": round(dt_ms, 1),
                "top_doc_ids": [h.get("document_id") for h in results[:3]],
            }
            rows.append(row)
            with out_path.open("a") as f:
                f.write(json.dumps(row) + "\n")

    _print_table(channel, rows)


async def _step_header_ablation(state: dict, out_path: Path) -> None:
    """Custom step: same body content under different ingest variants.

    For each header_path query, we want to see whether HDR_WITH (markdown
    headers intact -> breadcrumb in indexed_content) outranks HDR_WITHOUT
    (no markdown headers -> no breadcrumb).
    """
    channel = "hybrid_rerank"  # fairest comparator: full stack
    rows: list[dict] = []
    async with httpx.AsyncClient(base_url=API_URL, timeout=120.0) as client:
        for q in QUERIES:
            if q["scenario"] != "header_path":
                continue
            with_id = state["doc_ids"].get("HDR_WITH")
            without_id = state["doc_ids"].get("HDR_WITHOUT")
            if not with_id or not without_id:
                print("  skip header_path: HDR_WITH/HDR_WITHOUT not both ingested")
                continue

            t0 = time.perf_counter()
            body = await _search(client, channel, q["text"], top_k=20)
            dt_ms = (time.perf_counter() - t0) * 1000
            results = body.get("results", [])
            rank_with = _rank_of_target(results, with_id)
            rank_without = _rank_of_target(results, without_id)
            row = {
                "channel": "header_ablation",
                "scenario": q["scenario"],
                "query": q["text"],
                "rank_HDR_WITH": rank_with,
                "rank_HDR_WITHOUT": rank_without,
                "latency_ms": round(dt_ms, 1),
                "top_doc_ids": [h.get("document_id") for h in results[:5]],
            }
            rows.append(row)
            with out_path.open("a") as f:
                f.write(json.dumps(row) + "\n")
    _print_header_ablation(rows)


# ──────────────────────────────────────────────────────────────────────
# Reporting
# ──────────────────────────────────────────────────────────────────────


def _print_table(channel: str, rows: list[dict]) -> None:
    if not rows:
        print(f"\n[{channel}] no rows")
        return
    print(f"\n=== Channel: {channel} ===")
    fmt = "{:<14} {:<60} {:>5} {:>10} {}"
    print(fmt.format("scenario", "query", "rank", "lat_ms", "favors"))
    print("-" * 110)
    hits, ranks = 0, []
    for r in rows:
        rank_str = str(r["rank"]) if r["rank"] is not None else "miss"
        favors = ",".join(r["favors"])
        print(fmt.format(r["scenario"], r["query"][:60], rank_str, r["latency_ms"], favors))
        if r["rank"] is not None:
            hits += 1
            ranks.append(r["rank"])
    n = len(rows)
    hit5 = sum(1 for r in rows if r["rank"] is not None and r["rank"] <= 5) / n
    hit1 = sum(1 for r in rows if r["rank"] == 1) / n
    mrr = sum(1 / r["rank"] for r in rows if r["rank"] is not None) / n
    print("-" * 110)
    print(f"hit@1={hit1:.2f}  hit@5={hit5:.2f}  mrr={mrr:.3f}  n={n}  found={hits}/{n}")


def _print_header_ablation(rows: list[dict]) -> None:
    if not rows:
        print("\n[header_ablation] no rows")
        return
    print("\n=== Channel: header_ablation (hybrid_rerank backend) ===")
    fmt = "{:<60} {:>10} {:>14} {:>10}"
    print(fmt.format("query", "HDR_WITH", "HDR_WITHOUT", "lat_ms"))
    print("-" * 100)
    for r in rows:
        rw = str(r["rank_HDR_WITH"]) if r["rank_HDR_WITH"] is not None else "miss"
        rwo = str(r["rank_HDR_WITHOUT"]) if r["rank_HDR_WITHOUT"] is not None else "miss"
        print(fmt.format(r["query"][:60], rw, rwo, r["latency_ms"]))


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("ingest", help="ingest the corpus (no LightRAG)")
    sub.add_parser("ingest-graph", help="also feed multi-hop docs into LightRAG")
    sub.add_parser("cleanup", help="delete every ingested doc")
    sub.add_parser("status", help="show ingest state")

    sp = sub.add_parser("step", help="run one evaluation step")
    sp.add_argument(
        "channel",
        choices=[
            "lexical",
            "semantic",
            "hybrid_norerank",
            "hybrid_rerank",
            "hybrid_graph_rerank",
            "hybrid_graph_norerank",
            "graph_only",
            "header_ablation",
        ],
    )

    args = p.parse_args()

    handlers = {
        "ingest": cmd_ingest,
        "ingest-graph": cmd_ingest_graph,
        "cleanup": cmd_cleanup,
        "status": cmd_status,
        "step": cmd_step,
    }
    asyncio.run(handlers[args.cmd](args))


if __name__ == "__main__":
    main()
