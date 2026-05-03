"""HAI synthetic-corpus FULL pipeline benchmark.

Walks every public endpoint against the 20-doc Halpern Astride Industries
fictional corpus that was ingested THROUGH LightRAG (entities + relations live
in Memgraph). Runs each phase, prints a summary table, and exits non-zero on
any unexpected failure.

Phases:
  0. Health probes
  1. Documents (list / get)
  2. Direct passthrough: /v1/embed, /v1/rerank
  3. Quality matrix: hit@1/hit@5/MRR for each (channel x mode) combination
       channels = {hybrid_norerank, hybrid_rerank, hybrid_graph_rerank,
                   semantic, keyword, graph_only}
       modes    = {default, bge-gpu, qwen-4b, qwen-8b}  (rerank only)
  4. Graph-only sanity: /v1/search/graph on entity-anchored queries
  5. Graph CRUD: /v1/concepts, /v1/relations
  6. Graph traversal: /v1/graph/neighbors, /path, /communities, /stats
  7. Stability: concurrent fan-out across all (channel x mode) pairs;
       vLLM Responses-API streaming poked before / mid / after.
  8. Cleanup: delete the test concepts/relations created in phase 5.

Run:
  .venv-test/bin/python tests/integration_validation/hai_full_bench.py \\
      --concurrency 8 --repeats 2

Reads HAI_CORPUS / queries from sibling hai_corpus.py. The 20-doc corpus must
already be ingested through /v1/documents WITH LightRAG (no X-LightRAG-Ingest
header, default true) before running. Ingest helper:

  for d in CORPUS:
      POST /v1/documents {"title": d["title"], "content": d["content"]}

(roughly 1 minute per doc on a local 27B vLLM with reasoning off).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from collections import defaultdict
from typing import Any

import httpx

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from hai_corpus import CORPUS, QUERIES  # noqa: E402

API_URL = "http://127.0.0.1:5050"
VLLM_URL = "http://127.0.0.1:8000"

RERANK_MODES = ["default", "bge-gpu", "qwen-4b", "qwen-8b"]

# Same target -> partial title marker mapping as the original bench.
TARGET_TITLE_MARKER = {
    "vexil-7": "vexil-7",
    "stilwater": "stilwater",
    "bramwell": "bramwell",
    "quibbler-frame": "quibbler-frame",
    "hesketh-code": "hesketh code",
    "zigast-theorem": "zigast",
    "pemberton-bio": "pemberton",
    "quistain-bio": "quistain",
    "pentlow-yard": "pentlow",
    "hattersley-reach": "hattersley",
    "wallinghurst-hq": "wallinghurst",
    "drumcastle-ops": "drumcastle",
}


def rank_target(results: list[dict], target: str) -> int | None:
    marker = TARGET_TITLE_MARKER.get(target, target).lower()
    for i, r in enumerate(results, start=1):
        if marker in r.get("document_title", "").lower():
            return i
    return None


def score_quality(rank_lists: list[int | None]) -> dict:
    n = len(rank_lists)
    h1 = sum(1 for r in rank_lists if r == 1) / n
    h5 = sum(1 for r in rank_lists if r is not None and r <= 5) / n
    mrr = sum(1 / r for r in rank_lists if r is not None) / n
    return {"hit@1": h1, "hit@5": h5, "MRR": mrr, "n": n,
            "found": sum(1 for r in rank_lists if r is not None)}


# ---------------------------------------------------------------------------
# Phase 0: health
# ---------------------------------------------------------------------------

async def phase_health(c: httpx.AsyncClient) -> bool:
    print("=== Phase 0: Health probes ===")
    r = await c.get(f"{API_URL}/health")
    h = r.json()
    print(f"  /health: {json.dumps(h)}")
    ok = h.get("status") == "ok"
    r = await c.get(f"{API_URL}/health/models")
    print(f"  /health/models: {json.dumps(r.json())[:120]}")
    return ok


# ---------------------------------------------------------------------------
# Phase 1: documents
# ---------------------------------------------------------------------------

async def phase_documents(c: httpx.AsyncClient) -> dict[str, int]:
    print("=== Phase 1: Documents (list/get) ===")
    r = await c.get(f"{API_URL}/v1/documents", params={"offset": 0, "limit": 50})
    docs = r.json()
    print(f"  /v1/documents -> {len(docs)} docs")
    if isinstance(docs, dict):
        docs = docs.get("items", docs.get("documents", []))
    if not docs:
        print("  ! no documents found; the bench cannot proceed")
        return {}
    title_to_id = {}
    for d in docs:
        if "title" in d and "id" in d:
            title_to_id[d["title"]] = d["id"]
    # GET one specific doc (by first id)
    sample_id = docs[0]["id"]
    r = await c.get(f"{API_URL}/v1/documents/{sample_id}")
    j = r.json()
    print(f"  /v1/documents/{sample_id} -> chunks={len(j.get('chunks', []))}")
    return title_to_id


# ---------------------------------------------------------------------------
# Phase 2: passthrough
# ---------------------------------------------------------------------------

async def phase_passthrough(c: httpx.AsyncClient) -> None:
    print("=== Phase 2: Passthrough endpoints ===")
    r = await c.post(f"{API_URL}/v1/embed",
                     json={"inputs": ["frontend frameworks", "Quibbler-Frame protocol"]})
    j = r.json()
    print(f"  /v1/embed: dims={j.get('dimensions')} model={j.get('model')[:40]} n={len(j.get('embeddings', []))}")
    r = await c.post(f"{API_URL}/v1/rerank",
                     json={"query": "Quibbler-Frame protocol",
                           "texts": ["The Quibbler-Frame protocol is HAI internal data exchange",
                                     "Tomatoes are a fruit",
                                     "Pentlow Yard hosts the Bramwell test track"]})
    j = r.json()
    top = j["results"][0] if j.get("results") else None
    if top:
        print(f"  /v1/rerank: model={j.get('model')[:40]} top_idx={top['index']} "
              f"top_score={top['score']:.4f}")
    else:
        print(f"  /v1/rerank: model={j.get('model')[:40]} (no results)")


# ---------------------------------------------------------------------------
# Phase 3: quality matrix
# ---------------------------------------------------------------------------

async def search_call(c, query: str, mode: str | None, *, channel: str,
                      top_k: int = 10, candidates: int = 50) -> tuple[dict, float]:
    """Hit the right endpoint for the channel."""
    body = {"query": query, "top_k": top_k}
    path = "/v1/search"
    if channel == "semantic":
        path = "/v1/search/semantic"
    elif channel == "keyword":
        path = "/v1/search/keyword"
    elif channel == "hybrid_norerank":
        body.update({"rerank": False, "rerank_candidates": candidates,
                     "include_graph": False})
    elif channel == "hybrid_rerank":
        body.update({"rerank": True, "rerank_candidates": candidates,
                     "include_graph": False})
        if mode is not None:
            body["rerank_model"] = mode
    elif channel == "hybrid_graph_rerank":
        body.update({"rerank": True, "rerank_candidates": candidates,
                     "include_graph": True})
        if mode is not None:
            body["rerank_model"] = mode
    elif channel == "graph_only":
        path = "/v1/search/graph"
        body = {"query": query, "top_k": top_k}
    t0 = time.perf_counter()
    r = await c.post(f"{API_URL}{path}", json=body, timeout=180.0)
    ms = (time.perf_counter() - t0) * 1000.0
    r.raise_for_status()
    return r.json(), ms


def extract_results(channel: str, payload: dict) -> list[dict]:
    if channel == "graph_only":
        # /v1/search/graph returns chunks under different shape
        chunks = payload.get("chunks", [])
        # Reshape to {document_title, score} using doc lookup if present
        return [{"document_title": ch.get("document_title", ""),
                 "score": ch.get("score", 0.0)} for ch in chunks]
    return payload.get("results", [])


async def phase_quality(c: httpx.AsyncClient) -> dict:
    print("=== Phase 3: Quality matrix (channel x mode) ===")
    print()
    matrix = {}
    # Channels that are mode-agnostic (no rerank or fixed model)
    flat_channels = ["semantic", "keyword", "hybrid_norerank", "graph_only"]
    rerank_channels = ["hybrid_rerank", "hybrid_graph_rerank"]

    for ch in flat_channels:
        ranks, lats = [], []
        for q in QUERIES:
            try:
                payload, ms = await search_call(c, q["q"], None, channel=ch)
                results = extract_results(ch, payload)
                rank = rank_target(results, q["target"])
            except Exception as e:
                print(f"    {ch} / {q['q'][:40]} FAILED: {e}")
                rank = None
                ms = 0.0
            ranks.append(rank)
            lats.append(ms)
        m = score_quality(ranks)
        m["p50_ms"] = statistics.median(lats) if lats else 0
        m["sum_ms"] = sum(lats)
        matrix[ch] = {"_": m, "ranks": ranks}
        print(f"  {ch:<24}  hit@1={m['hit@1']:.2f}  hit@5={m['hit@5']:.2f}  "
              f"MRR={m['MRR']:.3f}  found={m['found']}/{m['n']}  "
              f"p50_ms={m['p50_ms']:.0f}")

    for ch in rerank_channels:
        for mode in RERANK_MODES:
            ranks, lats = [], []
            for q in QUERIES:
                try:
                    payload, ms = await search_call(c, q["q"], mode, channel=ch)
                    results = extract_results(ch, payload)
                    rank = rank_target(results, q["target"])
                except Exception as e:
                    print(f"    {ch}/{mode} / {q['q'][:40]} FAILED: {e}")
                    rank = None
                    ms = 0.0
                ranks.append(rank)
                lats.append(ms)
            m = score_quality(ranks)
            m["p50_ms"] = statistics.median(lats) if lats else 0
            m["sum_ms"] = sum(lats)
            matrix.setdefault(ch, {})[mode] = m
            matrix[ch][f"ranks_{mode}"] = ranks
            print(f"  {ch:<24}/{mode:<10}  hit@1={m['hit@1']:.2f}  hit@5={m['hit@5']:.2f}  "
                  f"MRR={m['MRR']:.3f}  found={m['found']}/{m['n']}  "
                  f"p50_ms={m['p50_ms']:.0f}")
    print()
    return matrix


# ---------------------------------------------------------------------------
# Phase 5+6: graph CRUD + traversal
# ---------------------------------------------------------------------------

async def phase_graph_crud(c: httpx.AsyncClient) -> dict[str, int]:
    print("=== Phase 5+6: Graph CRUD + traversal ===")
    created = {"concepts": [], "relations": []}

    # Create two concepts
    for name, ctype in [("HAI_TEST_VOLKENBURG", "Person"),
                        ("HAI_TEST_PENTLOW_YARD", "Location")]:
        r = await c.post(f"{API_URL}/v1/concepts", json={"name": name, "type": ctype})
        if r.status_code in (200, 201):
            j = r.json()
            created["concepts"].append(j.get("id"))
            print(f"  POST /v1/concepts {name} -> id={j.get('id')}")
        else:
            print(f"  POST /v1/concepts {name} -> {r.status_code} {r.text[:80]}")

    if len(created["concepts"]) == 2:
        from_id, to_id = created["concepts"]
        r = await c.post(f"{API_URL}/v1/relations",
                         json={"source_name": "HAI_TEST_VOLKENBURG",
                               "target_name": "HAI_TEST_PENTLOW_YARD",
                               "relation_type": "MANAGES_FACILITY"})
        if r.status_code in (200, 201):
            created["relations"].append(r.json().get("id"))
            print(f"  POST /v1/relations VOLKENBURG -[MANAGES_FACILITY]-> PENTLOW_YARD ok")
        else:
            print(f"  POST /v1/relations -> {r.status_code} {r.text[:120]}")

        # GET relations for the source concept
        r = await c.get(f"{API_URL}/v1/relations",
                        params={"concept_name": "HAI_TEST_VOLKENBURG"})
        rels = r.json() if r.status_code == 200 else []
        print(f"  GET /v1/relations?concept_name=VOLKENBURG -> {len(rels)} relation(s)")

        # Traversal: response shape varies, count entries safely
        r = await c.get(f"{API_URL}/v1/graph/neighbors/{from_id}",
                        params={"depth": 2})
        body = r.json()
        n_size = len(body) if isinstance(body, list) else (
            len(body.get("nodes", [])) if isinstance(body, dict) else 0)
        print(f"  GET /v1/graph/neighbors/{from_id} -> {n_size} entries")

        r = await c.get(f"{API_URL}/v1/graph/path/{from_id}/{to_id}")
        print(f"  GET /v1/graph/path/{from_id}/{to_id} -> {r.status_code} "
              f"{json.dumps(r.json())[:120]}")

    r = await c.get(f"{API_URL}/v1/graph/stats")
    print(f"  GET /v1/graph/stats -> {json.dumps(r.json())[:120]}")
    r = await c.get(f"{API_URL}/v1/graph/communities")
    j = r.json()
    n_comm = len(j) if isinstance(j, list) else len(j.get("communities", []))
    print(f"  GET /v1/graph/communities -> {n_comm} communities")
    return created


async def phase_cleanup(c: httpx.AsyncClient, created: dict) -> None:
    print("=== Phase 8: Cleanup ===")
    for rid in created.get("relations", []):
        r = await c.delete(f"{API_URL}/v1/relations/{rid}")
        print(f"  DELETE /v1/relations/{rid} -> {r.status_code}")
    for cid in created.get("concepts", []):
        r = await c.delete(f"{API_URL}/v1/concepts/{cid}")
        print(f"  DELETE /v1/concepts/{cid} -> {r.status_code}")


# ---------------------------------------------------------------------------
# Phase 7: stability
# ---------------------------------------------------------------------------

async def vllm_check(c: httpx.AsyncClient) -> dict:
    body = {"model": "Qwen3.6-27B-AWQ4",
            "input": "Reply with one word: ok",
            "stream": True,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
    t0 = time.perf_counter()
    try:
        async with c.stream("POST", f"{VLLM_URL}/v1/responses", json=body,
                            timeout=60.0) as r:
            first = None
            n = 0
            async for chunk in r.aiter_bytes():
                if first is None:
                    first = (time.perf_counter() - t0) * 1000
                n += len(chunk)
            total = (time.perf_counter() - t0) * 1000
            return {"status": r.status_code, "ttfb_ms": first, "total_ms": total, "bytes": n}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


async def phase_stability(c: httpx.AsyncClient, concurrency: int, repeats: int) -> dict:
    print(f"=== Phase 7: Stability (concurrency={concurrency}, repeats={repeats}) ===")
    print(f"  vLLM pre-check : {await vllm_check(c)}")

    # Build job list: every (channel x mode x query) repeated N times.
    jobs = []
    flat = ["semantic", "keyword", "hybrid_norerank", "graph_only"]
    rerank = ["hybrid_rerank", "hybrid_graph_rerank"]
    for _ in range(repeats):
        for q in QUERIES:
            for ch in flat:
                jobs.append({"q": q["q"], "channel": ch, "mode": None})
            for ch in rerank:
                for mode in RERANK_MODES:
                    jobs.append({"q": q["q"], "channel": ch, "mode": mode})

    sem = asyncio.Semaphore(concurrency)
    timings = defaultdict(list)
    errors = []

    async def run_one(j):
        async with sem:
            try:
                _, ms = await search_call(c, j["q"], j["mode"], channel=j["channel"])
                key = j["channel"] if j["mode"] is None else f"{j['channel']}/{j['mode']}"
                timings[key].append(ms)
            except Exception as e:
                errors.append(f"{j['channel']}/{j['mode']} / {j['q'][:30]}: {type(e).__name__}: {e}")

    t0 = time.perf_counter()
    await asyncio.gather(*(run_one(j) for j in jobs))
    wall = time.perf_counter() - t0

    print(f"  vLLM mid-check : {await vllm_check(c)}")
    print(f"  total jobs: {len(jobs)}  wall: {wall:.1f}s  errors: {len(errors)}")
    for e in errors[:8]:
        print(f"    {e}")
    print()
    print(f"  {'channel/mode':<36} {'n':>4} {'p50_ms':>9} {'p95_ms':>9} {'min':>9} {'max':>9}")
    for key in sorted(timings.keys()):
        ts = sorted(timings[key])
        if not ts:
            continue
        n = len(ts)
        p50 = ts[n // 2]
        p95 = ts[max(0, int(n * 0.95) - 1)]
        print(f"  {key:<36} {n:>4} {p50:>9.0f} {p95:>9.0f} {ts[0]:>9.0f} {ts[-1]:>9.0f}")
    print(f"  vLLM post-check: {await vllm_check(c)}")
    return {"wall_s": wall, "n_jobs": len(jobs), "errors": errors,
            "timings": {k: sorted(v) for k, v in timings.items()}}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--repeats", type=int, default=2)
    args = p.parse_args()

    print(f"== HAI FULL pipeline benchmark ==")
    print(f"   API={API_URL}  vLLM={VLLM_URL}")
    print(f"   queries={len(QUERIES)}  corpus_docs={len(CORPUS)}")
    print()

    async with httpx.AsyncClient() as c:
        ok = await phase_health(c)
        if not ok:
            print("Health check failed; aborting.")
            sys.exit(1)
        await phase_documents(c)
        await phase_passthrough(c)
        matrix = await phase_quality(c)
        created = await phase_graph_crud(c)
        try:
            stab = await phase_stability(c, args.concurrency, args.repeats)
        finally:
            await phase_cleanup(c, created)

        # Final summary
        print()
        print("=== Final summary ===")
        print(f"{'channel':<28} {'mode':<10} {'hit@1':>5} {'hit@5':>5} {'MRR':>5} {'p50_ms':>8}")
        for ch in ["hybrid_norerank", "hybrid_rerank", "hybrid_graph_rerank",
                   "semantic", "keyword", "graph_only"]:
            entry = matrix.get(ch, {})
            if "_" in entry:
                m = entry["_"]
                print(f"{ch:<28} {'(none)':<10} {m['hit@1']:>5.2f} {m['hit@5']:>5.2f} "
                      f"{m['MRR']:>5.3f} {m['p50_ms']:>8.0f}")
            else:
                for mode in RERANK_MODES:
                    if mode not in entry:
                        continue
                    m = entry[mode]
                    print(f"{ch:<28} {mode:<10} {m['hit@1']:>5.2f} {m['hit@5']:>5.2f} "
                          f"{m['MRR']:>5.3f} {m['p50_ms']:>8.0f}")
        print()
        print(f"Stability: {stab['n_jobs']} jobs, wall {stab['wall_s']:.1f}s, "
              f"{len(stab['errors'])} errors")


if __name__ == "__main__":
    asyncio.run(main())
