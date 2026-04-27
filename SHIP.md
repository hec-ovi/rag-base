# rag-base ship report

**Date:** 2026-04-27
**Branch:** main
**Engine state:** Phase 5 closed, full audition pass complete.

This is the close-out report for rag-base as a standalone retrieval engine. After this ship, work shifts to the `knowledge-base` wrapper layer (Anthropic Contextual Retrieval at ingest, Karpathy LLM-Wiki, MemPalace supersession, MCP tool surface). The engine itself is feature-complete for the agreed scope.

---

## Audition pass summary

Performed in four steps, in order:

1. **Wipe.** Removed the `rag-base-api` docker image. Wiped `./data/postgres`, `./data/memgraph`, `./data/lightrag` to bare empty dirs (with the memgraph UID 101 chown re-applied). Started from a clean slate so the rest of the audit measures a freshly-built system, not warm caches.

2. **Codebase audit.** Walked every source file, every README, every config. Found and fixed:
   - `llm.txt` was Phase 1 era. Rewrote to current state (right postgres image, BM25 not tsvector, bind mounts not named volumes, `/v1/search/graph`, LightRAG, GLiNER, Phase 3c columns, LLM env section).
   - `.env` had `LLM_REASONING_EFFORT=minimal` (Phase 4.0 swap missed). Changed to `LLM_ENABLE_THINKING=false`. Removed stale `PGDATA_VOLUME` / `MEMGRAPH_VOLUME` named-volume entries.
   - `.env.template` had the same stale named-volume block and was missing all `LLM_*` vars. Rewrote with current state and a clear first-boot recipe.
   - `memgraph/README.md` claimed data lived in named volume `ragbase_mgdata`. Updated to bind mount with the UID 101 chown gotcha noted.
   - `postgres/README.md` had the BM25 index documented on `chunks.content`; actual is `chunks.indexed_content`. Fixed.
   - `tests/golden/CURATION.md` referenced the obsolete `tsv tsvector` column. Updated to `indexed_content` with the migration history noted.
   - `api/src/routers/search.py` had an unused `from src.config import settings`. Removed.
   - `api/src/services/llm_responses.py` defined `make_lightrag_llm_func` that nothing imported (the version in `lightrag_store.py` is the one actually wired in). Removed the dead duplicate.
   - `README.md` configuration section was missing `LLM_*` env vars. Added them with the 50x latency note. Persistence table was missing the LightRAG bind mount. Added it. The first-boot `mkdir` line did not include `./data/lightrag`. Added it.
   - All remaining `ts_rank` / `tsvector` mentions in the tree are intentional historical migration notes (in `init.sql` comment, `keyword_search.py` docstring, `README.md` Phase 3a swap row, `CURATION.md` schema note). Left as-is.

   Zero TODOs / FIXMEs / HACKs in shipped code (verified by grep).

3. **End-to-end retrieval matrix on a clean stack.** Built the api fresh, brought up `--profile rerank --profile graph`, ingested the integration_validation corpus (11 adversarial docs, 13 tagged queries), then ran every retrieval mode:

   | Channel | hit@1 | hit@5 | MRR | p50 latency | Notes |
   |---|---|---|---|---|---|
   | `lexical` (BM25 only) | 0.92 | 1.00 | 0.962 | ~3 ms | one paraphrase miss at rank 2 |
   | `semantic` (pgvector only) | 0.92 | 1.00 | 0.942 | ~85 ms | one multi-hop miss at rank 4 |
   | `hybrid_norerank` (RRF over semantic + keyword) | 0.92 | 1.00 | 0.962 | ~80 ms | RRF promotes the paraphrase miss back to 1 |
   | `hybrid_rerank` (RRF + cross-encoder) | 0.85 | 1.00 | 0.904 | ~28 s | reranker reorders multi-hop down to rank 2; latency dominated by CPU inference |
   | `hybrid_graph_norerank` (RRF + graph channel, no rerank) | 0.92 | 1.00 | 0.962 | ~880 ms | graph channel injects extra candidates without disrupting other channels' ranks |
   | `hybrid_graph_rerank` (full stack) | 0.92 | 1.00 | 0.949 | ~19 s | reranker still pushes the multi-hop miss to rank 3, but graph channel keeps recall at 100% |
   | `graph_only` (`/v1/search/graph`, NER -> Memgraph -> chunks) | 0.15 | 0.15 | 0.154 | ~240 ms | by design only fires on entity-anchored queries; on the 3 multi-hop entity queries it gets 2/3 |

   Caveats already noted in `tests/integration_validation/REPORT.md`: at 11 docs every channel saturates `hit@5`. The graph_only channel is purposefully narrow: it returns empty (200) on non-entity queries so callers can fall back to `/v1/search`. The latency cost of the cross-encoder reranker (~17-29 s per query on CPU) is the main thing the matrix measures cleanly; the quality lift is invisible at this corpus size, which is why upgrading the reranker is in `project_rag_base_engine_deferrals.md` until a harder corpus exists.

4. **Persistence tests.**
   - `docker compose down + up` (no rebuild). All 14 docs (11 corpus + 3 graph variants) survived. Lexical, semantic, hybrid_graph_norerank, graph_only all returned identical results. Tiny MRR drift on lexical (0.962 -> 0.949) is because the graph-variant docs now compete in the candidate pool, not a persistence issue. Once the corpus is stable the number is deterministic.
   - `docker compose down`, `docker volume rm` for the orphaned `ragbase_mgdata` / `ragbase_pgdata` named volumes from earlier sessions, `docker volume prune -f`, `docker compose up`. All 14 docs still survived. All channels returned identical results.

   This proves the no-named-volumes rule from `feedback_no_docker_volumes.md` actually holds. `docker volume prune` does NOT touch project state because postgres + memgraph + lightrag all live on host bind mounts under `./data/`.

---

## Engine state at ship

### Endpoints (all on `http://localhost:5050`)

- `POST /v1/documents`, `GET /v1/documents`, `GET /v1/documents/{id}`, `DELETE /v1/documents/{id}` (CRUD; ingest auto-chunks, embeds via TEI, runs LightRAG entity extraction in the background)
- `POST /v1/search` (hybrid: semantic + keyword + optional graph, RRF, optional rerank)
- `POST /v1/search/semantic` (vector-only)
- `POST /v1/search/keyword` (BM25-only via ParadeDB pg_search)
- `POST /v1/search/graph` (graph-only fast mode: GLiNER NER -> Memgraph match -> 0/1/2-hop traversal -> chunk bridge; no embedding, no LLM at query time, no rerank)
- `POST /v1/concepts`, `GET /v1/concepts/{id}`, `DELETE /v1/concepts/{id}` (`:Concept` typed graph)
- `POST /v1/relations`, `GET /v1/relations`, `DELETE /v1/relations/{id}` (typed directed edges)
- `GET /v1/graph/neighbors/{id}`, `GET /v1/graph/path/{from}/{to}`, `GET /v1/graph/communities`, `GET /v1/graph/stats` (graph traversal + algorithms via Memgraph MAGE)
- `POST /v1/embed`, `POST /v1/rerank` (TEI passthroughs)
- `GET /health`, `GET /health/models` (status + model info)

### Persistence (all bind-mounted, all gitignored)

| Path on host | Inside container | Survives `docker compose down`? | Survives `docker volume prune`? |
|---|---|---|---|
| `./data/postgres/` | `/var/lib/postgresql/data` (PGDATA subdir) | yes (verified) | yes (verified) |
| `./data/memgraph/` | `/var/lib/memgraph` | yes (verified) | yes (verified) |
| `./data/lightrag/` | `/app/lightrag_data` | yes (verified) | yes (verified) |
| `EMBEDDING_MODELS_DIR` (`~/models/embeddings`) | `/data` in TEI embed | yes | yes |
| `RERANKER_MODELS_DIR` (`~/models/rerankers`) | `/data` in TEI rerank | yes | yes |

### Stack

- Postgres: paradedb/paradedb 0.23.1-pg17 (pgvector + pg_search BM25)
- Embedding: TEI cpu-1.9 with BAAI/bge-m3 (1024d, dev override of the Snowflake default)
- Reranker: TEI cpu-1.9 with BAAI/bge-reranker-v2-m3 (CPU; ~17-29 s per query on this corpus)
- Graph: memgraph/memgraph-mage 3.9.0
- API: FastAPI on python:3.12-slim, with LightRAG (`lightrag-hku>=1.4.15`) for ingest-time entity/relation extraction and GLiNER (`knowledgator/gliner-multitask-v1.0`) for query-time NER on the graph-only endpoint
- LLM: local Qwen3.6-27B-AWQ4 via vLLM Responses API with `enable_thinking=false` (50x latency improvement on short prompts)

### Test suites

- Unit tests: `tests/unit/` (chunking, fusion, models, graph_only_search internals)
- Integration tests: `tests/integration/` (lifecycle, search paths, lightrag wiring, phase2/3c/5 use cases, endpoint regressions). All slow tests gated by `@pytest.mark.slow`.
- Integration validation runner: `tests/integration_validation/run.py` with channels `lexical | semantic | hybrid_norerank | hybrid_rerank | hybrid_graph_rerank | hybrid_graph_norerank | graph_only | header_ablation`. Each step is ctrl-C-able and persists state across invocations in `results/state.json`.
- Phase 4 retrieval matrix findings: `tests/integration_validation/REPORT.md`.
- This audition pass: `SHIP.md` (this file).

### Known limitations carried forward (intentional)

- LightRAG ingest is ~1-3 min per chunk on the local Qwen3.6-27B-AWQ4 even with thinking off. Sustainable for ad-hoc ingest; bulk pipelines should ingest with `X-LightRAG-Ingest: false` then bulk-extract later.
- `BGE-reranker-v2-m3` on CPU TEI is generation-behind. Documented decision to defer in `README.md` "Phase 3 / Decisions deferred". Reason: TEI cannot load the SOTA candidates and a sidecar vLLM rerank container would contend for VRAM with the host LLM.
- `Qwen3-Embedding-8B` semantic upgrade also deferred for VRAM/cost reasons. Same rationale.
- Graph-only endpoint scoring is "supporting-entity count" then "node degree", not an LLM judge. Cheap and predictable; trade-off is that purely structural queries ("the researcher whose company is in Berkeley") need the hybrid path.
- TEI CPU only. GPU swap is a tag change (cuda/rocm).
- No authentication. Wrap in nginx/caddy + auth for production.
- Single node. No replication. Postgres backup/restore via `scripts/backup.sh` is the migration story today.

---

## What ships next

1. The audition pass is complete. rag-base is feature-frozen for the engine layer.
2. Next phase opens in `knowledge-base` (sibling repo, `/home/hec/workspace/knowledge-base/`). The integration plan is captured under task #50.
3. rag-base may pick up follow-on work later (bigger adversarial corpus to actually separate channel quality, GPU TEI, SOTA reranker swap when a second VRAM budget exists), but those are conditional on the wrapper's evaluation needs.
