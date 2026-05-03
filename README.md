<h1 align="center">rag-base</h1>

<p align="center">
  <strong>Standalone RAG backend. Hybrid search (semantic + BM25 + graph), cross-encoder rerank, LightRAG entity graph, GLiNER fast-mode.<br>
  One <code>docker compose up</code>. REST API on :5050. Bring your own corpus.</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Status-Shipped_2026--04--27-brightgreen" alt="Status" />
  <img src="https://img.shields.io/badge/hit@5-1.00_across_all_hybrid_channels-success" alt="hit@5" />
  <img src="https://img.shields.io/badge/Endpoints-4_search_modes-blue" alt="Endpoints" />
  <img src="https://img.shields.io/badge/Persistence-bind--mounts_(prune--proof)-0b7285" alt="Persistence" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Postgres-17_+_pgvector_+_pg__search-336791?logo=postgresql&logoColor=white" alt="Postgres" />
  <img src="https://img.shields.io/badge/Embedder-BGE--M3_1024d-yellow?logo=huggingface&logoColor=black" alt="Embedder" />
  <img src="https://img.shields.io/badge/Reranker-BGE--reranker--v2--m3-yellow?logo=huggingface&logoColor=black" alt="Reranker" />
  <img src="https://img.shields.io/badge/Graph-Memgraph_MAGE_3.9-FF7A00" alt="Memgraph" />
  <img src="https://img.shields.io/badge/Entity_extraction-LightRAG_HKUDS-7e57c2" alt="LightRAG" />
  <img src="https://img.shields.io/badge/NER-GLiNER_multitask--v1.0-7e57c2" alt="GLiNER" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/API-FastAPI_+_async-009688?logo=fastapi&logoColor=white" alt="FastAPI" />
  <img src="https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white" alt="Docker" />
  <img src="https://img.shields.io/badge/License-MIT-lightgrey" alt="License" />
</p>

---

## ⚡ The numbers (audition pass, 2026-04-27)

11-doc adversarial corpus, 13 tagged queries (polysemy, paraphrase, exact-phrase rare jargon, multi-hop entity chain, header-path ablation, distractors). Full matrix in `tests/integration_validation/REPORT.md`.

| Channel | hit@1 | hit@5 | MRR | p50 latency | Notes |
|---|---:|---:|---:|---:|---|
| 🟢 `lexical` (BM25) | **0.92** | **1.00** | **0.949** | **3 ms** | one paraphrase miss at rank 3 |
| 🟧 `semantic` (pgvector cosine) | 0.85 | 0.92 | 0.897 | 45 ms | two multi-hop misses (rank 2 and rank 6); only non-fused channel that does not saturate hit@5 |
| 🟢 `hybrid_norerank` (RRF) | **0.92** | **1.00** | **0.962** | 80 ms | RRF promotes paraphrase miss back to 1 |
| 🟧 `hybrid_rerank` (RRF + cross-encoder) | 0.85 | 1.00 | 0.904 | **~28 s** | reranker reorders multi-hop down; latency dominated by CPU inference |
| 🟢 `hybrid_graph_norerank` (RRF + graph) | **0.92** | **1.00** | **0.962** | 880 ms | graph adds candidates without disrupting other channels |
| 🟧 `hybrid_graph_rerank` (full stack) | 0.92 | 1.00 | 0.949 | ~19 s | graph keeps recall at 100% even when reranker hurts MRR |
| 🟦 `graph_only` (`/v1/search/graph`) | 0.15 | 0.15 | 0.154 | 240 ms | by design: empty 200 on non-entity queries; 2/3 on the multi-hop entity slice |

> **At this corpus size every hybrid channel saturates `hit@5 = 1.00`.** The discriminative signal is in `hit@1` and per-query rank. Bigger adversarial corpora will separate channels further; that's tracked in `tests/integration_validation/REPORT.md`.

### 🏁 Full-pipeline benchmark (HAI synthetic corpus, 2026-05-03)

Why a second corpus: the audition numbers above were taken on an 11-doc adversarial corpus before the GPU rerank work landed. This benchmark exercises **every public endpoint and every retrieval channel** end-to-end on a fresh non-trainable corpus, with all four `rerank_model` modes plus the LightRAG entity-extraction pipeline turned on. Repeatable runner: `tests/integration_validation/hai_full_bench.py` + sibling `hai_corpus.py`.

**Why "non-trainable"**: every proper noun (companies, projects, theorems, people, locations) was invented for this benchmark. No public corpus contains "Quibbler-Frame protocol" or "Zigast's theorem". The reranker cannot pattern-match from training data; it has to read the chunk against the query and score relevance honestly.

**Setup**
- **Corpus**: 20 fictional docs (16 in-world + 4 distractors), each ~80 words, ingested through `POST /v1/documents` *with* LightRAG entity extraction (no `X-LightRAG-Ingest: false` header). Ingest cost: ~56 s/doc on local Qwen3.6-27B-AWQ4 vLLM with `enable_thinking=false`. Total ingest wall: ~19 min.
- **Memgraph state after ingest**: 98 entities, 122 directed relations (HAI added 71 entities + 96 relations to the pre-existing 27/26 baseline).
- **Queries**: 12 hand-crafted, none copy-paste of any chunk text. Mix of paraphrase, exact rare-jargon, multi-hop, and polysemy.
- **Hardware**: AMD Strix Halo (gfx1151), 128 GiB UMA, vLLM at `gpu-memory-utilization=0.55` running the whole time.

#### Quality matrix (sequential)

`top_k=10`, `rerank_candidates=50`. `hybrid_graph_rerank` adds the LightRAG graph channel to the RRF fusion before rerank.

| Channel / mode | hit@1 | hit@5 | MRR | p50_ms |
|---|---:|---:|---:|---:|
| `keyword` (BM25 only) | 0.67 | 1.00 | 0.819 | **2** |
| `semantic` (pgvector only) | 0.67 | 1.00 | 0.833 | 63 |
| `hybrid_norerank` (RRF, no rerank) | 0.67 | 1.00 | 0.833 | 68 |
| `hybrid_rerank` / `default` (CPU TEI BGE-v2-m3) | **0.75** | **1.00** | **0.861** | 3,173 |
| `hybrid_rerank` / `bge-gpu` | **0.75** | **1.00** | **0.861** | **153** |
| `hybrid_rerank` / `qwen-4b` | 0.67 | 1.00 | 0.819 | 1,377 |
| `hybrid_rerank` / `qwen-8b` | 0.67 | 1.00 | 0.833 | 2,519 |
| `hybrid_graph_rerank` / `default` | **0.75** | **1.00** | **0.861** | 4,628 |
| `hybrid_graph_rerank` / `bge-gpu` | **0.75** | **1.00** | **0.861** | 1,775 |
| `hybrid_graph_rerank` / `qwen-4b` | 0.67 | 1.00 | 0.819 | 3,024 |
| `hybrid_graph_rerank` / `qwen-8b` | 0.67 | 1.00 | 0.833 | 4,103 |
| `graph_only` (`/v1/search/graph`) | 0.00 | 0.00 | 0.000 | 263 |

What this says:

- **bge-gpu is the practical winner**: identical hit@1/MRR to the CPU default (same model, fp16 noise), 20x faster sequentially. The same comparison vs `hybrid_norerank` shows the rerank stage adds +0.08 hit@1 on this corpus.
- **Qwen-4B and Qwen-8B trail BGE on hit@1** for this 12-query slice: 0.67 vs 0.75. Qwen-8B beats 4B by +0.014 MRR (one rank-2 improvement on the multi-hop coastal-lab query). On a longer adversarial corpus the LM-style scoring is more likely to widen, but on this slice it is a wash. Plug-and-play available; enable when your corpus benefits.
- **`graph_only` returns 0/12** as designed: the queries are paraphrases, never exact entity-name lookups, so GLiNER cannot anchor in Memgraph. This channel is meant for entity-anchored questions like "give me everything about Hardin Volkenburg", not paraphrase questions. Verified by `phase_graph_crud` below: `/v1/graph/path/152/153` returns the correct 2-hop path between two seeded test concepts, so the channel itself works.
- **`hybrid_graph_rerank` matches `hybrid_rerank`** on hit@1/MRR. The graph channel adds candidates without disrupting rank order; the cost is one extra LLM call (entity extraction) at ~1.4 s.

#### Endpoints exercised (full coverage)

| Endpoint | Status |
|---|:---:|
| `GET /health`, `/health/models` | green |
| `POST /v1/documents` (LightRAG on) | green, 20/20 docs, ~56 s/doc |
| `GET /v1/documents`, `GET /v1/documents/{id}` | green |
| `POST /v1/embed` | green (1024d, BGE-M3) |
| `POST /v1/rerank` (passthrough) | green |
| `POST /v1/search` (default + bge-gpu + qwen-4b + qwen-8b) | green |
| `POST /v1/search/semantic`, `/keyword`, `/graph` | green |
| `POST /v1/concepts`, `DELETE /v1/concepts/{id}` | green |
| `POST /v1/relations`, `GET /v1/relations`, `DELETE /v1/relations/{id}` | green |
| `GET /v1/graph/neighbors/{id}`, `/graph/path/{a}/{b}` | green |
| `GET /v1/graph/stats` (concepts=2 test, relations=131) | green |
| `GET /v1/graph/communities` (Louvain, 105 communities) | green |

#### Stability under concurrent load

Concurrency 8, 2 repeats x 12 queries x (4 flat channels + 2 rerank-channels x 4 rerank modes) = **288 total live-pipeline calls fired in parallel**. vLLM Responses API (`enable_thinking=false`, `stream=true`) was poked before, mid-run, and after.

| Channel / mode | n | p50 ms | p95 ms | min | max |
|---|--:|---:|---:|---:|---:|
| `keyword` | 24 | **7** | 15 | 3 | 45 |
| `semantic` | 24 | 254 | 446 | 138 | 586 |
| `hybrid_norerank` | 24 | 392 | 691 | 108 | 865 |
| `hybrid_rerank` / `bge-gpu` | 24 | **1,283** | 1,635 | 953 | 1,850 |
| `hybrid_rerank` / `default` | 24 | 13,255 | 21,143 | 4,950 | 30,293 |
| `hybrid_rerank` / `qwen-4b` | 24 | 13,631 | 16,772 | 8,188 | 17,269 |
| `hybrid_rerank` / `qwen-8b` | 24 | 27,794 | 38,645 | 12,492 | 52,504 |
| `hybrid_graph_rerank` / `bge-gpu` | 24 | 9,309 | 16,306 | 4,344 | 19,610 |
| `hybrid_graph_rerank` / `default` | 24 | 18,821 | 25,091 | 8,134 | 25,895 |
| `hybrid_graph_rerank` / `qwen-4b` | 24 | 22,320 | 27,569 | 17,559 | 29,421 |
| `hybrid_graph_rerank` / `qwen-8b` | 24 | 38,067 | 47,646 | 20,037 | 54,208 |
| `graph_only` | 24 | 9,780 | 10,946 | 431 | 11,403 |

Total wall: **459 s for 288 jobs, 0 errors.** vLLM stayed healthy across the run: 3-poke TTFT 7 ms / 19 ms / 16 ms, all status 200, no degradation. All four reranker containers (CPU TEI + BGE-GPU + Qwen-4B + Qwen-8B) plus Postgres + Memgraph + embedding TEI stayed healthy throughout. No tracebacks in any sidecar log.

#### Direct sidecar microbench (no API in the loop)

Per-batch latency at the `/rerank` endpoint, identical 32-doc payload:

| Sidecar | batch=4 | batch=16 | batch=32 |
|---|---:|---:|---:|
| CPU TEI (`reranker:80`) | 172 ms | 732 ms | **1,446 ms** |
| `bge-gpu` (`:8083`) | 16 ms | 35 ms | **56 ms** |
| `qwen-4b` (`:8084`) | 146 ms | 507 ms | **993 ms** |
| `qwen-8b` (`:8085`) | est. ~270 | est. ~1,000 | **~2,000 ms** |

### 💾 Persistence (verified)

| Test | Survived? |
|---|---:|
| 🟢 `docker compose down + up` | **yes** (14 docs intact, all channels return identical results) |
| 🟢 `docker volume prune -f` | **yes** (no project state in named volumes; everything bind-mounted) |
| 🟢 `docker compose build api` rebuild | **yes** (LightRAG KV is bind-mounted to `./data/lightrag/`) |

The whole stack is bind-mounted to host dirs under `./data/`. There are **no docker named volumes**. `docker volume prune` is safe.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│              rag-base  (docker compose up)              │
│                                                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────┐   │
│  │ Postgres │  │ TEI      │  │ TEI      │  │Memgraph│   │
│  │ pgvector │  │ Embed    │  │ Rerank   │  │  MAGE  │   │
│  │ pg_search│  │  :8081   │  │  :8082   │  │ :7687  │   │
│  │  :5433   │  │ BGE-M3   │  │ BGE-r-m3 │  │  3.9   │   │
│  └──────────┘  └──────────┘  └──────────┘  └────────┘   │
│       ▲              ▲             ▲            ▲       │
│       └──────────────┴─────────────┴────────────┘       │
│                         │                               │
│              ┌──────────▼──────────┐                    │
│              │   API (FastAPI)     │                    │
│              │   :5050             │                    │
│              │   + LightRAG        │                    │
│              │   + GLiNER          │                    │
│              └─────────────────────┘                    │
└─────────────────────────────────────────────────────────┘
                      ▲
                      │  REST API (:5050)
                      │
              Any client application
```

**5 always-on containers (4 prebuilt images, 1 custom API), plus 2 optional GPU rerank sidecars on AMD Strix Halo.**

| Service | Image | Role | Profile |
|---|---|---|---|
| 🟦 **Postgres + pgvector + pg_search** | `paradedb/paradedb:0.23.1-pg17` | Documents, chunks, vectors. HNSW for semantic + Tantivy BM25 for keyword. | required |
| 🟨 **TEI Embed** | `text-embeddings-inference:cpu-1.9` | Text to 1024d vectors via [HuggingFace TEI](https://github.com/huggingface/text-embeddings-inference). | required |
| 🟨 **TEI Rerank** (CPU) | `text-embeddings-inference:cpu-1.9` | BGE-reranker-v2-m3 on CPU. Default rerank backend. | `--profile rerank` |
| 🟩 **GPU Rerank: BGE** | `reranker-rocm:local` (custom) | Same BGE-reranker-v2-m3 on AMD gfx1151 ROCm. Picked by `rerank_model: "bge-gpu"`. | `--profile rerank-bge-gpu` |
| 🟩 **GPU Rerank: Qwen-4B** | `reranker-rocm:local` (custom) | Qwen3-Reranker-4B (LM-style yes/no scoring). Picked by `rerank_model: "qwen-4b"`. | `--profile rerank-qwen` |
| 🟩 **GPU Rerank: Qwen-8B** | `reranker-rocm:local` (custom) | Qwen3-Reranker-8B (~16 GiB VRAM). Picked by `rerank_model: "qwen-8b"`. | `--profile rerank-qwen-8b` |
| 🟧 **Memgraph MAGE** | `memgraph/memgraph-mage:3.9.0` | Knowledge graph: LightRAG entities (`:base`) + custom concepts (`:Concept`). Cypher, PageRank, Louvain, BFS. | `--profile graph` |
| 🟪 **API** | Custom (`python:3.12-slim`) | Orchestrator. FastAPI + LightRAG (ingest-time entity extraction) + GLiNER (query-time NER). | required |

> Reranker, Memgraph, and the GPU sidecars are **fully optional**. The api detects their absence at startup and degrades gracefully. Hybrid + rerank still work without graph; semantic + BM25 still work without rerank. Requesting an unavailable GPU sidecar via `rerank_model` silently falls back to the default CPU TEI reranker.

---

## 🚀 Quick start

<details>
<summary><b>Click to expand - first boot in ≈ 4-5 min, subsequent boots in ≈ 30 s</b></summary>

### 1. Prereqs
- Docker + docker compose
- ~10 GB free disk for model caches
- A host LLM endpoint compatible with the OpenAI Responses API (vLLM, llama.cpp `server` with `--responses`, etc). Any OpenAI-compatible LLM works for ingest-time entity extraction. **The api still runs without an LLM**, you just lose the LightRAG entity graph at ingest.

### 2. Clone + configure

```bash
git clone git@github.com:hec-ovi/rag-base.git
cd rag-base
cp .env.template .env
nano .env
```

Three things need attention:
```env
POSTGRES_PASSWORD=your-secure-password
EMBEDDING_MODELS_DIR=/absolute/path/for/embedder/cache
RERANKER_MODELS_DIR=/absolute/path/for/reranker/cache
LLM_BASE_URL=http://host.docker.internal:8000     # your host LLM endpoint
LLM_MODEL=Qwen3.6-27B-AWQ4                        # model id your endpoint serves
```

### 3. First-boot prereq (Memgraph UID 101 chown)

Memgraph runs as UID 101 inside the container. The host bind-mount dir must be owned by 101 or it will SIGSEGV at startup with banner-only logs:

```bash
mkdir -p ./data/postgres ./data/memgraph ./data/lightrag
docker run --rm -v "$PWD/data/memgraph":/data alpine chown -R 101:101 /data
```

Postgres uses the `PGDATA=/var/lib/postgresql/data/pgdata` subdir trick so it handles its own ownership; the api runs as root so `./data/lightrag` is fine without chown.

### 4. Bring it up

```bash
# Core only (postgres + embedding + api)
docker compose up -d

# Add reranking (recommended for quality)
docker compose --profile rerank up -d

# Add knowledge graph (LightRAG entity extraction + graph endpoints)
docker compose --profile graph up -d

# Add GPU reranker sidecars (AMD Strix Halo / gfx1151 only, requires /dev/kfd + /dev/dri)
docker compose --profile rerank-bge-gpu up -d         # same model as CPU TEI, ~25x faster
docker compose --profile rerank-qwen up -d            # Qwen3-Reranker-4B, better on hard corpora
docker compose --profile rerank-qwen-8b up -d         # Qwen3-Reranker-8B, ~16 GiB VRAM

# Everything
docker compose --profile rerank --profile rerank-bge-gpu --profile rerank-qwen --profile rerank-qwen-8b --profile graph up -d
```

The GPU sidecars only run when their profile is active. Set the matching URL in `.env` so the api can find them; clients then opt in per request via `rerank_model`. Omit the field to keep the existing CPU TEI path.

| Sidecar | env var | value |
|---|---|---|
| BGE on gfx1151 | `BGE_GPU_RERANKER_URL` | `http://reranker-bge-gpu:80` |
| Qwen-4B on gfx1151 | `QWEN_RERANKER_URL` | `http://reranker-qwen:80` |
| Qwen-8B on gfx1151 | `QWEN_8B_RERANKER_URL` | `http://reranker-qwen-8b:80` |

TEI downloads embed/rerank models on first boot (~3 min on a normal connection). The api starts as soon as postgres + embedding are healthy, so the reranker may finish AFTER api boots. One-time fix:

```bash
docker compose restart api
```

### 5. Verify

```bash
curl http://localhost:5050/health
```
```json
{"status": "ok", "postgres": "connected", "embedding": "healthy", "reranker": "healthy", "memgraph": "connected"}
```

### 6. Ingest + search

```bash
curl -X POST http://localhost:5050/v1/documents \
  -H "Content-Type: application/json" \
  -d '{
    "title": "React Documentation",
    "content": "React is a JavaScript library for building user interfaces.",
    "metadata": {"source": "docs"}
  }'

curl -X POST http://localhost:5050/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query": "frontend frameworks", "top_k": 5}'
```

</details>

---

## 🔌 API reference

OpenAPI docs auto-generated at <http://localhost:5050/docs>. Full request/response specs in [`llm.txt`](llm.txt).

### Search modes

| Endpoint | Channels | Rerank | Graph | When to use |
|---|---|:---:|:---:|---|
| `POST /v1/search` | semantic + keyword + (graph) -> RRF | configurable | configurable | default for general queries; recall + precision |
| `POST /v1/search/semantic` | pgvector cosine only | 🔴 | 🔴 | paraphrase-heavy queries |
| `POST /v1/search/keyword` | BM25 only (`indexed_content`) | 🔴 | 🔴 | rare jargon, exact terms, statute citations |
| `POST /v1/search/graph` | GLiNER NER -> Memgraph -> chunks | 🔴 | 🟢 only | entity-anchored queries; latency-sensitive (no LLM at query time) |

#### `rerank_model` (request field on `/v1/search`)

Picks which reranker backend to use. Optional; omit to keep today's behavior.

| Value | Backend | Latency (this engine, batch 32 direct) | When to use |
|---|---|---:|---|
| omitted / `"default"` | CPU TEI BGE-v2-m3 (`reranker` service) | 1446 ms | default; no GPU needed |
| `"bge-gpu"` | Same BGE-v2-m3 on AMD gfx1151 GPU | 56 ms | same quality as default, ~25x faster |
| `"qwen-4b"` | Qwen3-Reranker-4B on AMD gfx1151 GPU | 993 ms | LM-style scoring; useful on semantically subtle queries |
| `"qwen-8b"` | Qwen3-Reranker-8B on AMD gfx1151 GPU | ~2 s | larger LM reranker; slight edge over 4B on multi-hop chains, ~16 GiB VRAM |

If `"bge-gpu"` or `"qwen-4b"` is requested but the corresponding sidecar URL is unset or unreachable, the API silently falls back to the default CPU TEI reranker (a warning is logged). This makes the new modes safe to send from any client without coordination.

### Endpoints at a glance

| Method | Endpoint | Purpose |
|---|---|---|
| **Documents** | | |
| `POST` | `/v1/documents` | Ingest: chunk, embed, store, optionally LightRAG-extract entities |
| `GET` | `/v1/documents?offset=0&limit=20` | List documents (paginated) |
| `GET` | `/v1/documents/{id}` | Get document with chunks |
| `DELETE` | `/v1/documents/{id}` | Delete document + chunks (cascade) |
| **Search** | | |
| `POST` | `/v1/search` | Hybrid (semantic + keyword + graph + rerank) |
| `POST` | `/v1/search/semantic` | Vector-only |
| `POST` | `/v1/search/keyword` | BM25-only |
| `POST` | `/v1/search/graph` | Graph-only fast mode (NER -> Memgraph -> chunks) |
| **Graph** | | |
| `POST` | `/v1/concepts` | Create/update typed concept (upsert by name) |
| `GET` | `/v1/concepts/{id}` | Get concept with relations |
| `DELETE` | `/v1/concepts/{id}` | Delete concept + edges |
| `POST` | `/v1/relations` | Create directed edge between concepts |
| `GET` | `/v1/relations?concept_name=X` | Get relations for a concept |
| `DELETE` | `/v1/relations/{id}` | Delete relation |
| `GET` | `/v1/graph/neighbors/{id}?depth=N` | Multi-hop traversal (max depth 5) |
| `GET` | `/v1/graph/path/{from_id}/{to_id}` | Shortest path (BFS) |
| `GET` | `/v1/graph/communities` | Louvain community detection |
| `GET` | `/v1/graph/stats` | Node/edge counts |
| **Passthrough** | | |
| `POST` | `/v1/embed` | Embed text(s) directly via TEI |
| `POST` | `/v1/rerank` | Rerank candidates directly via TEI |
| **Health** | | |
| `GET` | `/health` | All-services status |
| `GET` | `/health/models` | Loaded model info |

---

## ✅ What works · 🟡 What's by design · 🔴 Known limits

| | Status |
|---|---|
| 🟢 Atomic ingest (doc + chunks + vectors in one PG transaction) | shipped |
| 🟢 Real BM25 via ParadeDB pg_search (Tantivy in Postgres) | shipped, replaces older `tsvector + ts_rank` |
| 🟢 Contextual Chunk Headers (title + metadata + markdown header path) auto-prepended | shipped |
| 🟢 RRF fusion across semantic / keyword / graph channels | shipped |
| 🟢 Cross-encoder rerank (last stage) | shipped (CPU TEI default; optional GPU sidecars: BGE on gfx1151, Qwen3-Reranker-4B on gfx1151) |
| 🟢 GPU reranker sidecars on AMD Strix Halo (gfx1151) | shipped 2026-05-03; same BGE-v2-m3 (~25x faster than CPU), Qwen3-Reranker-4B (~1.5x faster), or Qwen3-Reranker-8B (~1.3x faster). All three share the same 1.97 GiB image; pick at request time via `rerank_model`. |
| 🟢 LightRAG entity + relation extraction at ingest -> Memgraph | shipped (LightRAG 1.4.15+, CVE-patched) |
| 🟢 GLiNER NER for graph-only fast mode (no LLM at query time) | shipped |
| 🟢 Graph channel results actually merged into RRF (closes the historical search.py:66 TODO) | shipped |
| 🟢 Graceful degradation: api never crashes on missing optional services | shipped |
| 🟢 Anthropic Contextual Retrieval (`contextual_retrieval: true` flag on `POST /v1/documents`) | shipped, opt-in; per-chunk LLM blurb prepended to `indexed_content`, vLLM auto-caches the document prefix |
| 🟢 Editable prompt files in `api/prompts/*.md` (decoupled from Python code) | shipped; covers CR + query-time entity extraction, restart api to reload |
| 🟢 Test gates: every SOTA mechanism has a "did it actually fire?" test | shipped (131 tests: 123 fast + 7 slow + 1 environment-conditional skip; the +16 from the rerank work: 17 sidecar unit tests, 12 routing unit tests, 4 live-stack 3-mode tests; sidecar tests run without torch via FakeCrossEncoder, integration tests gated by env when sidecars are not running) |
| 🟡 LightRAG ingest is ~1-3 min per chunk on local 27B reasoning LLM | by design; bulk-ingest path uses `X-LightRAG-Ingest: false` header |
| 🟡 CPU TEI rerank is slow (multi-second at batch 50) | by design; opt-in GPU sidecars (`rerank-bge-gpu` / `rerank-qwen` profiles) drop full-pipeline `/v1/search` p50 from ~530 ms to ~65 ms (BGE-GPU) or ~291 ms (Qwen-4B) on this engine |
| 🟡 Contextual Retrieval blurbing reuses `LLM_BASE_URL`, so a 27B reasoning LLM blurbs slowly | by design; per-purpose LLM env vars are a deferred refactor (see "Deferred decisions") |
| 🔴 No authentication; intended behind a reverse proxy | known |
| 🔴 Single node; no clustering/replication | known |
| 🔴 TEI cpu-1.9; no GPU image baked in | tag swap to `cuda` / `rocm` variants when needed |

---

## 🧠 Retrieval channel deep dive

### Hybrid search pipeline (`POST /v1/search`)

```
   query
     │
     ├──► [embed via TEI] ──► query_vector
     │
     ┌─────────────────────────────────────────┐
     │  PARALLEL FAN-OUT                       │
     ├─────────────────────────────────────────┤
     │  semantic   pgvector cosine over chunks │
     │  keyword    BM25 over indexed_content   │
     │  graph      LLM ent. extraction ->      │
     │             Memgraph match -> doc-      │
     │             restricted semantic search  │
     └─────────────────────────────────────────┘
                          │
                          ▼
                 RRF (1 / (60 + rank_i))
                          │
                          ▼
        cross-encoder rerank (top-N -> top-K)
                          │
                          ▼
                       results
```

### Graph-only fast mode (`POST /v1/search/graph`)

```
   query
     │
     ▼
   GLiNER NER (no LLM)
     │
     ▼
   Memgraph entity match (exact_ci, fallback CONTAINS)
     │
     ▼
   0/1/2-hop subgraph (two-query nodes-then-edges)
     │
     ▼
   degree/none ranking (seeds first)
     │
     ▼
   source_id bridge: chunk-<hash> -> LightRAG KV ->
   (full_doc_id, chunk_order_index) -> chunks.id
     │
     ▼
   {matched_entities, subgraph: {nodes, edges}, chunks, trace}
```

**Why both?** `/v1/search` answers general questions with full retrieval power (~80 ms without rerank, ~28 s with). `/v1/search/graph` answers entity-anchored questions in ~240 ms with no LLM call, and returns the actual subgraph + per-chunk source attribution so you can show your work.

### Contextual Chunk Headers (built-in)

rag-base auto-prepends structural prefixes to each chunk before embedding and BM25:

1. Document title and metadata: `[<title> | <k1>: <v1> | <k2>: <v2>]`
2. Markdown heading breadcrumb at chunk start: `[Section > Subsection > Heading]` (omitted for headerless chunks)
3. **Optional**: Anthropic Contextual Retrieval blurb (a 50-100 token LLM-generated context line) when `contextual_retrieval: true` is sent on `POST /v1/documents`
4. The raw chunk text

The augmented form lives in `chunks.indexed_content` and is what the embedder sees and what BM25 indexes. Raw chunk text in `chunks.content` stays clean for display.

**Steps 1-2 are always on, no configuration.** Step 3 is opt-in per ingest:

```bash
curl -X POST http://localhost:5050/v1/documents \
  -H 'Content-Type: application/json' \
  -d '{"title": "...", "content": "...", "metadata": {...}, "contextual_retrieval": true}'
```

When enabled, the api makes one LLM call per chunk (the configured `LLM_BASE_URL`) with the full document as a stable prefix. vLLM's automatic prefix caching shares the document KV across the per-chunk calls, dropping the cost to ~$1 per 1M document tokens (Anthropic's measured number with their cookbook + prompt cache). On per-chunk LLM failure the chunk gets no blurb but ingest still succeeds. The prompt body is `api/prompts/contextual_retrieval.md`, editable without code changes; restart the api container to reload after editing.

Anthropic reports **35% / 49% / 67%** reduction in retrieval failures (alone, with BM25, with rerank). Lift is corpus-dependent; on saturated smoke sets the gain is invisible. Measure on your real corpus before deciding whether to enable corpus-wide.

#### CR verification on this engine (2026-04-27)

Ingested an 1784-word doc (`llm.txt` head, 4 chunks) through `POST /v1/documents` with `contextual_retrieval: true` against the local Qwen3.6-27B-AWQ4 vLLM. All 4 blurbs landed clean and specific (e.g. *"Technical Reference: System Purpose, Ingest Pipeline (Chunking, Contextual Chunk Headers, Embedding, Postgres Storage, LightRAG Entity Extraction), and Hybrid Retrieval Pipeline …"* ), no generic sludge, no per-chunk failures. Wall: 580 s for 4 chunks (serial; cache-warm pattern: first call solo to populate the prefix cache, the rest sequential).

Retrieval matrix against the same doc, 4 queries each crafted to target a specific chunk:

| Query (paraphrased) | target chunk | lexical | semantic | hybrid |
|---|---|:---:|:---:|:---:|
| graph fast mode endpoint with NER | 1 | **1** | **1** | **1** |
| reranker container memgraph ports | 2 | **1** | **1** | **1** |
| startup order health checks | 3 | **1** | **1** | **1** |
| system purpose ingest pipeline | 0 | 2 | **1** | **1** |

**11/12 = 91.7% hit@1** across the matrix. The single rank-2 (lexical) put the CR-augmented chunk at rank 2 behind a leftover non-CR chunk with overlapping content; semantic and hybrid both pulled it back to rank 1, with the CR-augmented chunk outranking the non-CR equivalent in the same retrieval space. That's CR doing its job: the blurb adds disambiguating context the embedder uses to break ties.

`chunks.content` integrity verified: all 4 chunks hold unmodified slices of the source doc; `indexed_content` carries the CCH + blurb. The reranker (which sees `chunks.content`, not `indexed_content`) is unaffected by CR.

---

## 💾 Persistence

| What | Where | Survives `down + up`? | Survives `docker volume prune`? |
|---|---|:---:|:---:|
| Documents + chunks + vectors | `./data/postgres/` -> `/var/lib/postgresql/data` (PGDATA subdir) | 🟢 | 🟢 |
| LightRAG entity graph | `./data/memgraph/` -> `/var/lib/memgraph` | 🟢 | 🟢 |
| LightRAG KV + nano-vectordb | `./data/lightrag/` -> `/app/lightrag_data` | 🟢 | 🟢 |
| Embedding/reranker model cache | `EMBEDDING_MODELS_DIR` / `RERANKER_MODELS_DIR` (host) | 🟢 | 🟢 |

All persistent state lives on **host bind mounts under `./data/`**, never in docker named volumes. `./data/` is gitignored. `docker compose down` and `docker volume prune` both leave project data untouched. To wipe, `rm -rf ./data/{postgres,memgraph,lightrag}`.

> **Why bind mounts not named volumes?** I `docker volume prune` regularly during development; project state living in named volumes is just a footgun. Bind mounts are also easier to back up, inspect, and migrate.

---

## 📊 Quality roadmap

### Phase 3 swaps (status)

| Swap | Stage | Today | Result | Status |
|---|---|---|---|:---:|
| **3a. ParadeDB pg_search (real BM25)** | Stage 1 keyword | Was Postgres `ts_rank` (no IDF, no length norm) | True BM25 via Tantivy. Pre-swap `hit@5 = 0.47`; post-swap `hit@5 = 1.00` (+113%). | 🟢 done |
| **3c. Header-path prefix on chunks** | Ingest | Was title + metadata only | + markdown heading breadcrumb. Keyword `hit@1` 0.87 -> 0.93 on saturated smoke set; bigger lift expected on harder corpora. | 🟢 done |
| ~~3b. SOTA cross-encoder rerank swap~~ | Stage 3 rerank | BGE-reranker-v2-m3 | est. +3-8 NDCG@10 with bge-reranker-v2-gemma / Qwen3-Reranker-8B / mxbai-large-v2 | 🟧 deferred (see below) |

### 🟧 Deferred decisions

<details>
<summary><b>Qwen3-Embedding-8B (semantic): deferred</b></summary>

Open-weight embedding leader on MMTEB at **70.58** vs BGE-M3 around **64**. The 6-point lift is real but small relative to cost: 8B vs ~568M params is **14× larger**. VRAM jumps from ~16 GB host RAM (BGE-M3 on CPU TEI) to ~30-40 GB GPU TEI. Per-query embed latency rises from ~10-30 ms to **100-300 ms**. A swap also invalidates every existing vector and forces a full re-ingest.

**Decision: keep BGE-M3** until an eval shows the embedding channel is the bottleneck. If it does, Qwen3-Embedding-8B is the swap.
</details>

<details>
<summary><b>Cross-encoder rerank upgrade: shipped 2026-05-03 as opt-in GPU sidecars</b></summary>

The 3-mode rerank shape lets each request pick its own backend via `rerank_model` on `POST /v1/search`. Default behavior is byte-identical to before; the new modes are silent fall-back to default if the corresponding sidecar is not running.

| Mode | Backend | Hardware | Latency p50 (full `/v1/search`, candidates=50) | Quality vs default |
|---|---|---|---:|---|
| (omitted) / `"default"` | CPU TEI BGE-v2-m3 | CPU | 553 ms | baseline |
| `"bge-gpu"` | Same BGE-v2-m3 on ROCm | AMD gfx1151 | 65 ms | identical (within fp16 noise) |
| `"qwen-4b"` | Qwen3-Reranker-4B on ROCm | AMD gfx1151 | 291 ms | LM-style yes/no scoring; ranks subtly-relevant docs higher |

Both GPU sidecars use the same 1.97 GiB image (`reranker-rocm/`) on `ubuntu:rolling` + ROCm gfx1151 prerelease wheels + `sentence-transformers >= 5.4`. Qwen pin: model commit `22e683669bc0f0bd69640a1354a6d0aebcfeede5` (the 2026-04-16 ST integration). VRAM use: BGE ~1.5 GiB, Qwen ~10 GiB; both fit comfortably alongside a vLLM at `gpu-memory-utilization 0.55` in 128 GiB UMA.

**Why kept the 8B candidate out:** Qwen3-Reranker-8B is broken on vLLM gfx1151 today (lemonade-sdk/vllm-rocm #3 EngineCore HIP failure; vLLM #21681 random scores). Qwen-4B sits one notch below 8B on quality but ships clean.

**Why a custom FastAPI sidecar instead of TEI/vLLM:** TEI can't load any SOTA reranker (encoder-only loader). vLLM's reranker sidecar mode is unstable on gfx1151 V1 (#32180) and would compete with the host LLM for the same GPU. ~150 LOC of FastAPI + `CrossEncoder.predict()` is enough.
</details>

<details>
<summary><b>Anthropic Contextual Retrieval: shipped as opt-in flag (was deferred to wrapper, now in engine)</b></summary>

CR is an ingest-time content transformation: a cheap LLM generates a 50-100 token chunk-specific blurb, prepended to each chunk before embedding and BM25 indexing. Anthropic reports **35% / 49% / 67%** reduction in retrieval failures (alone, with BM25, with rerank).

The deferral originally placed CR in the consumer (knowledge-base wrapper). Reconsidered: the engine already has an LLM dependency for LightRAG ingest and query-time entity extraction; adding CR as a third user of the same dependency is cheap, and putting it inline avoids the chunk-boundary coordination problem the wrapper-side approach would have. **Shipped 2026-04-27 as an opt-in flag**: `POST /v1/documents` with `{"contextual_retrieval": true}` runs CR; default false preserves byte-identical behavior for callers who don't opt in. vLLM's automatic prefix caching shares the document KV across the per-chunk calls, so cost stays at ~$1 per 1M document tokens.

The prompt body lives in `api/prompts/contextual_retrieval.md` and is editable without touching Python. Bring up the api with `LLM_BASE_URL` reachable; per-chunk LLM failure degrades gracefully (that chunk gets no blurb, ingest still succeeds).

**Deferred follow-on (still open)**: per-purpose LLM env vars (`LLM_INGEST_*` / `LLM_QUERY_*` / `LLM_BLURBER_*`) so callers can route CR blurbing to a small fast model while keeping the big model for entity extraction. Today CR shares `LLM_BASE_URL` with the rest, so blurbing on a 27B reasoning model is slower than necessary. ~50 LOC additive change with no breakage when revisited.
</details>

### Do the deferred-quality losses compound?

**No, not linearly.** The two channels lost play different roles:

- **Stage 1 (semantic + keyword + graph fused via RRF)** governs *recall*: "did we get the right doc into the candidate pool at all?"
- **Stage 2 (rerank)** governs *precision*: "given recall, is it ranked #1?"

`hit@1 = P(recall) × P(rerank ranks it #1 | recall)`. Losing the embedding lift reduces `P(recall)` for the semantic channel only; BM25 + graph still fire and RRF fuses them. Losing the rerank lift reduces `P(rank #1 | recall)` directly with no fallback. A naive sum predicts 6+5 = 11% end-to-end loss; realistic figure is closer to **3-7% hit@1**.

### Metric definitions

- **`hit@K`**: fraction of test queries where the expected doc appeared in top K. `0.47` = 47%.
- **`NDCG@10`**: 0-1 score for top-10 ordering quality vs ideal. BEIR averages NDCG@10 across 13 datasets.
- **`MRR`**: mean reciprocal rank of the expected doc. 1.0 if always rank 1; 0 if not in top K.

Baselines per-channel in `tests/golden/eval_history.jsonl`. Rerun with `python tests/baseline/run_baseline.py --phase N` after any swap.

---

## 🛠️ Configuration

All via `.env`. Full annotated template at [`.env.template`](.env.template).

### Required

| Variable | Description |
|---|---|
| `POSTGRES_PASSWORD` | Database password |
| `EMBEDDING_MODELS_DIR` | Absolute host path for embedding model cache |
| `RERANKER_MODELS_DIR` | Absolute host path for reranker model cache |

### Models

| Variable | Default | Description |
|---|---|---|
| `EMBEDDING_MODEL` | `Snowflake/snowflake-arctic-embed-l-v2.0` | HF embedding model id (1024d expected) |
| `RERANK_MODEL` | `BAAI/bge-reranker-v2-m3` | HF reranker model id (default CPU TEI) |
| `BGE_GPU_RERANKER_URL` | *(empty)* | Set to `http://reranker-bge-gpu:80` after starting `--profile rerank-bge-gpu`. Enables `rerank_model: "bge-gpu"`. |
| `QWEN_RERANKER_URL` | *(empty)* | Set to `http://reranker-qwen:80` after starting `--profile rerank-qwen`. Enables `rerank_model: "qwen-4b"`. |
| `QWEN_8B_RERANKER_URL` | *(empty)* | Set to `http://reranker-qwen-8b:80` after starting `--profile rerank-qwen-8b`. Enables `rerank_model: "qwen-8b"`. |
| `LLM_BASE_URL` | `http://host.docker.internal:8000` | OpenAI Responses API endpoint (LightRAG entity extraction) |
| `LLM_MODEL` | `Qwen3.6-27B-AWQ4` | Model id for the LLM endpoint |
| `LLM_API_KEY` | *(empty)* | Bearer token if needed |
| `LLM_ENABLE_THINKING` | `false` | When false, sends `chat_template_kwargs.enable_thinking=false`. **50x latency improvement** on Qwen3.6-27B-AWQ4 (684 to 3 output tokens, 33 s to 0.7 s). |

### Tuning

| Variable | Default | Description |
|---|---|---|
| `HNSW_M` | `16` | HNSW build param (higher = denser graph) |
| `HNSW_EF_CONSTRUCTION` | `64` | HNSW build accuracy |
| `HNSW_EF_SEARCH` | `100` | HNSW query-time accuracy |
| `DEFAULT_SEARCH_TOP_K` | `20` | Default `top_k` |
| `DEFAULT_RERANK_CANDIDATES` | `50` | RRF candidates passed to reranker |
| `CHUNK_SIZE` | `512` | Words per chunk |
| `CHUNK_OVERLAP` | `50` | Overlap words between chunks |

### Ports (all bound to 127.0.0.1)

| Service | Default | Env var |
|---|---:|---|
| API | `5050` | `API_PORT` |
| Postgres | `5433` | `POSTGRES_PORT` |
| TEI Embed | `8081` | `EMBEDDING_PORT` |
| TEI Rerank (CPU) | `8082` | `RERANK_PORT` |
| GPU Rerank: BGE | `8083` | `BGE_GPU_RERANK_PORT` |
| GPU Rerank: Qwen-4B | `8084` | `QWEN_RERANK_PORT` |
| GPU Rerank: Qwen-8B | `8085` | `QWEN_8B_RERANK_PORT` |
| Memgraph | `7687` | `MEMGRAPH_PORT` |

> Port 8000 is deliberately avoided (common for vLLM and other local services).

---

## 🧪 Tests

```bash
# Full suite (131 tests: 123 fast + 7 slow + 1 environment-conditional skip)
pytest

# Just the GPU reranker sidecar unit tests (no torch / no GPU required)
pytest reranker-rocm/tests/

# Fast iteration (deselect slow LightRAG ingest tests)
pytest -m 'not slow'

# Integration validation runner (step-by-step retrieval matrix, ctrl-C-able)
python tests/integration_validation/run.py ingest         # load corpus
python tests/integration_validation/run.py ingest-graph   # also feed multi-hop docs to LightRAG
python tests/integration_validation/run.py step lexical
python tests/integration_validation/run.py step semantic
python tests/integration_validation/run.py step hybrid_norerank
python tests/integration_validation/run.py step hybrid_rerank
python tests/integration_validation/run.py step hybrid_graph_rerank
python tests/integration_validation/run.py step hybrid_graph_norerank
python tests/integration_validation/run.py step graph_only
python tests/integration_validation/run.py step header_ablation
```

State persists across invocations in `tests/integration_validation/results/state.json`. Findings: `tests/integration_validation/REPORT.md`.

---

## 📦 Backup / restore

```bash
./scripts/backup.sh                        # dumps to backup_YYYYMMDD_HHMMSS.sql
./scripts/restore.sh backup_20260411.sql   # restores from file (destructive!)
```

Or just `tar czf data-snapshot.tar.gz ./data/` for a complete on-disk snapshot (postgres + memgraph + lightrag in one shot).

---

## 🤖 LLM-ready

This repo includes [`llm.txt`](llm.txt), a structured technical reference for LLM agents and AI-assisted development:
- Full API spec (request/response examples for every endpoint)
- Configuration reference
- Startup behavior + healthcheck semantics
- Verified behavior catalog
- Known limitations

Point your agent at `llm.txt` for complete, current context.

---

## 📁 Project structure

```
rag-base/
├── docker-compose.yml      # All 5 services
├── .env.template           # Config template (committed)
├── .env                    # Your config (gitignored)
├── data/                   # Bind mounts (gitignored): postgres, memgraph, lightrag
├── postgres/               # Schema (init.sql) + container README
├── embedding/              # TEI embed container README
├── reranker/               # TEI rerank (CPU) container README
├── reranker-rocm/          # GPU rerank sidecar (Dockerfile + server.py + tests). One image, two roles (bge-gpu, qwen-4b) selected by RERANK_MODEL.
├── memgraph/               # Memgraph container README
├── api/                    # FastAPI app (only custom code)
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── prompts/            # User-editable LLM prompt templates (.md, with {{var}} substitution)
│   │   ├── contextual_retrieval.md
│   │   └── query_entity_extraction.md
│   └── src/
│       ├── routers/        # FastAPI routes (search, graph_search, documents, ...)
│       ├── services/       # chunking, embedding, fusion, lightrag_store, ner, prompts, ...
│       └── models/         # Pydantic request/response models
├── tests/                  # Unit + integration tests, golden set, integration_validation runner
├── scripts/                # Backup/restore
├── llm.txt                 # Full technical reference (LLM-ready)
└── README.md               # This file
```

---

## 📜 License

MIT.
