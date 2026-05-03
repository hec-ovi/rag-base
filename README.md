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

**5 containers. 4 prebuilt images, 1 custom API.**

| Service | Image | Role | Profile |
|---|---|---|---|
| 🟦 **Postgres + pgvector + pg_search** | `paradedb/paradedb:0.23.1-pg17` | Documents, chunks, vectors. HNSW for semantic + Tantivy BM25 for keyword. | required |
| 🟨 **TEI Embed** | `text-embeddings-inference:cpu-1.9` | Text -> 1024d vectors via [HuggingFace TEI](https://github.com/huggingface/text-embeddings-inference). | required |
| 🟨 **TEI Rerank** | `text-embeddings-inference:cpu-1.9` | Cross-encoder rerank. Same TEI image, different model. | optional (`--profile rerank`) |
| 🟧 **Memgraph MAGE** | `memgraph/memgraph-mage:3.9.0` | Knowledge graph: LightRAG entities (`:base`) + custom concepts (`:Concept`). Cypher, PageRank, Louvain, BFS. | optional (`--profile graph`) |
| 🟪 **API** | Custom (`python:3.12-slim`) | Orchestrator. FastAPI + LightRAG (ingest-time entity extraction) + GLiNER (query-time NER). | required |

> Reranker and Memgraph are **fully optional**. The api detects their absence at startup and degrades gracefully. Hybrid + rerank still work without graph; semantic + BM25 still work without rerank.

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

# Everything
docker compose --profile rerank --profile graph up -d
```

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
| 🟢 Cross-encoder rerank (last stage) | shipped (CPU; ~17-29 s/query, see [Deferred decisions](#-deferred-decisions)) |
| 🟢 LightRAG entity + relation extraction at ingest -> Memgraph | shipped (LightRAG 1.4.15+, CVE-patched) |
| 🟢 GLiNER NER for graph-only fast mode (no LLM at query time) | shipped |
| 🟢 Graph channel results actually merged into RRF (closes the historical search.py:66 TODO) | shipped |
| 🟢 Graceful degradation: api never crashes on missing optional services | shipped |
| 🟢 Anthropic Contextual Retrieval (`contextual_retrieval: true` flag on `POST /v1/documents`) | shipped, opt-in; per-chunk LLM blurb prepended to `indexed_content`, vLLM auto-caches the document prefix |
| 🟢 Editable prompt files in `api/prompts/*.md` (decoupled from Python code) | shipped; covers CR + query-time entity extraction, restart api to reload |
| 🟢 Test gates: every SOTA mechanism has a "did it actually fire?" test | shipped (115 tests passing: 107 fast + 7 slow + 1 environment-conditional skip) |
| 🟡 LightRAG ingest is ~1-3 min per chunk on local 27B reasoning LLM | by design; bulk-ingest path uses `X-LightRAG-Ingest: false` header |
| 🟡 Cross-encoder rerank is CPU-bound and slow at multi-second p50 | by design; GPU TEI is a tag swap |
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
<summary><b>Cross-encoder rerank upgrade: deferred</b></summary>

| Model | Size | BEIR NDCG@10 | Lift | License | Hosting |
|---|---:|---:|---:|---|---|
| `BGE-reranker-v2-m3` (current) | 568 M | 51.8 | baseline | Apache-2.0 | TEI 🟢 |
| `mxbai-rerank-base-v2` | 0.5 B | 55.57 | +3.77 | Apache-2.0 | vLLM (hf-overrides) |
| `BAAI/bge-reranker-v2-gemma` | 2 B | est. 55-57 | est. +3-5 | Apache-2.0 | vLLM (TEI can't load Gemma) |
| `mxbai-rerank-large-v2` | 1.5 B | 57.49 | +5.69 | Apache-2.0 | vLLM (hf-overrides) |
| `Qwen3-Reranker-8B` | 8 B | est. 58+ | est. +6-8 | Apache-2.0 | vLLM |
| `ZeroEntropy zerank-2` | 4 B | not pub | unknown | **CC-BY-NC-4.0** (no commercial) | hosted |

TEI cannot load any SOTA candidate (loader is restricted to encoder-only families). The clean drop-in is a vLLM rerank sidecar with Cohere-compatible `/v1/rerank`. With the host LLM already at ~92/118 GB VRAM, a second vLLM rerank container would need `--gpu-memory-utilization` ~0.10 and could still see prefill spikes.

**Decision: keep BGE-reranker-v2-m3 on CPU TEI** until VRAM pressure relaxes (smaller LLM, off-host LLM, more GPU) OR a harder corpus reveals rerank as the bottleneck. Then revisit with `BGE-reranker-v2-gemma` first, then `Qwen3-Reranker-8B`.
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
| `RERANK_MODEL` | `BAAI/bge-reranker-v2-m3` | HF reranker model id |
| `LLM_BASE_URL` | `http://host.docker.internal:8000` | OpenAI Responses API endpoint (LightRAG entity extraction) |
| `LLM_MODEL` | `Qwen3.6-27B-AWQ4` | Model id for the LLM endpoint |
| `LLM_API_KEY` | *(empty)* | Bearer token if needed |
| `LLM_ENABLE_THINKING` | `false` | When false, sends `chat_template_kwargs.enable_thinking=false`. **50× latency improvement** on Qwen3.6-27B-AWQ4 (684 -> 3 output tokens, 33 s -> 0.7 s). |

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
| TEI Rerank | `8082` | `RERANK_PORT` |
| Memgraph | `7687` | `MEMGRAPH_PORT` |

> Port 8000 is deliberately avoided (common for vLLM and other local services).

---

## 🧪 Tests

```bash
# Full suite (115 tests: 107 fast + 7 slow + 1 environment-conditional skip)
pytest

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
├── reranker/               # TEI rerank container README
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
├── SHIP.md                 # Audit + ship report (2026-04-27)
└── README.md               # This file
```

---

## 📜 License

MIT.
