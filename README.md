# rag-base

> **LLM agents**: see [`llm.txt`](llm.txt) for the full technical reference with all endpoints, request/response examples, configuration, and verified behavior.

Standalone RAG backend. Hybrid search with embeddings, reranking, and knowledge graph.

One `docker compose up`. Any future project gets a working retrieval system.

## Why

Every RAG project needs the same infrastructure: vector storage, embeddings, reranking, search. This repo solves that once. Any future project (legal, medical, whatever) spins this up, connects to the API, and gets a full retrieval backend without writing any infrastructure code.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│              rag-base  (docker compose up)          │
│                                                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────┐   │
│  │ Postgres │  │ TEI      │  │ TEI      │  │Memgraph│   │
│  │ pgvector │  │ Embed    │  │ Rerank   │  │  MAGE  │   │
│  │  :5433   │  │  :8081   │  │  :8082   │  │ :7687  │   │
│  └──────────┘  └──────────┘  └──────────┘  └────────┘   │
│       ▲              ▲             ▲            ▲       │
│       └──────────────┴─────────────┴────────────┘       │
│                         │                               │
│              ┌──────────▼──────────┐                    │
│              │   API (FastAPI)     │                    │
│              │     :5050           │                    │
│              └─────────────────────┘                    │
└─────────────────────────────────────────────────────────┘
                      ▲
                      │  REST API (:5050)
                      │
              Any client application
```

**5 containers. 4 prebuilt images, 1 custom API.**

| Service | Image | Role |
|---|---|---|
| **Postgres + pgvector** | `pgvector/pgvector:0.8.2-pg17` | Stores documents, chunks, vectors. Handles semantic search (HNSW) and keyword search (tsvector). |
| **TEI Embed** | `text-embeddings-inference:cpu-1.9` | Turns text into 1024-dim vectors. [HuggingFace TEI](https://github.com/huggingface/text-embeddings-inference) is a prebuilt Rust server - same concept as vLLM but for embedding models. |
| **TEI Rerank** | `text-embeddings-inference:cpu-1.9` | Cross-encoder reranker. Takes a query + candidates and re-scores them by relevance. Same TEI image, different model. |
| **Memgraph MAGE** | `memgraph/memgraph-mage:3.9.0` | Knowledge graph engine. Stores concepts and relations. Supports Cypher queries, PageRank, community detection, shortest path. |
| **API** | Custom (`python:3.12-slim`) | The only custom code. Orchestrates all services: ingests documents, runs hybrid search, merges results. |

TEI Rerank and Memgraph are **optional** - the system works without them, just with reduced capabilities.

## Quickstart

```bash
git clone <repo-url> && cd rag-base
cp .env.template .env
```

Edit `.env` - three things need your attention:
```env
POSTGRES_PASSWORD=your-secure-password
EMBEDDING_MODELS_DIR=/absolute/path/to/your/models/embeddings
RERANKER_MODELS_DIR=/absolute/path/to/your/models/rerankers
```

Start the services:
```bash
# Core only (postgres + embedding + api)
docker compose up -d

# Add reranking (recommended - big quality boost)
docker compose --profile rerank up -d

# Add knowledge graph
docker compose --profile graph up -d

# Everything
docker compose --profile rerank --profile graph up -d
```

### First boot

TEI downloads the embedding model on first boot. This takes **1-3 minutes** depending on your internet. The API waits for TEI to finish before starting. Subsequent boots use the cached model and start in seconds.

If you enabled the rerank profile, the reranker model also downloads (~1.2 GB). Since the API only waits for postgres and embedding, it may start before the reranker finishes. Once the reranker is ready, restart the API to connect:

```bash
docker compose restart api
```

### Verify

```bash
curl http://localhost:5050/health
```
```json
{"status": "ok", "postgres": "connected", "embedding": "healthy", "reranker": "healthy", "memgraph": "connected"}
```

Optional services show `"disabled"` if not running. Status is `"ok"` as long as required services (postgres, embedding) are healthy.

## How it works

### Ingest

Send a document. The API chunks it, embeds each chunk, and stores everything atomically.

```bash
curl -X POST http://localhost:5050/v1/documents \
  -H "Content-Type: application/json" \
  -d '{
    "title": "React Documentation",
    "content": "React is a JavaScript library for building user interfaces...",
    "metadata": {"source": "docs", "language": "en"}
  }'
```

What happens internally:
1. Text is split into chunks (512 words, paragraph boundaries, configurable)
2. Each chunk is embedded via TEI (1024-dim vector)
3. Document + chunks + vectors + tsvector stored in Postgres in a single transaction

If embedding fails, nothing is stored - no orphan documents.

### Search

Send a query. The API runs up to 4 retrieval strategies in parallel, merges results, and returns the best matches.

```bash
curl -X POST http://localhost:5050/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query": "frontend frameworks", "top_k": 10}'
```

What happens internally:
1. Query is embedded via TEI
2. Three retrievers run **in parallel**:
   - **Semantic**: pgvector cosine similarity over chunk embeddings
   - **Keyword**: PostgreSQL full-text search (tsvector/BM25-style)
   - **Graph**: expand entities from query via Memgraph neighbors (if enabled)
3. Results merged via **RRF** (Reciprocal Rank Fusion) - a standard formula that boosts results found by multiple retrievers
4. Top candidates re-scored by the **cross-encoder reranker** (if enabled)
5. Final top K returned

You can also hit individual retrievers directly:
- `POST /v1/search/semantic` - vector only
- `POST /v1/search/keyword` - keyword only

### Knowledge graph

The graph stores concepts (nodes) and relations (edges). It's separate from document storage - you build it explicitly.

```bash
# Create concepts
curl -X POST http://localhost:5050/v1/concepts \
  -H "Content-Type: application/json" \
  -d '{"name": "React", "type": "Framework", "description": "JavaScript UI library"}'

curl -X POST http://localhost:5050/v1/concepts \
  -H "Content-Type: application/json" \
  -d '{"name": "JavaScript", "type": "Language", "description": "Programming language"}'

# Create a relation
curl -X POST http://localhost:5050/v1/relations \
  -H "Content-Type: application/json" \
  -d '{"source_name": "React", "target_name": "JavaScript", "relation_type": "DEPENDS_ON"}'

# Traverse: what's within 2 hops of React?
curl http://localhost:5050/v1/graph/neighbors/{concept_id}?depth=2

# Shortest path between two concepts
curl http://localhost:5050/v1/graph/path/{from_id}/{to_id}

# Community detection (Louvain clustering)
curl http://localhost:5050/v1/graph/communities

# Graph stats
curl http://localhost:5050/v1/graph/stats
```

Concepts are upserted by name - posting the same name twice updates the existing node instead of creating a duplicate. Self-referencing relations are blocked.

## API reference

Auto-generated OpenAPI docs at `http://localhost:5050/docs`.

| Method | Endpoint | Purpose |
|---|---|---|
| **Documents** | | |
| `POST` | `/v1/documents` | Ingest document (auto-chunk, embed, store) |
| `GET` | `/v1/documents?offset=0&limit=20` | List documents (paginated) |
| `GET` | `/v1/documents/{id}` | Get document with chunks |
| `DELETE` | `/v1/documents/{id}` | Delete document + chunks (cascade) |
| **Search** | | |
| `POST` | `/v1/search` | Hybrid search (semantic + keyword + graph + rerank) |
| `POST` | `/v1/search/semantic` | Vector-only search |
| `POST` | `/v1/search/keyword` | Keyword-only search |
| **Graph** | | |
| `POST` | `/v1/concepts` | Create/update concept (upsert by name) |
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
| `GET` | `/health` | All services status |
| `GET` | `/health/models` | Loaded model info |

### Search request fields

```json
{
  "query": "your search query",
  "top_k": 20,
  "rerank": true,
  "rerank_candidates": 50,
  "min_score": 0.0,
  "include_graph": true
}
```

- `top_k`: number of results to return (1-100, default 20)
- `rerank`: whether to use cross-encoder reranking (default true, falls back to RRF-only if reranker is disabled)
- `rerank_candidates`: how many RRF results to send to the reranker before cutting to top_k (default 50)
- `min_score`: minimum cosine similarity for semantic results (default 0.0)
- `include_graph`: whether to include graph expansion in retrieval (default true, ignored if Memgraph is disabled)

## Configuration

All via `.env`. See `.env.template` for the full list with defaults.

### Key settings

| Variable | Default | Description |
|---|---|---|
| `POSTGRES_PASSWORD` | *(required)* | Database password |
| `EMBEDDING_MODELS_DIR` | *(required)* | Absolute host path for embedding model cache |
| `RERANKER_MODELS_DIR` | *(required)* | Absolute host path for reranker model cache |
| `EMBEDDING_MODEL` | `Snowflake/snowflake-arctic-embed-l-v2.0` | HuggingFace embedding model ID |
| `RERANK_MODEL` | `BAAI/bge-reranker-v2-m3` | HuggingFace reranker model ID |
| `CHUNK_SIZE` | `512` | Words per chunk (lower = more precise retrieval, more chunks) |
| `CHUNK_OVERLAP` | `50` | Overlap words between chunks |
| `DEFAULT_SEARCH_TOP_K` | `20` | Default results returned |
| `DEFAULT_RERANK_CANDIDATES` | `50` | Candidates passed to reranker |

### Ports

| Service | Default | Env var |
|---|---|---|
| Postgres | 5433 | `POSTGRES_PORT` |
| TEI Embed | 8081 | `EMBEDDING_PORT` |
| TEI Rerank | 8082 | `RERANK_PORT` |
| Memgraph | 7687 | `MEMGRAPH_PORT` |
| API | 5050 | `API_PORT` |

All bound to `127.0.0.1` (localhost only). Postgres defaults to 5433 to avoid conflicts with any existing Postgres on the host. Port 8000 is deliberately avoided (common for other services).

### Swap models

Change `EMBEDDING_MODEL` or `RERANK_MODEL` in `.env`, restart. TEI downloads the new model on boot.

**Important**: if you change the embedding model, existing vectors become invalid. You must re-ingest all documents.

## Persistence

| What | Where | Survives restart? |
|---|---|---|
| Documents + chunks + vectors | Docker volume `ragbase_pgdata` | Yes |
| Graph data (concepts, relations) | Docker volume `ragbase_mgdata` | Yes |
| Model cache | Host dirs (`EMBEDDING_MODELS_DIR`, `RERANKER_MODELS_DIR`) | Yes |

`docker compose down` preserves all data. Only `docker compose down -v` removes volumes.

## Backup / Restore

```bash
./scripts/backup.sh                        # dumps to backup_YYYYMMDD_HHMMSS.sql
./scripts/restore.sh backup_20260411.sql   # restores from file (destructive!)
```

## Project structure

```
rag-base/
├── docker-compose.yml      # All 5 services
├── .env.template           # Config template (committed)
├── .env                    # Your config (gitignored)
├── postgres/               # Schema + container README
├── embedding/              # TEI embed container README
├── reranker/               # TEI rerank container README
├── memgraph/               # Memgraph container README
├── api/                    # FastAPI app (only custom code)
│   ├── Dockerfile
│   ├── requirements.txt
│   └── src/
├── tests/                  # Unit + integration tests
├── scripts/                # Backup/restore
├── llm.txt                 # Full technical reference (LLM-ready)
└── README.md               # This file
```

## LLM-ready

This repo includes [`llm.txt`](llm.txt), a structured technical reference designed for LLM agents and AI-assisted development. It contains the full API specification with request/response examples, configuration reference, startup behavior, verified behavior, and known limitations. Point your agent at `llm.txt` for complete context about this project.
