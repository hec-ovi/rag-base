# reranker

Hugging Face Text Embeddings Inference (TEI) — serves the cross-encoder reranking model.

**Image:** `ghcr.io/huggingface/text-embeddings-inference:cpu-1.9`
**Port:** 8082 → container port 80 (configurable via `RERANK_PORT`)

## What it does

Takes a query and a list of candidate texts, scores each candidate by relevance, returns them sorted. This is the final quality pass after vector + keyword search results are merged.

## Default model

`BAAI/bge-reranker-v2-m3` (568M params, multilingual). Configurable via `RERANK_MODEL` in `.env`.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/rerank` | Rerank candidates against a query |
| `GET` | `/health` | Health check |
| `GET` | `/info` | Loaded model info |

### Rerank request format

```json
{
  "query": "frontend frameworks",
  "texts": ["React is a JS library...", "PostgreSQL is a database..."],
  "raw_scores": false
}
```

### Rerank response

```json
[
  {"index": 0, "score": 0.95},
  {"index": 1, "score": 0.12}
]
```

## Model cache

Models are downloaded to the host directory set by `RERANKER_MODELS_DIR` (mounted as `/data` in the container).

## Optional

This container only starts with the `rerank` profile. To enable: set `RERANK_ENABLED=true` and run `docker compose --profile rerank up -d`. Or to disable reranking entirely, just don't activate the profile.
