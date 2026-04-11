# embedding

Hugging Face Text Embeddings Inference (TEI) - serves the embedding model.

**Image:** `ghcr.io/huggingface/text-embeddings-inference:cpu-1.9`
**Port:** 8081 → container port 80 (configurable via `EMBEDDING_PORT`)

## What it does

Turns text into vectors. The API sends it text, TEI returns a 1024-dimensional vector. No custom code - just a prebuilt image with a model name.

## Default model

`Snowflake/snowflake-arctic-embed-l-v2.0` (568M params, 1024 dims). Configurable via `EMBEDDING_MODEL` in `.env`.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/embed` | Embed text(s), returns float arrays |
| `POST` | `/v1/embeddings` | OpenAI-compatible embedding endpoint |
| `GET` | `/health` | Health check |
| `GET` | `/info` | Loaded model info |

## Model cache

Models are downloaded to the host directory set by `EMBEDDING_MODELS_DIR` (mounted as `/data` in the container). First boot downloads the model; subsequent boots use the cache.

## Swap model

Change `EMBEDDING_MODEL` in `.env`, restart the container. If you change the model, you must re-embed all existing chunks (different model = different vector space).
