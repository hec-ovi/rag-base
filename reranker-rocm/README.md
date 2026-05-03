# reranker-rocm

GPU reranker sidecar for AMD Strix Halo (gfx1151). Speaks the TEI `/rerank`
shape, so the rag-base API client uses it unchanged.

One image, two supported roles, selected by env vars at container start:

| Role     | RERANK_MODEL                  | RERANK_REVISION                          | VRAM     |
|----------|-------------------------------|------------------------------------------|----------|
| bge-gpu  | BAAI/bge-reranker-v2-m3       | (latest)                                 | ~1.5 GiB |
| qwen-4b  | Qwen/Qwen3-Reranker-4B        | 22e683669bc0f0bd69640a1354a6d0aebcfeede5 | ~10 GiB  |

The Qwen revision is the `sentence-transformers >= 5.4` integration commit
(2026-04-16) that lets the standard `CrossEncoder` interface drive the
yes/no logit scoring head. Older revisions will not load with this server.

## Build

```
docker build -t reranker-rocm:local reranker-rocm/
```

## Run (BGE on GPU)

```
docker run --rm --device /dev/kfd --device /dev/dri \
    -e RERANK_MODEL=BAAI/bge-reranker-v2-m3 \
    -v "$RERANKER_MODELS_DIR":/data/models \
    -p 127.0.0.1:8083:80 reranker-rocm:local
```

## Run (Qwen-4B on GPU)

```
docker run --rm --device /dev/kfd --device /dev/dri \
    -e RERANK_MODEL=Qwen/Qwen3-Reranker-4B \
    -e RERANK_REVISION=22e683669bc0f0bd69640a1354a6d0aebcfeede5 \
    -v "$RERANKER_MODELS_DIR":/data/models \
    -p 127.0.0.1:8084:80 reranker-rocm:local
```

## API

`POST /rerank`

```
{ "query": "...", "texts": ["...", "..."], "raw_scores": false, "return_text": false }
```

Returns:

```
[ {"index": 0, "score": 0.95}, {"index": 1, "score": 0.12} ]
```

Sorted by score descending. With `return_text: true` each entry also carries
its original text.

`GET /health` returns 200 once the model is loaded; 503 otherwise.

## Environment

| Var                | Default        | Notes                                                       |
|--------------------|----------------|-------------------------------------------------------------|
| RERANK_MODEL       | (required)     | Hugging Face id, e.g. BAAI/bge-reranker-v2-m3               |
| RERANK_REVISION    | (none)         | Pin to a specific commit. Required for Qwen3-Reranker-4B.   |
| MODEL_CACHE        | /data/models   | Where HF cache lives. Bind-mount your host dir.             |
| RERANK_BATCH_SIZE  | 8              | predict() batch size                                        |
| LOG_LEVEL          | INFO           | uvicorn / module log level                                  |

ROCm env vars (`HSA_OVERRIDE_GFX_VERSION=11.5.1`, `HSA_NO_SCRATCH_RECLAIM=1`,
`MIOPEN_FIND_MODE=FAST`, `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True`,
`TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`) are baked into the image. They
mirror what the user's vllm-awq4-qwen container uses on the same hardware.

## Tests

```
pytest reranker-rocm/tests/
```

Unit tests run without torch or ROCm by injecting a fake CrossEncoder; the
real model never gets loaded, so the suite works on any machine.
