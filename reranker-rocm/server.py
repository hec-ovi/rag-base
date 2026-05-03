"""GPU reranker sidecar (TEI-compatible).

Single image, two supported roles, picked by env vars at container start:

  bge-gpu : RERANK_MODEL=BAAI/bge-reranker-v2-m3
  qwen-4b : RERANK_MODEL=Qwen/Qwen3-Reranker-4B
            RERANK_REVISION=22e683669bc0f0bd69640a1354a6d0aebcfeede5

Speaks the TEI /rerank shape so the rag-base API client is unchanged:

  POST /rerank   { query, texts[], raw_scores?, return_text? }
                 -> [ {index, score, text?} ]   (sorted desc)
  GET  /health   -> 200 {ok: true}  once the model is loaded
"""
from __future__ import annotations

import logging
import math
import os
from contextlib import asynccontextmanager
from typing import Any, Callable

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


def _maybe_sigmoid(scores: list[float]) -> list[float]:
    """Sigmoid-normalize when any score escapes [0, 1].

    BGE-style heads emit probabilities in [0, 1] already; pass them through.
    Qwen3-Reranker (LogitScore module in sentence-transformers >=5.4) emits
    raw logits that span the real line; sigmoid keeps the same rank order
    while giving callers the same [0, 1] contract as TEI.
    """
    if not scores:
        return scores
    if any(s < 0.0 or s > 1.0 for s in scores):
        return [1.0 / (1.0 + math.exp(-s)) for s in scores]
    return scores

logger = logging.getLogger("reranker-rocm")
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO").upper())


def _load_model() -> Any:
    """Load the CrossEncoder. Imports torch/sentence_transformers lazily so the
    module is importable in unit tests that have neither installed."""
    import torch
    from sentence_transformers import CrossEncoder

    model_id = os.environ["RERANK_MODEL"]
    revision = os.environ.get("RERANK_REVISION") or None
    cache = os.environ.get("MODEL_CACHE", "/data/models")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(
        "Loading CrossEncoder model=%s revision=%s device=%s",
        model_id, revision, device,
    )
    return CrossEncoder(
        model_id,
        revision=revision,
        device=device,
        cache_folder=cache,
        model_kwargs={"torch_dtype": "float16"},
    )


class RerankRequest(BaseModel):
    query: str
    texts: list[str]
    raw_scores: bool = False
    return_text: bool = False


def create_app(
    *,
    model: Any | None = None,
    model_loader: Callable[[], Any] | None = None,
) -> FastAPI:
    """Build the FastAPI app.

    Production: ``create_app()`` -- the lifespan calls ``_load_model``.
    Tests:
      ``create_app(model=fake_cross_encoder)`` -- lifespan no-ops.
      ``create_app(model_loader=lambda: None)`` -- /health returns 503.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if getattr(app.state, "model", None) is None:
            loader = model_loader or _load_model
            app.state.model = loader()
        app.state.batch_size = int(os.environ.get("RERANK_BATCH_SIZE", "8"))
        yield

    app = FastAPI(title="reranker-rocm", lifespan=lifespan)
    if model is not None:
        app.state.model = model

    @app.get("/health")
    def health():
        if getattr(app.state, "model", None) is None:
            raise HTTPException(status_code=503, detail="model not loaded")
        return {"ok": True}

    @app.post("/rerank")
    def rerank(body: RerankRequest):
        m = getattr(app.state, "model", None)
        if m is None:
            raise HTTPException(status_code=503, detail="model not loaded")
        if not body.texts:
            return []
        pairs = [(body.query, t) for t in body.texts]
        batch_size = getattr(app.state, "batch_size", 8)
        raw = m.predict(pairs, batch_size=batch_size, convert_to_numpy=True)
        scores = [float(s) for s in raw]
        if not body.raw_scores:
            scores = _maybe_sigmoid(scores)
        results: list[dict[str, Any]] = []
        for i, s in enumerate(scores):
            entry: dict[str, Any] = {"index": i, "score": s}
            if body.return_text:
                entry["text"] = body.texts[i]
            results.append(entry)
        return sorted(results, key=lambda r: r["score"], reverse=True)

    return app


app = create_app()
