"""GLiNER-based query-time NER for the Phase 5 graph-only endpoint.

GLiNER is a generalist bidirectional-encoder NER model that extracts user-supplied
entity types from text in a single parallel pass (no autoregressive generation,
no LLM). Per the research finding `.research/graph-only-retrieval-no-embedding-fast-mode/`,
this is the load-bearing replacement for the LLM at query time: an embedding-free,
CPU-friendly, sub-50 ms entity extractor whose output feeds Memgraph Cypher.

The wrapper here is intentionally thin:
- Lazy singleton: the model is loaded on first call, not at api startup, so the
  /health probe and the existing endpoints stay snappy.
- Async-safe: a single asyncio.Lock prevents two concurrent first-callers from
  loading the model twice.
- CPU-pinned: torch is installed CPU-only in the Dockerfile; we do NOT call
  `.to('cuda')`. Speeds and memory have been measured against this assumption.
- Stateless after load: subsequent calls just call `predict_entities`.
"""

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)


# Default entity inventory used when the caller does not supply one. Broad enough
# to cover the rag-base graph (people, orgs, places, products, concepts, dates,
# events, technical artifacts) without leaking the upstream LightRAG entity_type
# vocabulary, which we do not want clients to depend on.
DEFAULT_ENTITY_LABELS: list[str] = [
    "person",
    "organization",
    "location",
    "product",
    "concept",
    "technology",
    "event",
    "date",
    "work",
]


class NERService:
    """Lazy-loaded GLiNER wrapper. Construct once per process, share via app.state."""

    def __init__(
        self,
        model_name: str = "knowledgator/gliner-multitask-v1.0",
        default_threshold: float = 0.45,
        default_labels: list[str] | None = None,
    ) -> None:
        self.model_name = model_name
        self.default_threshold = default_threshold
        self.default_labels = list(default_labels or DEFAULT_ENTITY_LABELS)
        self._model: Any | None = None
        self._load_lock = asyncio.Lock()

    async def _ensure_loaded(self) -> Any:
        if self._model is not None:
            return self._model
        async with self._load_lock:
            if self._model is not None:
                return self._model
            logger.info("Loading GLiNER model: %s", self.model_name)
            # Import inside the method so `import api.src.services.ner` is cheap
            # (the GLiNER + torch import chain is heavy and we want it deferred
            # until a /v1/search/graph request actually arrives).
            from gliner import GLiNER

            # Load on the default executor so we do not block the event loop
            # for the ~2-5 s model load.
            loop = asyncio.get_running_loop()
            self._model = await loop.run_in_executor(
                None,
                lambda: GLiNER.from_pretrained(self.model_name),
            )
            logger.info("GLiNER model ready (%s)", self.model_name)
            return self._model

    async def extract(
        self,
        text: str,
        labels: list[str] | None = None,
        threshold: float | None = None,
    ) -> list[dict]:
        """Extract entities from `text`.

        Returns a list of dicts: {"text": str, "label": str, "score": float, "start": int, "end": int}.
        Empty list when the model finds nothing above threshold (NOT an error).
        """
        if not text or not text.strip():
            return []
        model = await self._ensure_loaded()
        labels = labels or self.default_labels
        threshold = threshold if threshold is not None else self.default_threshold

        loop = asyncio.get_running_loop()
        # GLiNER's predict_entities is sync; offload to default executor.
        raw = await loop.run_in_executor(
            None,
            lambda: model.predict_entities(text, labels, threshold=threshold),
        )
        # Normalize: ensure stable key set even if a future GLiNER version
        # returns extra fields.
        return [
            {
                "text": str(e.get("text", "")).strip(),
                "label": str(e.get("label", "")),
                "score": float(e.get("score", 0.0)),
                "start": int(e.get("start", 0)) if e.get("start") is not None else 0,
                "end": int(e.get("end", 0)) if e.get("end") is not None else 0,
            }
            for e in (raw or [])
            if e.get("text")
        ]
