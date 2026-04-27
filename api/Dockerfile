FROM python:3.12-slim-bookworm

WORKDIR /app

# System deps for asyncpg (needs libpq headers to build)
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc libpq-dev curl && \
    rm -rf /var/lib/apt/lists/*

# Phase 5 (graph-only fast mode): install CPU-only torch ahead of the rest.
# Default torch wheel is CUDA-enabled and ~2 GB. We do not use a GPU in this
# container (TEI handles the embed and rerank; api just runs NER on CPU), so
# the smaller CPU-only wheel is correct. Installed before requirements.txt so
# that gliner's transitive torch dep is satisfied without re-pulling CUDA.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
    "torch>=2.4,<3"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Phase 5: pre-download the GLiNER multitask model into the HuggingFace cache
# at build time so the first /v1/search/graph request does not block on a
# ~190 MB model fetch. CPU-only; loaded lazily at runtime via api/src/services/ner.py.
RUN python -c "from gliner import GLiNER; GLiNER.from_pretrained('knowledgator/gliner-multitask-v1.0')"

COPY src/ ./src/
COPY prompts/ ./prompts/

EXPOSE 8000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--loop", "uvloop"]
