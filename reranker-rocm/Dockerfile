# Minimal GPU reranker sidecar for AMD Strix Halo (gfx1151).
# Mirrors the openclaw-strix-embed pattern: ubuntu:rolling base + uv +
# ROCm gfx1151 prerelease wheels + sentence-transformers + FastAPI.
# Single image; the model is selected at runtime by RERANK_MODEL env.

FROM ubuntu:rolling

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HSA_OVERRIDE_GFX_VERSION=11.5.1 \
    HSA_NO_SCRATCH_RECLAIM=1 \
    MIOPEN_FIND_MODE=FAST \
    PYTORCH_HIP_ALLOC_CONF=expandable_segments:True \
    TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1

# libatomic1 / libgomp1 / libnuma1 are runtime deps the ROCm SDK loads via
# ctypes; without them torch import fails with "libatomic.so.1: cannot open
# shared object file". curl + ca-certificates are for the uv installer.
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates libatomic1 libgomp1 libnuma1 \
    && rm -rf /var/lib/apt/lists/*

# uv (fast resolver/installer)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /app

RUN uv venv /opt/venv --python 3.12
ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:${PATH}"

# torch + torchvision from the gfx1151 prerelease index. Kept in its own
# layer so requirements.txt churn does not invalidate the (large) torch layer.
RUN uv pip install --pre torch torchvision \
        --index-url https://rocm.prereleases.amd.com/whl/gfx1151/

COPY requirements.txt /app/requirements.txt
RUN uv pip install -r /app/requirements.txt

COPY server.py /app/server.py

ENV MODEL_CACHE=/data/models \
    RERANK_BATCH_SIZE=8 \
    LOG_LEVEL=INFO

EXPOSE 80

# Long start-period: first boot downloads the model from HF (4B is ~8 GiB).
HEALTHCHECK --interval=30s --timeout=10s --start-period=600s --retries=5 \
    CMD curl -fsS http://localhost:80/health || exit 1

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "80"]
