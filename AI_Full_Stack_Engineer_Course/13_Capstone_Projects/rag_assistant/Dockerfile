# syntax=docker/dockerfile:1
# Download and ingest first (data/corpus/ and data/index/ must exist), then:
#   docker build -t rag-assistant:1.0.0 .
#   docker run --rm -p 8000:8000 -e LLM_BASE_URL=http://host.docker.internal:8080/v1 -e RAG_INGEST_API_KEY=change-me rag-assistant:1.0.0

# ---------- stage 1: builder — uv, caches and model downloads stay here ----------
FROM python:3.12-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.10.3 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /build
# A C compiler for dependencies without a prebuilt wheel on this platform (e.g. PyStemmer on linux/aarch64,
# which is what Docker Desktop runs on Apple Silicon). Builder stage only: the runtime image stays slim.
RUN apt-get update && apt-get install -y --no-install-recommends gcc libc6-dev && rm -rf /var/lib/apt/lists/*
# Dependencies first: this layer is reused until requirements.lock changes. CPU-only PyTorch keeps CUDA wheels out of the image.
COPY requirements.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv /opt/venv && \
    uv pip install --python /opt/venv/bin/python --torch-backend cpu -r requirements.lock
# Then the package itself (changes more often)
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/venv/bin/python --no-deps .
# Bake the embedding model, the reranker and the tokenizer file into the image: no downloads at startup
ENV HF_HOME=/opt/models/hf TIKTOKEN_CACHE_DIR=/opt/models/tiktoken
RUN /opt/venv/bin/python -m rag_assistant warmup

# ---------- stage 2: runtime — Python, the venv, the models, the corpus and the index only ----------
FROM python:3.12-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    HF_HOME=/opt/models/hf \
    HF_HUB_OFFLINE=1 \
    TIKTOKEN_CACHE_DIR=/opt/models/tiktoken \
    TOKENIZERS_PARALLELISM=false \
    RAG_DATA_DIR=/srv/data
RUN groupadd --system --gid 10001 app && useradd --system --uid 10001 --gid app --no-create-home app
WORKDIR /srv
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/models /opt/models
# The pinned corpus (needed by POST /ingest) and the index built from it; traces go to /srv/data/traces
COPY --chown=app:app data/corpus/ ./data/corpus/
COPY --chown=app:app data/index/ ./data/index/
RUN mkdir -p /srv/data/traces && chown -R app:app /srv/data
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=60s --start-interval=2s --retries=3 \
    CMD ["python", "-c", "import sys, urllib.request; sys.exit(urllib.request.urlopen('http://127.0.0.1:8000/ready', timeout=2).status != 200)"]
CMD ["uvicorn", "rag_assistant.api:app", "--host", "0.0.0.0", "--port", "8000"]
