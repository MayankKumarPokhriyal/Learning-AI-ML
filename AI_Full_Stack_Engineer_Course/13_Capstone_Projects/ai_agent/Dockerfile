# syntax=docker/dockerfile:1
# Build the paper corpus first (data/papers.sqlite must exist: python -m research_agent.ingest), then:
#   docker build -t research-agent:1.0.0 .
#   docker run --rm -p 8000:8000 -e LLM_BASE_URL=http://host.docker.internal:8080/v1 research-agent:1.0.0

# ---------- stage 1: builder — uv, caches and build tools stay here ----------
FROM python:3.12-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.10.3 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /build
# Dependencies first: this layer is reused until requirements.lock changes
COPY requirements.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv /opt/venv && \
    uv pip install --python /opt/venv/bin/python -r requirements.lock
# Then the package itself (changes more often)
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/venv/bin/python --no-deps .

# ---------- stage 2: runtime — Python, the venv, and the paper corpus only ----------
FROM python:3.12-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    AGENT_DATA_DIR=/srv/data \
    ARXIV_MODE=record \
    LLM_BASE_URL=http://host.docker.internal:8080/v1
RUN groupadd --system --gid 10001 app && useradd --system --uid 10001 --gid app --no-create-home app && \
    mkdir -p /srv/data && chown app:app /srv/data
WORKDIR /srv
COPY --from=builder /opt/venv /opt/venv
# The corpus is baked in (the image tag pins code + data). State, traces and reports are written next to it:
# mount a volume on /srv/data to keep them across container restarts.
COPY --chown=app:app data/papers.sqlite /srv/data/papers.sqlite
USER app
EXPOSE 8000
# Liveness only: /ready also checks the external LLM server, and restarting this container would not fix that.
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --start-interval=2s --retries=3 \
    CMD ["python", "-c", "import sys, urllib.request; sys.exit(urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2).status != 200)"]
CMD ["uvicorn", "research_agent.api:app", "--host", "0.0.0.0", "--port", "8000"]
