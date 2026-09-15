# syntax=docker/dockerfile:1
# The BIRD databases are NOT baked into the image (CC BY-SA 4.0 data, ~110 MB): run `make data`, then mount them read-only.
#   docker build -t data-analyst-agents:1.0.0 .
#   docker run --rm -p 8000:8000 -v "$PWD/data/bird_minidev:/srv/bird:ro" -e LLM_BASE_URL=http://host.docker.internal:8080/v1 data-analyst-agents:1.0.0

# ---------- stage 1: builder — uv and build caches stay here ----------
FROM python:3.12-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.10.3 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /build
# Dependencies first: this layer is reused until requirements.lock changes
COPY requirements.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv /opt/venv && \
    uv pip install --python /opt/venv/bin/python -r requirements.lock
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/venv/bin/python --no-deps .

# ---------- stage 2: runtime — Python and the venv only, non-root ----------
FROM python:3.12-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    ANALYST_DATA_DIR=/srv/state \
    ANALYST_BIRD_DIR=/srv/bird \
    MPLBACKEND=Agg \
    LLM_BASE_URL=http://host.docker.internal:8080/v1
RUN groupadd --system --gid 10001 app && useradd --system --uid 10001 --gid app --no-create-home app && \
    mkdir -p /srv/state /srv/bird && chown app:app /srv/state
WORKDIR /srv
COPY --from=builder /opt/venv /opt/venv
USER app
EXPOSE 8000
# Liveness only: /ready also checks the databases and the external LLM, and restarting the container fixes neither.
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --start-interval=2s --retries=3 \
    CMD ["python", "-c", "import sys, urllib.request; sys.exit(urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2).status != 200)"]
# The analysis sandbox runs child Python processes inside this container: add --network none, --read-only (with a tmpfs for
# /tmp and /srv/state) and --memory in production so model-written code can't reach the network or exhaust the host.
CMD ["uvicorn", "data_analyst_agents.api:app", "--host", "0.0.0.0", "--port", "8000"]
