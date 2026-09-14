# syntax=docker/dockerfile:1
# Train first (artifacts/ must exist), then:
#   docker build -t churn-service:1.0.0 .
#   docker run --rm -p 8000:8000 churn-service:1.0.0

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

# ---------- stage 2: runtime — Python, the venv, and the model artifacts only ----------
FROM python:3.12-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    CHURN_ARTIFACTS_DIR=/srv/artifacts
RUN groupadd --system --gid 10001 app && useradd --system --uid 10001 --gid app --no-create-home app
WORKDIR /srv
COPY --from=builder /opt/venv /opt/venv
COPY artifacts/model.joblib artifacts/metadata.json artifacts/reference_stats.json ./artifacts/
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --start-interval=2s --retries=3 \
    CMD ["python", "-c", "import sys, urllib.request; sys.exit(urllib.request.urlopen('http://127.0.0.1:8000/ready', timeout=2).status != 200)"]
CMD ["uvicorn", "churn_service.api:app", "--host", "0.0.0.0", "--port", "8000"]
