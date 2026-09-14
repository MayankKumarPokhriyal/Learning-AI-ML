# syntax=docker/dockerfile:1
# CPU-only PyTorch serving image. Export a model first (artifacts/model.pt2 + metadata.json), then:
#   docker build -t image-classifier:1.0.0 .
#   docker run --rm -p 8000:8000 image-classifier:1.0.0

# ---------- stage 1: builder ----------
FROM python:3.12-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.10.3 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /build
COPY requirements.lock ./
# The CPU wheel index keeps CUDA libraries (several GB) out of the image
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv /opt/venv && \
    uv pip install --python /opt/venv/bin/python -r requirements.lock \
        --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/venv/bin/python --no-deps .

# ---------- stage 2: runtime ----------
FROM python:3.12-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    IMAGE_CLASSIFIER_ARTIFACTS_DIR=/srv/artifacts \
    OMP_NUM_THREADS=2
RUN groupadd --system --gid 10001 app && useradd --system --uid 10001 --gid app --home-dir /tmp app
WORKDIR /srv
COPY --from=builder /opt/venv /opt/venv
COPY artifacts/model.pt2 artifacts/metadata.json ./artifacts/
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=30s --start-interval=2s --retries=3 \
    CMD ["python", "-c", "import sys, urllib.request; sys.exit(urllib.request.urlopen('http://127.0.0.1:8000/ready', timeout=2).status != 200)"]
CMD ["uvicorn", "image_classifier.api:app", "--host", "0.0.0.0", "--port", "8000"]
