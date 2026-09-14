# rag-assistant — question answering over the uv docs, with verified citations

A complete, small, production-style retrieval-augmented generation (RAG) service. It answers questions about
[uv](https://docs.astral.sh/uv/) (Astral's Python package and project manager) using only uv's own documentation,
cites the exact passages it used, checks that every quoted citation really exists in the retrieved text, refuses when the
docs don't contain the answer, and is measured by an evaluation harness with a regression gate for CI.

It is the project folder of the course notebook [`03_LLM_RAG_Assistant_Project.ipynb`](../03_LLM_RAG_Assistant_Project.ipynb),
which walks through every part and runs all of this code against a real LLM.

## What it does

| Part | Where |
|---|---|
| Download 47 pages of the uv docs pinned to release `0.10.3`, recording URL + SHA-256 per page | `src/rag_assistant/corpus.py` |
| Clean MkDocs Markdown (front matter, admonitions, content tabs, links) while keeping headings and code | `parsing.py` |
| Structure-aware chunking: one chunk per section, long sections split at paragraph boundaries with token overlap | `chunking.py` |
| Persisted, versioned index with **incremental** ingestion (only documents whose content hash changed are re-embedded) and an atomic version switch | `index.py` |
| Hybrid retrieval: BM25 + dense vectors fused with reciprocal rank fusion, cross-encoder reranking, token-budgeted context | `retrieval.py` |
| Grounded generation with a JSON schema: answer + citations (chunk id + verbatim quote) + answerable flag; citation validation; refusal policy | `generation.py` |
| Guardrails: input limits, prompt-injection detection and quarantine at ingestion, spotlighted prompts, link allowlist (also while streaming), redaction of secrets/PII in logs | `guardrails.py` |
| Exact or semantic response cache scoped to the index version and configuration | `cache.py` |
| One pipeline for API, CLI and evaluation; JSONL traces with retrieval ids/scores, per-stage latency and tokens | `pipeline.py`, `tracing.py` |
| FastAPI service: `/ask` (JSON or Server-Sent Events), `/ingest` (API key), `/health`, `/ready`, `/stats` | `api.py` |
| Evaluation harness: recall@k, MRR, correctness, faithfulness (LLM judge with agreement checks), citation accuracy, refusals, latency, tokens; configuration comparison; regression gate | `evaluation.py`, `eval/` |
| OpenAI-compatible LLM client with optional record/replay for development, plus test doubles | `llm.py`, `embeddings.py` |
| Unit, API and live smoke tests | `tests/` |
| Multi-stage, non-root Docker image with models baked in and a health check | `Dockerfile`, `.dockerignore` |
| Example GitHub Actions pipeline (lint → tests → retrieval gate → image smoke test; weekly answer gate) | `ci/github-actions.yml` |

## Quickstart

```bash
# 1. Environment (uv: https://docs.astral.sh/uv/) — locked runtime versions + dev tools
uv venv .venv && source .venv/bin/activate
uv pip install -e ".[dev]" -c requirements.lock        # the lock targets Linux x86_64 with CPU PyTorch; on macOS drop "-c requirements.lock"

# 2. Data: download the pinned docs (~370 kB) and build the index (downloads two small models, ~220 MB, once)
python -m rag_assistant download
python -m rag_assistant ingest                          # run it again: nothing is re-embedded

# 3. Tests (unit + API with test doubles; the live test runs when an LLM server is reachable)
pytest

# 4. An LLM: any OpenAI-compatible server, e.g. llama.cpp with gpt-oss-20b
brew install llama.cpp && llama-server -hf ggml-org/gpt-oss-20b-GGUF --port 8080 --jinja
#    (or point LLM_BASE_URL / LLM_MODEL / LLM_API_KEY at Ollama, LM Studio, vLLM or a hosted provider)

# 5. Ask and serve
python -m rag_assistant ask "How do I clear the cache for one package?"
RAG_INGEST_API_KEY=change-me uvicorn rag_assistant.api:app --reload
curl -s localhost:8000/ask -H "content-type: application/json" -d '{"question": "What is uvx?"}'
curl -N localhost:8000/ask -H "content-type: application/json" -d '{"question": "What is uvx?", "stream": true}'
curl -s -X POST localhost:8000/ingest -H "X-API-Key: change-me"

# 6. Evaluate and gate
python -m rag_assistant eval --judge --name local
python -m rag_assistant gate reports/eval_local.json    # exit code 1 = regression

# 7. Container (after step 2)
docker build -t rag-assistant:1.0.0 .
docker run --rm -p 8000:8000 -e LLM_BASE_URL=http://host.docker.internal:8080/v1 rag-assistant:1.0.0
```

`data/` (corpus, index, traces) and `reports/` are git-ignored: steps 2 and 6 recreate them.

## API

| Method | Path | Purpose |
|---|---|---|
| POST | `/ask` | `{"question": "...", "stream": false, "top_k": 5}` → `status` (`answered`, `refused`, `unsupported`, `blocked_output`, `invalid_output`), `answer`, `citations` (chunk id, quote, source URL, section), `cache`, `timings_ms`, `tokens`, `index_version` |
| POST | `/ask` with `"stream": true` | `text/event-stream`: `meta` (sources), `delta` (answer text as it is generated), then `final` (the validated answer — it replaces the streamed text) or `error` |
| POST | `/ingest` | header `X-API-Key`: incremental re-index of the corpus folder, then an atomic switch to the new index (401 bad key, 409 already running, 503 when no key is configured) |
| GET | `/health` | liveness — the process is up |
| GET | `/ready` | readiness — index and models loaded (503 otherwise); also reports whether the LLM is reachable |
| GET | `/stats` | cache hit rate and request counts by status |

Requests longer than `RAG_MAX_QUESTION_CHARS` (1,000) or `RAG_MAX_QUESTION_TOKENS` (250) get **413**; malformed bodies get **422**; an unreachable LLM gives **503** (or an `error` event).

## Configuration

Everything is an environment variable (see `src/rag_assistant/config.py`). The most useful ones:

| Variable | Default | Meaning |
|---|---|---|
| `LLM_BASE_URL`, `LLM_MODEL`, `LLM_API_KEY` | `http://127.0.0.1:8080/v1`, `gpt-oss-20b`, `local` | any OpenAI-compatible endpoint |
| `LLM_REASONING_EFFORT` | `low` | sent as `chat_template_kwargs` (gpt-oss); set `none` for other models |
| `RAG_RETRIEVAL_MODE`, `RAG_RERANK`, `RAG_TOP_K`, `RAG_CONTEXT_BUDGET` | `hybrid`, `true`, `5`, `1500` | retrieval configuration |
| `RAG_CACHE_MODE`, `RAG_CACHE_THRESHOLD` | `semantic`, `0.95` | `off` / `exact` / `semantic` response cache |
| `RAG_PROMPT_DEFENSE`, `RAG_QUARANTINE_FLAGGED`, `RAG_OUTPUT_URL_CHECK` | `true` | injection defenses (switch off only for experiments) |
| `RAG_INGEST_API_KEY` | unset | enables `POST /ingest` |
| `RAG_DEVICE` | `cpu` | `mps` or `cuda` for the embedding model and reranker |
| `RAG_DATA_DIR` | `./data` | corpus, index and traces |
| `LLM_CACHE_DIR` | unset | development only: record LLM responses and replay identical requests |

## Evaluation and the regression gate

`eval/questions.jsonl` holds 36 hand-written questions: 28 answerable (gold answer, gold evidence quoted from the parsed docs,
and regex rules) and 8 unanswerable (their key terms are verified absent from the corpus). Evidence is stored as text spans, not
chunk ids, so the labels survive re-chunking.

`eval/gate.json` holds the success criteria, fixed before the first run. `python -m rag_assistant gate` compares a report with
them and exits with code 1 on any miss. CI runs the deterministic retrieval gate on every change and the full answer gate
(which needs an LLM endpoint) weekly or by hand.

## Monitoring plan

- **Traces** (`data/traces/requests.jsonl`, redacted): alert on p95 latency per stage, error and `unsupported`/`blocked_output` rates, and token spend.
- **Quality**: sample production questions into the labeled set every month; re-run the answer gate on every model, prompt, index or configuration change.
- **Freshness**: when uv releases, download the new tag and run `ingest`: only changed pages are re-embedded and the service switches index atomically.
- **Security**: review chunks flagged by the injection detector (`manifest.json → flagged_chunks`) before trusting a new corpus.

## Data and license

Corpus: the uv documentation (`docs/` folder of [astral-sh/uv](https://github.com/astral-sh/uv)) at tag `0.10.3`, dual-licensed
**MIT OR Apache-2.0**. Each chunk keeps the pinned source URL, so every citation links to the exact text that was indexed.
Models: [`BAAI/bge-small-en-v1.5`](https://huggingface.co/BAAI/bge-small-en-v1.5) (MIT) and
[`cross-encoder/ms-marco-MiniLM-L6-v2`](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2) (Apache-2.0).
