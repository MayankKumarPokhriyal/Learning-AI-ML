"""Paths, the pinned corpus, and runtime settings. Every setting can be overridden with an environment variable."""

from __future__ import annotations

import os
from dataclasses import dataclass, fields, replace
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]  # the rag_assistant/ project folder (in a source checkout)

# ── Corpus: uv's own documentation, pinned to one release tag so answers and gold labels stay reproducible ──
UV_REPO = "astral-sh/uv"
UV_DOCS_TAG = "0.10.3"
UV_DOCS_LICENSE = "MIT OR Apache-2.0"
RAW_URL = "https://raw.githubusercontent.com/{repo}/{tag}/docs/{page}"
SOURCE_URL = "https://github.com/{repo}/blob/{tag}/docs/{page}"
DOCS_PAGES: tuple[str, ...] = (
    "getting-started/installation.md", "getting-started/features.md", "getting-started/help.md",
    "concepts/projects/init.md", "concepts/projects/layout.md", "concepts/projects/dependencies.md", "concepts/projects/run.md",
    "concepts/projects/sync.md", "concepts/projects/config.md", "concepts/projects/build.md", "concepts/projects/export.md",
    "concepts/projects/workspaces.md", "concepts/tools.md", "concepts/python-versions.md", "concepts/configuration-files.md",
    "concepts/indexes.md", "concepts/cache.md", "concepts/resolution.md", "concepts/build-backend.md", "concepts/preview.md",
    "concepts/authentication/http.md", "concepts/authentication/git.md", "concepts/authentication/certificates.md",
    "guides/install-python.md", "guides/scripts.md", "guides/tools.md", "guides/projects.md", "guides/package.md",
    "guides/integration/docker.md", "guides/integration/github.md", "guides/integration/gitlab.md", "guides/integration/fastapi.md",
    "guides/integration/pytorch.md", "guides/integration/jupyter.md", "guides/integration/pre-commit.md",
    "guides/integration/dependency-bots.md", "guides/integration/alternative-indexes.md",
    "pip/environments.md", "pip/packages.md", "pip/dependencies.md", "pip/compile.md",
    "reference/storage.md", "reference/installer.md", "reference/policies/versioning.md", "reference/policies/platforms.md",
    "reference/policies/license.md", "reference/troubleshooting/reproducible-examples.md",
)

# Domains an answer may link to: the domains that appear in the text of the pinned docs (subdomains included).
# A URL to any other domain in an answer is treated as a possible injection and blocked.
DEFAULT_ALLOWED_DOMAINS: tuple[str, ...] = (
    "astral.sh", "github.com", "githubusercontent.com", "pypi.org", "pythonhosted.org", "python.org", "pytorch.org",
    "dev.azure.com", "renovatebot.com", "slsa.dev", "apache.org", "opensource.org", "example.com", "private-index.com",
    "corp.dev", "localhost", "127.0.0.1",
)

ENV_VARS = {
    "data_dir": "RAG_DATA_DIR", "corpus_tag": "RAG_CORPUS_TAG", "corpus_dir": "RAG_CORPUS_DIR", "index_dir": "RAG_INDEX_DIR",
    "trace_path": "RAG_TRACE_PATH", "embedding_model": "RAG_EMBEDDING_MODEL", "reranker_model": "RAG_RERANKER_MODEL",
    "device": "RAG_DEVICE", "chunk_tokens": "RAG_CHUNK_TOKENS", "chunk_overlap": "RAG_CHUNK_OVERLAP",
    "retrieval_mode": "RAG_RETRIEVAL_MODE", "k_dense": "RAG_K_DENSE", "k_bm25": "RAG_K_BM25", "rerank": "RAG_RERANK",
    "rerank_depth": "RAG_RERANK_DEPTH", "top_k": "RAG_TOP_K", "context_budget": "RAG_CONTEXT_BUDGET", "llm_base_url": "LLM_BASE_URL",
    "llm_model": "LLM_MODEL", "llm_api_key": "LLM_API_KEY", "reasoning_effort": "LLM_REASONING_EFFORT", "max_output_tokens": "LLM_MAX_TOKENS",
    "llm_cache_dir": "LLM_CACHE_DIR", "llm_replay_realtime": "LLM_REPLAY_REALTIME",
    "max_question_chars": "RAG_MAX_QUESTION_CHARS", "max_question_tokens": "RAG_MAX_QUESTION_TOKENS",
    "prompt_defense": "RAG_PROMPT_DEFENSE", "quarantine_flagged": "RAG_QUARANTINE_FLAGGED", "output_url_check": "RAG_OUTPUT_URL_CHECK",
    "allowed_answer_domains": "RAG_ALLOWED_ANSWER_DOMAINS", "cache_mode": "RAG_CACHE_MODE", "cache_threshold": "RAG_CACHE_THRESHOLD",
    "cache_ttl_seconds": "RAG_CACHE_TTL_SECONDS", "ingest_api_key": "RAG_INGEST_API_KEY",
}


@dataclass(frozen=True)
class Settings:
    # paths
    data_dir: Path = PROJECT_DIR / "data"
    corpus_tag: str = UV_DOCS_TAG
    corpus_dir: Path | None = None  # default: <data_dir>/corpus/<corpus_tag>
    index_dir: Path | None = None  # default: <data_dir>/index
    trace_path: Path | None = None  # default: <data_dir>/traces/requests.jsonl
    # models (small enough for a CPU)
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    query_prompt: str = "Represent this sentence for searching relevant passages: "  # bge's instruction for short queries
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    device: str = "cpu"
    # chunking
    chunk_tokens: int = 300
    chunk_overlap: int = 50
    # retrieval
    retrieval_mode: str = "hybrid"  # dense | bm25 | hybrid
    k_dense: int = 30
    k_bm25: int = 30
    rrf_k: int = 60
    rerank: bool = True
    rerank_depth: int = 20
    top_k: int = 5
    context_budget: int = 1500  # LLM tokens of retrieved text per question
    # generation
    llm_base_url: str = "http://127.0.0.1:8080/v1"
    llm_model: str = "gpt-oss-20b"
    llm_api_key: str = "local"
    reasoning_effort: str = "low"  # sent as chat_template_kwargs (gpt-oss on llama.cpp); "none" disables it
    max_output_tokens: int = 1024
    temperature: float = 0.0
    llm_timeout: float = 180.0
    llm_cache_dir: Path | None = None  # development only: record LLM responses and replay identical requests
    llm_replay_realtime: bool = False  # when replaying, wait as long as the recorded call took
    # guardrails
    max_question_chars: int = 1000
    max_question_tokens: int = 250
    prompt_defense: bool = True
    quarantine_flagged: bool = True
    output_url_check: bool = True
    allowed_answer_domains: tuple[str, ...] = DEFAULT_ALLOWED_DOMAINS
    # response cache
    cache_mode: str = "semantic"  # off | exact | semantic
    cache_threshold: float = 0.95
    cache_max_entries: int = 1000
    cache_ttl_seconds: float = 3600.0
    # service
    ingest_api_key: str | None = None  # POST /ingest is disabled until this is set

    def __post_init__(self):
        object.__setattr__(self, "data_dir", Path(self.data_dir))
        defaults = {"corpus_dir": self.data_dir / "corpus" / self.corpus_tag, "index_dir": self.data_dir / "index",
                    "trace_path": self.data_dir / "traces" / "requests.jsonl"}
        for name, default in defaults.items():
            value = getattr(self, name)
            object.__setattr__(self, name, Path(value) if value is not None else default)
        if self.llm_cache_dir is not None:
            object.__setattr__(self, "llm_cache_dir", Path(self.llm_cache_dir))
        if self.retrieval_mode not in {"dense", "bm25", "hybrid"}:
            raise ValueError(f"retrieval_mode must be dense, bm25 or hybrid, not {self.retrieval_mode!r}")
        if self.cache_mode not in {"off", "exact", "semantic"}:
            raise ValueError(f"cache_mode must be off, exact or semantic, not {self.cache_mode!r}")
        if not 0 <= self.chunk_overlap < self.chunk_tokens:
            raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_tokens")
        if self.top_k < 1 or self.context_budget < 100:
            raise ValueError("top_k must be >= 1 and context_budget >= 100")

    @classmethod
    def from_env(cls, **overrides) -> Settings:
        """Defaults ← environment variables ← explicit overrides (highest priority)."""
        types = {f.name: str(f.type) for f in fields(cls)}
        values = {}
        for name, var in ENV_VARS.items():
            raw = os.environ.get(var)
            if raw is not None and raw.strip() != "":
                values[name] = _cast(types[name], raw)
        values.update(overrides)
        return cls(**values)

    def replace(self, **changes) -> Settings:
        return replace(self, **changes)


def _cast(type_name: str, raw: str):
    if type_name == "bool":
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    if type_name == "int":
        return int(raw)
    if type_name == "float":
        return float(raw)
    if type_name.startswith("Path"):
        return Path(raw)
    if type_name.startswith("tuple"):
        return tuple(part.strip() for part in raw.split(",") if part.strip())
    return raw
