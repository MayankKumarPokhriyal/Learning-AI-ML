"""Paths, environment-driven settings, limits, budgets and example prices — one place to change behaviour."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# src/data_analyst_agents/config.py -> parents[2] is the project folder in a source checkout (Docker sets the env vars below)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVALS_DIR = Path(os.environ.get("ANALYST_EVALS_DIR", PROJECT_ROOT / "evals"))

# The three BIRD mini-dev databases this product answers questions about (business-style data: sales, banking, club finances)
DATABASES = ("debit_card_specializing", "financial", "student_club")
HIDDEN_TABLES = frozenset({"sqlite_sequence", "sqlite_stat1"})  # internal tables are never visible to the agents

# Personal data. Reads of these columns return NULL (SQLite authorizer SQLITE_IGNORE), so they can't be selected,
# filtered on, joined on, or leaked into prompts, charts, reports or exports.
PII_COLUMNS: dict[str, dict[str, frozenset[str]]] = {
    "student_club": {"member": frozenset({"email", "phone"})},
    "financial": {"client": frozenset({"birth_date"})},
}
DENIED_SQL_FUNCTIONS = frozenset({"load_extension", "readfile", "writefile", "edit", "fts3_tokenizer", "zipfile", "sqlar_compress"})

# EXAMPLE prices for cost arithmetic only: a local model has no per-token bill, and hosted prices change often.
EXAMPLE_PRICES_PER_MILLION = {"input": 0.15, "output": 0.60}


def data_dir() -> Path:
    """Everything created at runtime (git-ignored): the BIRD subset, job state, traces, charts, exports, eval reports."""
    return Path(os.environ.get("ANALYST_DATA_DIR", PROJECT_ROOT / "data"))


def bird_dir() -> Path:
    return Path(os.environ.get("ANALYST_BIRD_DIR", data_dir() / "bird_minidev"))


def questions_path() -> Path:
    return bird_dir() / "mini_dev_sqlite.json"


def db_path(db_id: str) -> Path:
    return bird_dir() / db_id / f"{db_id}.sqlite"


def description_dir(db_id: str) -> Path:
    return bird_dir() / db_id / "database_description"


def service_dir() -> Path:
    return data_dir() / "service"


def traces_path() -> Path:
    return Path(os.environ.get("ANALYST_TRACES_PATH", data_dir() / "traces" / "spans.jsonl"))


def artifacts_dir() -> Path:
    return data_dir() / "artifacts"


def exports_dir() -> Path:
    return data_dir() / "exports"


def sandbox_cache_dir() -> Path:
    """matplotlib's font cache for sandboxed runs (building it from scratch costs seconds per run)."""
    return data_dir() / "sandbox_cache"


def _flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class LLMSettings:
    base_url: str = "http://127.0.0.1:8080/v1"
    model: str = "gpt-oss-20b"
    api_key: str = "local"
    reasoning_effort: str | None = "low"  # gpt-oss chat-template option (3–4× faster on the course server); others ignore it
    temperature: float = 0.3  # low but not zero, so repeated trials (pass^k) measure real run-to-run variation
    timeout_s: float = 120.0  # one HTTP request
    max_attempts: int = 3  # our own retry loop (jittered exponential backoff, transient errors only)
    max_concurrency: int = 2  # requests in flight: the shared server has few slots
    cache_path: Path | None = None  # JSONL(.gz) record of replies keyed by the exact request
    replay_only: bool = False  # never call the server: a request without a recorded reply raises ReplayMiss

    @classmethod
    def from_env(cls) -> LLMSettings:
        cache = os.environ.get("LLM_CACHE_PATH")
        return cls(
            base_url=os.environ.get("LLM_BASE_URL", cls.base_url),
            model=os.environ.get("LLM_MODEL", cls.model),
            api_key=os.environ.get("LLM_API_KEY", cls.api_key),
            reasoning_effort=os.environ.get("LLM_REASONING_EFFORT", "low") or None,
            temperature=float(os.environ.get("LLM_TEMPERATURE", cls.temperature)),
            timeout_s=float(os.environ.get("LLM_TIMEOUT_S", cls.timeout_s)),  # raise it on a busy shared server: requests wait for a free slot
            max_concurrency=int(os.environ.get("LLM_MAX_CONCURRENCY", cls.max_concurrency)),
            cache_path=Path(cache) if cache else None,
            replay_only=_flag("LLM_REPLAY_ONLY"),
        )


@dataclass(frozen=True)
class QueryLimits:
    max_rows: int = 10_000  # rows fetched per query; more sets `truncated`
    timeout_s: float = 5.0  # per query, enforced inside SQLite by a progress handler
    preview_rows: int = 20  # rows an LLM may see
    expensive_scan_rows: int = 500_000  # full-table-scan rows (EXPLAIN QUERY PLAN) above which a human must approve


@dataclass(frozen=True)
class JobBudget:
    max_llm_calls: int = 10
    max_total_tokens: int = 40_000
    max_wall_s: float = 300.0
    max_sql_runs: int = 40  # answer attempts + the verifier's re-runs, COUNT(*) cross-checks and filter-value probes

    @classmethod
    def from_env(cls) -> JobBudget:
        return cls(max_llm_calls=int(os.environ.get("ANALYST_MAX_LLM_CALLS", cls.max_llm_calls)),
                   max_total_tokens=int(os.environ.get("ANALYST_MAX_TOTAL_TOKENS", cls.max_total_tokens)),
                   max_wall_s=float(os.environ.get("ANALYST_MAX_WALL_S", cls.max_wall_s)))
