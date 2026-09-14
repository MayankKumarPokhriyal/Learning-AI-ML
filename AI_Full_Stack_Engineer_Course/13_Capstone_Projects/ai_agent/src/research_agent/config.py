"""Paths, environment-driven settings, budgets, tool policy, and example prices — one place to change behaviour."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# src/research_agent/config.py -> parents[2] is the project folder in a source checkout.
# Installed packages (e.g. in Docker) set AGENT_DATA_DIR instead.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

ARXIV_API_URL = "https://export.arxiv.org/api/query"
ARXIV_MIN_INTERVAL_S = 3.0  # arXiv API terms of use: at most one request every three seconds, one connection at a time
USER_AGENT = "research-agent-capstone/1.0 (educational project; 1 request per 3 s; responses cached)"

TOOL_ALLOWLIST = ("search_papers", "get_paper", "save_report")  # the only MCP tools the model is ever shown
APPROVAL_REQUIRED = frozenset({"save_report"})  # tools with side effects wait for a human decision
ALLOWED_ARCHIVES = frozenset({
    "cs", "stat", "math", "eess", "econ", "q-bio", "q-fin", "physics", "astro-ph", "cond-mat", "gr-qc", "hep-ex",
    "hep-lat", "hep-ph", "hep-th", "math-ph", "nlin", "nucl-ex", "nucl-th", "quant-ph",
})
MAX_ABSTRACT_CHARS = 1500  # keeps tool results small: the LLM server has an 8,192-token context per request
MAX_REPORT_CHARS = 4000
TRUSTED_LINK_DOMAINS = ("arxiv.org", "export.arxiv.org", "doi.org")  # links allowed in answers and reports

# EXAMPLE prices for cost arithmetic only. A local model has no per-token bill, and hosted prices change often.
EXAMPLE_PRICES_PER_MILLION = {"input": 0.15, "output": 0.60}


def data_dir() -> Path:
    """Everything the project creates at runtime (git-ignored): snapshots, corpus, state, traces, reports."""
    return Path(os.environ.get("AGENT_DATA_DIR", PROJECT_ROOT / "data"))


def snapshot_dir() -> Path:
    return data_dir() / "arxiv_snapshots"


def corpus_path() -> Path:
    return Path(os.environ.get("AGENT_CORPUS_PATH", data_dir() / "papers.sqlite"))


def state_db_path() -> Path:
    return data_dir() / "agent_state.sqlite"


def traces_dir() -> Path:
    return data_dir() / "traces"


def reports_dir() -> Path:
    return Path(os.environ.get("AGENT_REPORTS_DIR", data_dir() / "reports"))


def arxiv_mode() -> str:
    """replay = snapshots only (offline, reproducible) · record = snapshot if present, else fetch and save · live = always fetch."""
    return os.environ.get("ARXIV_MODE", "record")


@dataclass(frozen=True)
class LLMSettings:
    base_url: str = "http://127.0.0.1:8080/v1"
    model: str = "gpt-oss-20b"
    api_key: str = "local"
    reasoning_effort: str | None = "low"  # gpt-oss chat-template option (faster steps); other servers ignore it
    temperature: float = 0.0
    timeout_s: float = 180.0
    max_retries: int = 2
    max_concurrency: int = 2  # requests in flight — shared local servers have few slots
    cache_path: Path | None = None  # JSONL reply cache for reproducible re-runs (leave unset in production)

    @classmethod
    def from_env(cls) -> LLMSettings:
        cache = os.environ.get("LLM_CACHE_PATH")
        return cls(
            base_url=os.environ.get("LLM_BASE_URL", cls.base_url),
            model=os.environ.get("LLM_MODEL", cls.model),
            api_key=os.environ.get("LLM_API_KEY", cls.api_key),
            reasoning_effort=os.environ.get("LLM_REASONING_EFFORT", "low") or None,
            max_concurrency=int(os.environ.get("LLM_MAX_CONCURRENCY", cls.max_concurrency)),
            cache_path=Path(cache) if cache else None,
        )


@dataclass(frozen=True)
class Budget:
    """Hard limits for one agent run. The loop stops with an explicit reason instead of running away."""

    max_steps: int = 6  # LLM calls that may request tools
    max_output_tokens_per_step: int = 1024  # max_tokens per call (gpt-oss reasoning tokens count here too)
    max_prompt_tokens_per_step: int = 6000  # estimated before sending; the server's context is 8,192 tokens
    max_total_tokens: int = 40_000  # prompt + completion over the whole run (what you pay for)
    max_step_latency_s: float = 120.0  # one LLM call
    max_total_latency_s: float = 420.0  # the whole run, wall clock

    @classmethod
    def from_env(cls) -> Budget:
        return cls(max_steps=int(os.environ.get("AGENT_MAX_STEPS", cls.max_steps)),
                   max_total_tokens=int(os.environ.get("AGENT_MAX_TOTAL_TOKENS", cls.max_total_tokens)))
