"""Per-job budgets: LLM calls, tokens, SQL executions and wall-clock time. A job stops with an explicit reason."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

from data_analyst_agents import config
from data_analyst_agents.config import JobBudget


class BudgetExceeded(Exception):
    def __init__(self, reason: str, detail: str):
        super().__init__(f"{reason}: {detail}")
        self.reason, self.detail = reason, detail


@dataclass
class BudgetTracker:
    budget: JobBudget
    clock: Callable[[], float] = time.monotonic
    llm_calls: int = 0
    replayed_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    llm_latency_s: float = 0.0
    sql_runs: int = 0
    sql_seconds: float = 0.0
    started: float = field(default=0.0)

    def __post_init__(self) -> None:
        self.started = self.clock()

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def elapsed(self) -> float:
        return self.clock() - self.started

    def remaining_s(self) -> float:
        return max(0.0, self.budget.max_wall_s - self.elapsed())

    def check_time(self) -> None:
        if self.elapsed() >= self.budget.max_wall_s:
            raise BudgetExceeded("max_wall_s", f"{self.elapsed():.0f} s elapsed (limit {self.budget.max_wall_s:.0f} s)")

    def before_llm(self) -> None:
        if self.llm_calls >= self.budget.max_llm_calls:
            raise BudgetExceeded("max_llm_calls", f"{self.llm_calls} LLM calls used (limit {self.budget.max_llm_calls})")
        if self.total_tokens >= self.budget.max_total_tokens:
            raise BudgetExceeded("max_total_tokens", f"{self.total_tokens} tokens used (limit {self.budget.max_total_tokens})")
        self.check_time()

    def after_llm(self, reply) -> None:
        self.llm_calls += 1
        self.replayed_calls += int(bool(reply.cached))
        self.prompt_tokens += reply.prompt_tokens
        self.completion_tokens += reply.completion_tokens
        self.llm_latency_s += reply.latency_s

    def before_sql(self) -> None:
        if self.sql_runs >= self.budget.max_sql_runs:
            raise BudgetExceeded("max_sql_runs", f"{self.sql_runs} SQL executions (limit {self.budget.max_sql_runs})")
        self.check_time()
        self.sql_runs += 1

    def usage(self, prices: dict = config.EXAMPLE_PRICES_PER_MILLION) -> dict:
        cost = (self.prompt_tokens * prices["input"] + self.completion_tokens * prices["output"]) / 1e6
        return {"llm_calls": self.llm_calls, "replayed_calls": self.replayed_calls, "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens, "total_tokens": self.total_tokens, "sql_runs": self.sql_runs,
                "llm_latency_s": round(self.llm_latency_s, 3), "sql_seconds": round(self.sql_seconds, 4),
                "service_time_s": round(self.llm_latency_s + self.sql_seconds, 3), "wall_s": round(self.elapsed(), 3),
                "example_cost_usd": round(cost, 6)}
