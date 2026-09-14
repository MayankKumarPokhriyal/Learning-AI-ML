"""Budget enforcement for one agent run: steps, per-step and total tokens, per-step and total latency."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from research_agent.config import Budget

CHARS_PER_TOKEN = 3.2  # conservative for English + JSON; use the server's /tokenize endpoint when you need exact counts


class BudgetExceeded(Exception):
    def __init__(self, reason: str, detail: str):
        super().__init__(f"{reason}: {detail}")
        self.reason, self.detail = reason, detail


def estimate_tokens(messages: list[dict], tools: list[dict] | None = None) -> int:
    """Rough prompt size from the JSON length — cheap, provider-independent, and deliberately on the high side."""
    chars = len(json.dumps(messages, ensure_ascii=False)) + len(json.dumps(tools or [], ensure_ascii=False))
    return int(chars / CHARS_PER_TOKEN)


@dataclass
class BudgetTracker:
    budget: Budget
    clock: Callable[[], float] = time.perf_counter
    steps: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    started: float = field(default=0.0)

    def __post_init__(self) -> None:
        self.started = self.clock()

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def elapsed(self) -> float:
        return self.clock() - self.started

    def before_step(self, messages: list[dict], tools: list[dict] | None = None) -> int:
        """Raise BudgetExceeded if another tool-calling LLM step is not allowed; returns the prompt-token estimate."""
        b = self.budget
        if self.steps >= b.max_steps:
            raise BudgetExceeded("max_steps", f"{self.steps} steps used (limit {b.max_steps})")
        self._check_totals()
        estimate = estimate_tokens(messages, tools)
        if estimate > b.max_prompt_tokens_per_step:
            raise BudgetExceeded("max_prompt_tokens_per_step", f"next prompt ≈{estimate} tokens (limit {b.max_prompt_tokens_per_step})")
        return estimate

    def before_extra_call(self) -> None:
        """For calls outside the step count (the final structuring call): totals only."""
        self._check_totals()

    def _check_totals(self) -> None:
        b = self.budget
        if self.total_tokens >= b.max_total_tokens:
            raise BudgetExceeded("max_total_tokens", f"{self.total_tokens} tokens used (limit {b.max_total_tokens})")
        if self.elapsed() >= b.max_total_latency_s:
            raise BudgetExceeded("max_total_latency", f"{self.elapsed():.1f} s elapsed (limit {b.max_total_latency_s:.0f} s)")

    def step_timeout(self) -> float:
        """Seconds the next LLM call may take: the per-step limit, capped by what is left of the total."""
        return max(0.001, min(self.budget.max_step_latency_s, self.budget.max_total_latency_s - self.elapsed()))

    def record(self, prompt_tokens: int, completion_tokens: int, *, counts_as_step: bool = True) -> None:
        self.steps += int(counts_as_step)
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
