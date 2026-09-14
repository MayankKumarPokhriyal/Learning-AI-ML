"""Pydantic contracts of the HTTP API."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, StringConstraints

ConversationId = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_\-]{8,64}$")]


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)]
    conversation_id: ConversationId | None = None


class ApprovalDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: Literal["approve", "deny"]
    reviewer: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=80)]


class FinalAnswer(BaseModel):
    answer: str
    citations: list[str]
    outcome: Literal["answered", "not_found", "awaiting_approval", "refused"]
    approval_ids: list[str] = []


class Usage(BaseModel):
    steps: int
    tool_calls: int
    tool_errors: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_s: float
    wall_clock_s: float
    example_cost_usd: float


class ChatResponse(BaseModel):
    run_id: str
    conversation_id: str
    status: Literal["completed", "stopped", "failed"]
    stop_reason: str
    answer: FinalAnswer | None
    usage: Usage
    approvals: list[dict[str, Any]]
    guardrail_events: list[dict[str, Any]]
    error: str | None = None
