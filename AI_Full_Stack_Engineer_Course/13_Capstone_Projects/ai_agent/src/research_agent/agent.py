"""The agent loop.

For each user message: load compact history from SQLite → loop {budget check → LLM step → validate + run tool calls
(errors fed back, side effects queued for human approval, untrusted output spotlighted)} → one schema-constrained call
turns the final text into {answer, citations, outcome} → citation grounding + output filter → persist + trace.
"""

from __future__ import annotations

import asyncio
import copy
import inspect
import json
import os
import re
import secrets as secrets_lib
import sys
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

import jsonschema

from research_agent import approvals as approval_tokens
from research_agent import config
from research_agent.budget import BudgetExceeded, BudgetTracker
from research_agent.config import Budget
from research_agent.guardrails import (
    ARXIV_ID_IN_TEXT,
    SPOTLIGHT_RULE,
    filter_output,
    ground_citations,
    injection_flags,
    scrub_ungrounded_ids,
    side_effect_violations,
    spotlight,
)
from research_agent.llm import ChatModel, OpenAICompatibleLLM
from research_agent.store import StateStore
from research_agent.tools import ToolInputError, validate_report_arguments
from research_agent.tracing import JsonlTracer

SYSTEM_PROMPT = (
    "You are a research assistant. You answer questions about scientific papers using tools over a local snapshot of arXiv.\n"
    "Rules:\n"
    "- Look facts up with the tools. Never invent titles, arXiv identifiers, authors, dates, or numbers.\n"
    "- search_papers finds candidate papers; get_paper returns the full record and abstract of one arXiv id.\n"
    "- Mention the arXiv identifiers (like 2210.03629) of the papers your answer relies on.\n"
    "- If the tools find nothing relevant, say that you could not find it.\n"
    "- When the user explicitly asks you to save or write a report, call save_report yourself with the finished content. "
    "The system then holds it for a human reviewer automatically, so do not ask the user for permission first; afterwards, "
    "tell the user the report is waiting for approval. Never call save_report if the user did not ask for a report.\n"
    "- Answer in at most four sentences."
)
FINAL_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string", "description": "The answer for the user, at most four sentences."},
        "citations": {"type": "array", "items": {"type": "string"}, "maxItems": 8, "description": "arXiv ids the answer relies on."},
        "outcome": {"type": "string", "enum": ["answered", "not_found", "awaiting_approval", "refused"]},
    },
    "required": ["answer", "citations", "outcome"],
    "additionalProperties": False,
}
UNTRUSTED_OUTPUT_TOOLS = frozenset({"search_papers", "get_paper"})  # their text was written by third parties
SIDE_EFFECT_INTENT = re.compile(r"\b(save|write|store|export|create|make|put)\b.{0,80}\b(report|reading list|summary|notes?|file)\b",
                                re.IGNORECASE | re.DOTALL)
EventCallback = Callable[[dict], Awaitable[None] | None]


class ToolBackend(Protocol):
    async def list_tools(self) -> list[dict]: ...

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> tuple[bool, Any]: ...


class MCPToolBackend:
    """Adapter from a connected `mcp.Client` (stdio subprocess or in-memory server) to the agent."""

    def __init__(self, client):
        self.client = client

    async def list_tools(self) -> list[dict]:
        return [{"name": t.name, "description": t.description or "", "input_schema": t.input_schema,
                 "read_only": bool(t.annotations and t.annotations.read_only_hint)} for t in (await self.client.list_tools()).tools]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> tuple[bool, Any]:
        result = await self.client.call_tool(name, arguments)
        text = "\n".join(block.text for block in result.content if getattr(block, "type", "") == "text")
        if result.is_error:
            return True, text
        return False, result.structured_content if result.structured_content is not None else text


def to_openai_tool(tool: dict) -> dict:
    """MCP tool -> OpenAI function tool. Host-only approval arguments are removed from what the model sees."""
    schema = copy.deepcopy(tool["input_schema"])
    for hidden in approval_tokens.HIDDEN_ARGUMENTS:
        schema.get("properties", {}).pop(hidden, None)
        if hidden in schema.get("required", []):
            schema["required"].remove(hidden)
    for prop in schema.get("properties", {}).values():
        prop.pop("title", None)
    schema.pop("title", None)
    return {"type": "function", "function": {"name": tool["name"], "description": tool["description"], "parameters": schema}}


@dataclass
class GuardrailSettings:
    spotlight: bool = True  # wrap third-party text in untrusted markers + a system rule
    taint_policy: bool = True  # side effects only when the user asked, and without untrusted links after untrusted input
    output_filter: bool = True  # strip images / untrusted links / known secrets from answers
    ground_citations: bool = True  # only cite ids that a tool returned in this run
    require_approval: bool = True  # side effects wait for a human

    @classmethod
    def disabled(cls) -> GuardrailSettings:
        """Everything off — ONLY for measuring what the guardrails prevent (the attack test)."""
        return cls(spotlight=False, taint_policy=False, output_filter=False, ground_citations=False, require_approval=False)


@dataclass
class RunResult:
    run_id: str
    conversation_id: str
    question: str
    status: str = "running"  # completed | stopped | failed
    stop_reason: str = ""  # final_answer | max_steps | max_total_tokens | max_step_latency | ... | llm_error
    answer: dict | None = None  # what the user sees (after guardrails)
    raw_answer: dict | None = None  # structured answer before output guardrails (for measuring attacks)
    draft: str | None = None  # the model's final text before structuring
    steps: int = 0
    tool_calls: int = 0
    tool_errors: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0  # sum of measured LLM + tool latencies (a cached reply keeps the latency measured when it was made)
    wall_clock_s: float = 0.0  # elapsed time of this run as it happened
    tools_used: list[str] = field(default_factory=list)
    approvals: list[dict] = field(default_factory=list)
    executed_side_effects: list[dict] = field(default_factory=list)
    guardrail_events: list[dict] = field(default_factory=list)
    error: str | None = None
    trace_path: str | None = None

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def cost_usd(self, prices: dict = config.EXAMPLE_PRICES_PER_MILLION) -> float:
        return (self.prompt_tokens * prices["input"] + self.completion_tokens * prices["output"]) / 1e6

    def usage(self) -> dict:
        return {"steps": self.steps, "tool_calls": self.tool_calls, "tool_errors": self.tool_errors, "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens, "total_tokens": self.total_tokens, "latency_s": self.latency_s, "wall_clock_s": self.wall_clock_s,
                "example_cost_usd": round(self.cost_usd(), 6)}

    def to_dict(self) -> dict:
        record = asdict(self)
        record["usage"] = self.usage()
        return record


@dataclass
class _RunState:
    user_requested_side_effect: bool
    tainted: bool = False
    seen_ids: set[str] = field(default_factory=set)  # ids returned by tools in this run: the only citable ones
    question_ids: set[str] = field(default_factory=set)  # ids the user typed: may be repeated in the text, not cited


class StructuredOutputError(RuntimeError):
    pass


def _ids_in(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        if isinstance(value.get("arxiv_id"), str):
            found.add(value["arxiv_id"])
        for item in value.values():
            found |= _ids_in(item)
    elif isinstance(value, list):
        for item in value:
            found |= _ids_in(item)
    return found


class ResearchAgent:
    def __init__(self, llm: ChatModel, tools: ToolBackend, store: StateStore, *, budget: Budget | None = None,
                 guardrails: GuardrailSettings | None = None, traces_dir: str | Path | None = None, approval_key: bytes | None = None,
                 allowlist: Iterable[str] = config.TOOL_ALLOWLIST, system_prompt: str = SYSTEM_PROMPT, secrets: Iterable[str] = (),
                 history_turns: int = 3, tool_timeout_s: float = 150.0):
        self.llm, self.tools, self.store = llm, tools, store
        self.budget = budget or Budget()
        self.guardrails = guardrails or GuardrailSettings()
        self.traces_dir, self.approval_key = traces_dir, approval_key
        self.allowlist, self.system_prompt = tuple(allowlist), system_prompt
        self.secrets, self.history_turns, self.tool_timeout_s = tuple(secrets), history_turns, tool_timeout_s
        self._server_tools: dict[str, dict] = {}
        self._specs: list[dict] | None = None
        self.hidden_tools: list[str] = []

    async def tool_specs(self) -> list[dict]:
        """Tools listed by the MCP server, filtered by the allowlist (the model never sees the others)."""
        if self._specs is None:
            listed = await self.tools.list_tools()
            self._server_tools = {t["name"]: t for t in listed if t["name"] in self.allowlist}
            self.hidden_tools = sorted(t["name"] for t in listed if t["name"] not in self.allowlist)
            self._specs = [to_openai_tool(t) for t in self._server_tools.values()]
        return self._specs

    # ------------------------------------------------------------------------------------------------ the loop
    async def run(self, question: str, conversation_id: str | None = None, on_event: EventCallback | None = None) -> RunResult:
        question = question.strip()
        run_id = uuid.uuid4().hex[:12]
        conversation_id = self.store.ensure_conversation(conversation_id)
        tracer = JsonlTracer(self.traces_dir, run_id, self.secrets)
        result = RunResult(run_id=run_id, conversation_id=conversation_id, question=question,
                           trace_path=str(tracer.path) if tracer.path else None)
        started = time.perf_counter()

        async def emit(kind: str, **payload: Any) -> None:
            record = tracer.event(kind, **payload)  # the trace file gets a redacted copy
            if on_event is not None:
                maybe = on_event({"type": kind, "run_id": run_id, "seq": record["seq"], "ts": record["ts"], **payload})
                if inspect.isawaitable(maybe):
                    await maybe

        history = self.store.history(conversation_id, self.history_turns)
        system = self.system_prompt + ("\n" + SPOTLIGHT_RULE if self.guardrails.spotlight else "")
        messages: list[dict] = [{"role": "system", "content": system}, *history, {"role": "user", "content": question}]
        self.store.start_run(run_id, conversation_id, question)
        self.store.add_message(conversation_id, run_id, messages[-1])
        tracker = BudgetTracker(self.budget)
        state = _RunState(user_requested_side_effect=bool(SIDE_EFFECT_INTENT.search(question)), question_ids=set(ARXIV_ID_IN_TEXT.findall(question)))
        try:
            specs = await self.tool_specs()
            await emit("run_started", conversation_id=conversation_id, question=question, model=self.llm.name,
                       history_turns=len(history) // 2, tools=[s["function"]["name"] for s in specs])
            draft, nudged = None, False
            while draft is None:
                estimate = tracker.before_step(messages, specs)
                step = tracker.steps + 1
                try:
                    reply = await asyncio.wait_for(self.llm.complete(messages, tools=specs, max_tokens=self.budget.max_output_tokens_per_step),
                                                   timeout=tracker.step_timeout())
                except TimeoutError as exc:
                    raise BudgetExceeded("max_step_latency", f"LLM step {step} exceeded {tracker.step_timeout():.0f} s") from exc
                tracker.record(reply.prompt_tokens, reply.completion_tokens)
                result.steps = tracker.steps
                result.latency_s += reply.latency_s
                await emit("llm_step", step=step, model=self.llm.name, finish_reason=reply.finish_reason, prompt_tokens=reply.prompt_tokens,
                           completion_tokens=reply.completion_tokens, estimated_prompt_tokens=estimate, latency_s=reply.latency_s,
                           cached=reply.cached, requested_tools=[c["name"] for c in reply.tool_calls])
                messages.append(reply.as_message())
                self.store.add_message(conversation_id, run_id, messages[-1])
                if reply.tool_calls:
                    tool_messages = await asyncio.gather(*(self._execute(call, step, result, state, emit) for call in reply.tool_calls))
                    for tool_message in tool_messages:
                        messages.append(tool_message)
                        self.store.add_message(conversation_id, run_id, tool_message)
                    continue
                if reply.finish_reason == "length":
                    raise BudgetExceeded("max_output_tokens_per_step", "the model used all of max_tokens before answering")
                if not (reply.content or "").strip() and not nudged:  # an empty final message: ask once for the answer
                    nudged = True
                    messages.append({"role": "user", "content": "Please write your final answer to my question now."})
                    continue
                draft = (reply.content or "").strip()
            result.draft = draft
            tracker.before_extra_call()
            raw_answer = await self._structure(question, draft, state, tracker, emit, result)
            result.raw_answer = copy.deepcopy(raw_answer)
            result.answer = self._apply_output_guardrails(raw_answer, state, result)
            for event in result.guardrail_events:
                if event.get("stage") == "output":
                    await emit("guardrail", **event)
            result.status, result.stop_reason = "completed", "final_answer"
            self.store.add_message(conversation_id, run_id, {"role": "assistant", "content": result.answer["answer"]})
        except BudgetExceeded as exc:
            result.status, result.stop_reason, result.error = "stopped", exc.reason, exc.detail
            await emit("budget_stop", reason=exc.reason, detail=exc.detail)
        except asyncio.CancelledError:
            result.status, result.stop_reason = "failed", "cancelled"
            self._finish(result, tracker, started)
            raise
        except Exception as exc:  # LLM server errors, invalid structured output, MCP transport failures
            reason = "llm_error" if type(exc).__module__.startswith("openai") else (
                "invalid_structured_output" if isinstance(exc, StructuredOutputError) else "error")
            result.status, result.stop_reason, result.error = "failed", reason, f"{type(exc).__name__}: {str(exc)[:300]}"
            await emit("error", reason=reason, error=result.error)
        self._finish(result, tracker, started)
        await emit("final", status=result.status, stop_reason=result.stop_reason, answer=result.answer, usage=result.usage(),
                   approvals=[{"id": a["id"], "tool": a["tool"], "status": a["status"]} for a in result.approvals],
                   guardrail_events=result.guardrail_events, conversation_id=conversation_id)
        return result

    def _finish(self, result: RunResult, tracker: BudgetTracker, started: float) -> None:
        result.steps, result.prompt_tokens, result.completion_tokens = tracker.steps, tracker.prompt_tokens, tracker.completion_tokens
        result.wall_clock_s = round(time.perf_counter() - started, 3)
        result.latency_s = round(result.latency_s, 3)
        self.store.finish_run(result.run_id, status=result.status, stop_reason=result.stop_reason, answer=result.answer, usage=result.usage())

    async def stream(self, question: str, conversation_id: str | None = None) -> AsyncIterator[dict]:
        """Yield events while the run is in progress (used for Server-Sent Events)."""
        queue: asyncio.Queue[dict | None] = asyncio.Queue()
        task = asyncio.create_task(self.run(question, conversation_id, on_event=queue.put))
        task.add_done_callback(lambda _: queue.put_nowait(None))
        try:
            while (event := await queue.get()) is not None:
                yield event
            await task
        finally:
            if not task.done():
                task.cancel()  # the client disconnected: stop spending tokens

    # ------------------------------------------------------------------------------------------------ tools
    async def _execute(self, call: dict, step: int, result: RunResult, state: _RunState, emit) -> dict:
        name, raw_args = call["name"], call["arguments"] or "{}"
        start = time.perf_counter()
        error, content, flags, approval_id, shown_args = None, None, [], None, raw_args
        result.tool_calls += 1
        result.tools_used.append(name)
        try:
            if name not in self._server_tools:
                raise ToolInputError(f"tool '{name}' is not available; available tools: {sorted(self._server_tools)}")
            try:
                args = json.loads(raw_args)
            except json.JSONDecodeError as exc:
                raise ToolInputError(f"arguments were not valid JSON ({exc.msg}); send one JSON object") from exc
            if not isinstance(args, dict):
                raise ToolInputError("arguments must be a JSON object")
            shown_args = args
            if any(hidden in args for hidden in approval_tokens.HIDDEN_ARGUMENTS):
                raise ToolInputError("approval fields are added by the host after a human approves; do not send them")
            schema = self._server_tools[name]["input_schema"]
            problems = [e.message for e in jsonschema.Draft202012Validator(schema).iter_errors(args)]
            if problems:
                raise ToolInputError("invalid arguments: " + "; ".join(problems[:3]))
            if name in config.APPROVAL_REQUIRED:
                content, approval_id = await self._request_side_effect(name, args, result, state, emit)
            else:
                is_error, content = await asyncio.wait_for(self.tools.call_tool(name, args), timeout=self.tool_timeout_s)
                if is_error:
                    raise ToolInputError(str(content))
                if name in UNTRUSTED_OUTPUT_TOOLS:
                    state.tainted = True
                    state.seen_ids |= _ids_in(content)
                    flags = injection_flags(json.dumps(content, ensure_ascii=False))
        except TimeoutError:
            error = f"tool '{name}' timed out after {self.tool_timeout_s:.0f} s"
        except (ToolInputError, PermissionError) as exc:
            error = str(exc)
        if error is not None:
            result.tool_errors += 1
            content = {"error": error}
        text = json.dumps(content, ensure_ascii=False, default=str)
        if error is None and name in UNTRUSTED_OUTPUT_TOOLS and self.guardrails.spotlight:
            text = spotlight(text)
        if flags:
            result.guardrail_events.append({"stage": "input", "guardrail": "injection_heuristics", "tool": name, "flags": flags})
        tool_latency = round(time.perf_counter() - start, 4)
        result.latency_s += tool_latency
        await emit("tool_call", step=step, tool=name, arguments=shown_args, latency_s=tool_latency, error=error,
                   output_chars=len(text), injection_flags=flags, approval_id=approval_id)
        return {"role": "tool", "tool_call_id": call["id"], "content": text}

    async def _request_side_effect(self, name: str, args: dict, result: RunResult, state: _RunState, emit) -> tuple[dict, str]:
        if name == "save_report":
            args = validate_report_arguments(**args)  # immediate feedback to the model; the server checks again
        notes = ["untrusted third-party text was read earlier in this run"] if state.tainted else []
        if self.guardrails.taint_policy:
            if not state.user_requested_side_effect:
                raise PermissionError(f"blocked by policy: the user's message did not ask to save or write anything, so {name} is not allowed")
            if state.tainted and (violations := side_effect_violations(args)):
                raise PermissionError("blocked by the taint policy: " + "; ".join(violations[:3]))
        approval = self.store.create_approval(result.run_id, result.conversation_id, name, args, notes)
        result.approvals.append(approval)
        if self.guardrails.require_approval:
            await emit("approval_required", approval_id=approval["id"], tool=name, arguments=args, risk_notes=notes)
            return {"status": "pending_human_approval",
                    "message": f"Not executed. A human reviewer must approve this {name} call first; tell the user it is waiting for approval."}, approval["id"]
        # approvals disabled (only in the measured attack test): the host signs and executes immediately
        decided = self.store.decide_approval(approval["id"], approved=True, reviewer="auto (approvals disabled)")
        outcome = await self._execute_approved(decided)
        result.executed_side_effects.append({"tool": name, "arguments": args, **outcome})
        return outcome, approval["id"]

    async def _execute_approved(self, approval: dict) -> dict:
        if self.approval_key is None:
            outcome = {"executed": False, "result": None, "error": "no approval signing key configured"}
        else:
            token = approval_tokens.sign(self.approval_key, approval["id"], approval["tool"], approval["arguments"])
            is_error, content = await self.tools.call_tool(approval["tool"], {**approval["arguments"], "approval_id": approval["id"],
                                                                              "approval_token": token})
            outcome = {"executed": not is_error, "result": None if is_error else content, "error": content if is_error else None}
        self.store.set_approval_result(approval["id"], outcome)
        return outcome

    async def decide(self, approval_id: str, *, approved: bool, reviewer: str) -> dict:
        """A human's decision. The stored arguments run exactly as shown — the model is not asked again."""
        record = self.store.decide_approval(approval_id, approved=approved, reviewer=reviewer)  # KeyError / ApprovalConflict
        outcome = await self._execute_approved(record) if approved else {"executed": False, "result": None, "error": None}
        if not approved:
            self.store.set_approval_result(approval_id, outcome)
        JsonlTracer(self.traces_dir, record["run_id"], self.secrets).event(
            "approval_decided", approval_id=approval_id, tool=record["tool"], approved=approved, reviewer=reviewer,
            executed=outcome["executed"], error=outcome["error"])
        return self.store.get_approval(approval_id)

    # ------------------------------------------------------------------------------------------------ final answer
    async def _structure(self, question: str, draft: str, state: _RunState, tracker: BudgetTracker, emit, result: RunResult) -> dict:
        """One schema-constrained call turns the free-text answer into FINAL_SCHEMA (more robust than a final_answer tool)."""
        rules = ("Convert the assistant's draft answer into JSON. answer: the draft's answer, faithful, at most four sentences. "
                 "citations: the arXiv identifiers the draft relies on. outcome: answered, not_found (nothing relevant was found), "
                 "awaiting_approval (an action waits for human approval), or refused.")
        context = f"Question: {question}\n"
        if self.guardrails.ground_citations:
            rules += " Choose citations only from the retrieved identifiers."
            context += f"Retrieved arXiv identifiers: {', '.join(sorted(state.seen_ids)) or 'none'}\n"
        messages = [{"role": "system", "content": rules}, {"role": "user", "content": f"{context}Draft answer: {draft}"}]
        response_format = {"type": "json_schema", "json_schema": {"name": "final_answer", "schema": FINAL_SCHEMA, "strict": True}}
        try:
            reply = await asyncio.wait_for(self.llm.complete(messages, response_format=response_format,
                                                             max_tokens=self.budget.max_output_tokens_per_step), timeout=tracker.step_timeout())
        except TimeoutError as exc:
            raise BudgetExceeded("max_step_latency", "the structuring call exceeded the step latency limit") from exc
        tracker.record(reply.prompt_tokens, reply.completion_tokens, counts_as_step=False)
        result.latency_s += reply.latency_s
        await emit("llm_format", model=self.llm.name, finish_reason=reply.finish_reason, prompt_tokens=reply.prompt_tokens,
                   completion_tokens=reply.completion_tokens, latency_s=reply.latency_s, cached=reply.cached)
        try:
            answer = json.loads(reply.content or "")
            jsonschema.validate(answer, FINAL_SCHEMA)
        except (json.JSONDecodeError, jsonschema.ValidationError) as exc:
            raise StructuredOutputError(f"final answer did not match the schema ({type(exc).__name__})") from exc
        return answer

    def _apply_output_guardrails(self, raw: dict, state: _RunState, result: RunResult) -> dict:
        answer = copy.deepcopy(raw)
        if self.guardrails.ground_citations:
            kept, removed = ground_citations(answer["citations"], state.seen_ids)
            text, scrubbed = scrub_ungrounded_ids(answer["answer"], state.seen_ids | state.question_ids)
            answer["citations"], answer["answer"] = kept, text
            if removed or scrubbed:
                result.guardrail_events.append({"stage": "output", "guardrail": "citation_grounding", "removed": removed + scrubbed})
        if self.guardrails.output_filter:
            text, removed = filter_output(answer["answer"], secrets=self.secrets)
            if removed:
                answer["answer"] = text
                result.guardrail_events.append({"stage": "output", "guardrail": "output_filter", "removed": removed})
        if any(a["status"] == "pending" for a in result.approvals):
            answer["outcome"] = "awaiting_approval"  # decided by code, not by the model
        answer["approval_ids"] = [a["id"] for a in result.approvals]
        return answer


@asynccontextmanager
async def open_agent(*, llm: ChatModel | None = None, budget: Budget | None = None, guardrails: GuardrailSettings | None = None,
                     arxiv_mode: str | None = None, extra_env: dict[str, str] | None = None, tool_server: Any = None,
                     secrets: Iterable[str] = (), system_prompt: str = SYSTEM_PROMPT, state_db: str | Path | None = None,
                     approval_key: str | None = None) -> AsyncIterator[ResearchAgent]:
    """Start the MCP server as a stdio subprocess (or use an in-memory server object) and yield a connected agent."""
    from mcp import Client, StdioServerParameters

    key = approval_key or os.environ.get("APPROVAL_SIGNING_KEY") or secrets_lib.token_hex(32)
    if tool_server is None:
        src_dir = str(Path(__file__).resolve().parents[1])  # lets the subprocess import research_agent from a source checkout
        env = {**os.environ, "APPROVAL_SIGNING_KEY": key, "PYTHONPATH": os.pathsep.join(filter(None, [src_dir, os.environ.get("PYTHONPATH")])),
               **({"ARXIV_MODE": arxiv_mode} if arxiv_mode else {}), **(extra_env or {})}
        tool_server = StdioServerParameters(command=sys.executable, args=["-m", "research_agent.mcp_server"], env=env)
    async with Client(tool_server) as client:
        agent = ResearchAgent(llm or OpenAICompatibleLLM(), MCPToolBackend(client), StateStore(state_db or config.state_db_path()),
                              budget=budget, guardrails=guardrails, traces_dir=config.traces_dir(), approval_key=key.encode(),
                              secrets=secrets, system_prompt=system_prompt)
        await agent.tool_specs()
        yield agent
