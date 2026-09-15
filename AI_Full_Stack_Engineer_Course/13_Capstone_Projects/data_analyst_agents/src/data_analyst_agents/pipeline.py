"""The two systems under test.

AnalystTeam (multi-agent): planner → SQL writer (self-corrects from SQL errors) → verifier (deterministic checks +
LLM check, can send the SQL writer back once) → analyst (chart code in the sandbox) → report writer. Human approval
pauses the run before an unusually expensive query (resumable from a checkpoint) and before any export.

SingleAgentBaseline: one tool-calling agent with the full schema of every database and a run_sql tool — the strong
single-agent baseline a multi-agent design has to beat (16_Agentic_AI/04).
"""

from __future__ import annotations

import hashlib
import inspect
import json
import re
from dataclasses import dataclass, field

from data_analyst_agents.agents import SQL_RULES, Analyst, Planner, ReportWriter, RunContext, SQLWriter, Verifier, agent_span, call_llm, run_sql
from data_analyst_agents.budget import BudgetExceeded, BudgetTracker
from data_analyst_agents.config import JobBudget, QueryLimits
from data_analyst_agents.db import QueryResult, clean_sql
from data_analyst_agents.llm import LLMUnavailable, ReplayMiss
from data_analyst_agents.safety import Quarantine, export_requested
from data_analyst_agents.schema import SchemaCatalog
from data_analyst_agents.tracing import NOOP_TRACER


@dataclass
class TeamSettings:
    profile: str = "full"  # full: … → analyst → report writer · answer: planner → SQL → verifier (what execution accuracy measures)
    verifier_llm: bool = True
    guardrails: bool = True  # quarantine data cells, strip links, check numbers (False = the undefended condition in the red-team test)
    chart: bool = True
    max_sql_attempts: int = 3  # first query + up to two corrections (SQL errors or verifier feedback)
    max_verifier_repairs: int = 1
    approval_mode: str = "require"  # require: pause for expensive queries and exports · record: never pause, only count (evaluation)


@dataclass
class Outcome:
    system: str
    question: str
    evidence: str = ""
    trial: int = 0
    status: str = "running"  # completed | awaiting_approval | no_answer | budget_exceeded | failed
    database: str | None = None
    plan: dict | None = None
    sql: str | None = None
    first_sql: str | None = None
    result: dict | None = None
    attempts: list[dict] = field(default_factory=list)
    verification: dict | None = None
    first_verification: dict | None = None
    verifier_repairs: int = 0
    analysis: dict | None = None
    finding: str | None = None
    raw_finding: str | None = None
    caveats: list[str] = field(default_factory=list)
    report_markdown: str | None = None
    pending_approval: dict | None = None
    checkpoint: dict | None = None
    would_need_approval: list[str] = field(default_factory=list)
    guardrail_events: list[dict] = field(default_factory=list)
    stop_reason: str | None = None
    error: str | None = None
    usage: dict = field(default_factory=dict)
    result_obj: QueryResult | None = field(default=None, repr=False)
    first_result_obj: QueryResult | None = field(default=None, repr=False)
    artifacts: dict[str, bytes] = field(default_factory=dict, repr=False)

    def to_dict(self) -> dict:
        skip = {"result_obj", "first_result_obj", "artifacts"}
        return json.loads(json.dumps({k: v for k, v in self.__dict__.items() if k not in skip}, default=str))


def _emitter(on_event):
    async def emit(kind: str, data: dict) -> None:
        if on_event is None:
            return
        value = on_event(kind, data)
        if inspect.isawaitable(value):
            await value

    return emit


class AnalystTeam:
    name = "multi_agent"

    def __init__(self, llm, catalogs: dict[str, SchemaCatalog], *, settings: TeamSettings | None = None, budget: JobBudget | None = None,
                 tracer=NOOP_TRACER, limits: QueryLimits | None = None):
        self.llm, self.catalogs = llm, catalogs
        self.settings = settings or TeamSettings()
        self.budget = budget or JobBudget()
        self.tracer = tracer
        self.limits = limits or QueryLimits()
        self.planner = Planner(llm, catalogs)
        self.sql_writer = SQLWriter(llm, catalogs)
        self.verifier = Verifier(llm, use_llm=self.settings.verifier_llm)
        self.analyst = Analyst(llm)
        self.reporter = ReportWriter(llm, guardrails=self.settings.guardrails)

    async def run(self, question: str, evidence: str = "", *, trial: int = 0, on_event=None, checkpoint: dict | None = None,
                  approvals: frozenset[str] = frozenset(), job_id: str | None = None) -> Outcome:
        out = Outcome(system=self.name, question=question, evidence=evidence, trial=trial)
        tracker = BudgetTracker(self.budget)
        ctx = RunContext(question, evidence, tracker, Quarantine(enabled=self.settings.guardrails), self.tracer, trial, _emitter(on_event))
        attributes = {"gen_ai.operation.name": "invoke_workflow", "gen_ai.workflow.name": "data-analyst-team", "app.job_id": job_id or "",
                      "app.trial": trial, "app.profile": self.settings.profile, "app.resumed": bool(checkpoint)}
        with self.tracer.start_as_current_span("invoke_workflow data-analyst-team", attributes=attributes) as root:
            try:
                await self._run(ctx, out, checkpoint or {}, approvals)
            except BudgetExceeded as err:
                out.status, out.stop_reason, out.error = "budget_exceeded", err.reason, str(err)
                await ctx.event("budget_stop", reason=err.reason, detail=err.detail)
            except (LLMUnavailable, ReplayMiss) as err:
                out.status, out.error = "failed", f"{type(err).__name__}: {err}"
            finally:
                out.usage, out.guardrail_events = tracker.usage(), ctx.guardrail_events
                root.set_attributes({"app.status": out.status, "app.llm_calls": tracker.llm_calls, "gen_ai.usage.input_tokens": tracker.prompt_tokens,
                                     "gen_ai.usage.output_tokens": tracker.completion_tokens})
        await ctx.event("finished", status=out.status, usage=out.usage, error=out.error)
        return out

    async def _run(self, ctx: RunContext, out: Outcome, checkpoint: dict, approvals: frozenset[str]) -> None:
        out.attempts = list(checkpoint.get("attempts", []))
        if "plan" in checkpoint:
            plan = checkpoint["plan"]
        else:
            with agent_span(ctx, "planner"):
                plan = await self.planner.run(ctx)
            await ctx.event("plan", **plan)
        out.plan, out.database = plan, plan["database"]
        db = self.catalogs[plan["database"]].db
        result = await self._sql_loop(ctx, out, plan, db, feedback=checkpoint.get("feedback"), pending_sql=checkpoint.get("pending_sql"),
                                      source=checkpoint.get("source", "initial"), approvals=approvals)
        if result is None:
            if out.status != "awaiting_approval":
                out.status, out.error = "no_answer", f"no query succeeded in {len(out.attempts)} attempt(s)"
            return
        verification = await self._verify(ctx, out, plan, db, result)
        while verification["flagged"] and out.verifier_repairs < self.settings.max_verifier_repairs and len(out.attempts) < self.settings.max_sql_attempts:
            out.verifier_repairs += 1
            await ctx.event("repair", problems=verification["problems"])
            feedback = {"sql": result.sql, "problem": "the verifier flagged the result: " + " | ".join(verification["problems"])[:800]}
            candidate = await self._sql_loop(ctx, out, plan, db, feedback=feedback, source="verifier", approvals=approvals)
            if out.status == "awaiting_approval":
                return
            if candidate is None:
                break
            result = candidate
            verification = await self._verify(ctx, out, plan, db, result)
        out.verification = verification
        out.sql, out.result, out.result_obj = result.sql, result.to_dict(self.limits.preview_rows), result
        await ctx.event("answer_ready", sql=result.sql, n_rows=len(result.rows), columns=result.columns, flagged=verification["flagged"])
        if self.settings.profile == "full":
            if plan["needs_chart"] and self.settings.chart:
                with agent_span(ctx, "analyst"):
                    out.analysis, out.artifacts = await self.analyst.run(ctx, result)
            else:
                out.analysis = {"status": "skipped", "reason": "the planner did not ask for a chart" if self.settings.chart else "charts disabled"}
            with agent_span(ctx, "report_writer"):
                report = await self.reporter.run(ctx, plan, result, verification, out.analysis)
            out.finding, out.raw_finding, out.caveats, out.report_markdown = report["finding"], report["raw_finding"], report["caveats"], report["markdown"]
            await ctx.event("report_ready", finding=out.finding, chart=out.analysis.get("status") == "ok")
        out.status = "completed"
        if export_requested(ctx.question):  # intent from the user's words only; a model or a data cell can't request it
            out.would_need_approval.append("export")
            if self.settings.approval_mode == "require" and "export" not in approvals:
                out.status = "awaiting_approval"
                out.pending_approval = {"kind": "export", "key": "export", "database": plan["database"], "sql": result.sql, "rows": len(result.rows),
                                        "columns": result.columns, "masked_columns": result.masked_columns,
                                        "reason": "the user asked for an export: data would leave the service"}
                await ctx.event("approval_required", **out.pending_approval)

    async def _sql_loop(self, ctx: RunContext, out: Outcome, plan: dict, db, *, feedback: dict | None = None, pending_sql: str | None = None,
                        source: str = "initial", approvals: frozenset[str] = frozenset()) -> QueryResult | None:
        while True:
            approved = pending_sql is not None
            if approved:
                sql, pending_sql = pending_sql, None  # the exact query the reviewer approved, never regenerated
            else:
                if len(out.attempts) >= self.settings.max_sql_attempts:
                    return None
                with agent_span(ctx, "sql_writer"):
                    sql = await self.sql_writer.run(ctx, plan, feedback)
                if not sql:
                    out.attempts.append({"sql": None, "source": source, "error_kind": "format", "error": "the SQL writer's reply had no usable query", "n_rows": None})
                    feedback, source = {"sql": "", "problem": "your reply did not contain a query"}, "format_error"
                    continue
                cost = db.estimate_cost(sql)
                if cost["expensive"]:
                    key = f"expensive_query:{hashlib.sha256(sql.encode()).hexdigest()[:12]}"
                    out.would_need_approval.append(key)
                    if self.settings.approval_mode == "require" and key not in approvals:
                        out.status = "awaiting_approval"
                        out.pending_approval = {"kind": "expensive_query", "key": key, "database": db.db_id, "sql": sql, "scanned_rows": cost["scanned_rows"],
                                                "full_scans": cost["full_scans"], "reason": f"full table scans over {cost['scanned_rows']:,} rows"}
                        out.checkpoint = {"plan": plan, "pending_sql": sql, "attempts": out.attempts, "feedback": feedback, "source": source}
                        await ctx.event("approval_required", **out.pending_approval)
                        return None
            result = await run_sql(db, ctx, sql)
            attempt = {"sql": result.sql, "source": source, "error_kind": result.error_kind, "error": result.error,
                       "n_rows": len(result.rows) if result.ok else None, "elapsed_s": result.elapsed_s}
            out.attempts.append(attempt)
            await ctx.event("sql_attempt", n=len(out.attempts), **attempt)
            if result.ok:
                if out.first_sql is None:
                    out.first_sql, out.first_result_obj = result.sql, result
                return result
            feedback, source = {"sql": result.sql, "problem": f"{result.error_kind}: {result.error}"}, "sql_error"

    async def _verify(self, ctx: RunContext, out: Outcome, plan: dict, db, result: QueryResult) -> dict:
        with agent_span(ctx, "verifier"):
            verification = await self.verifier.run(ctx, plan, self.catalogs[plan["database"]], result)
        if out.first_verification is None:
            out.first_verification = verification
        await ctx.event("verification", flagged=verification["flagged"], problems=verification["problems"], llm_verdict=verification["llm_verdict"])
        return verification


# ---------------------------------------------------------------- the single-agent baseline
SINGLE_SYSTEM = """You are a data analyst with read-only SQL access to the SQLite databases described below. Answer the business question by
writing SQL and checking it with the run_sql tool. When you are confident, reply WITHOUT a tool call, in this format:
Database: <database name>
```sql
<the final query>
```
<one sentence with the answer>

""" + SQL_RULES


def parse_final_answer(text: str, db_ids: list[str]) -> tuple[str | None, str | None]:
    blocks = re.findall(r"```(?:sql|sqlite)?\s*(.*?)```", text or "", re.S | re.I)
    sql = clean_sql(blocks[-1]) if blocks else None
    match = re.search(r"database\s*:\s*`?([\w]+)`?", text or "", re.I)
    db_id = match.group(1) if match and match.group(1) in db_ids else None
    return sql or None, db_id


class SingleAgentBaseline:
    name = "single_agent"

    def __init__(self, llm, catalogs: dict[str, SchemaCatalog], *, budget: JobBudget | None = None, tracer=NOOP_TRACER, max_steps: int = 5,
                 limits: QueryLimits | None = None):
        self.llm, self.catalogs, self.tracer, self.max_steps = llm, catalogs, tracer, max_steps
        self.budget = budget or JobBudget()
        self.limits = limits or QueryLimits()
        self.system_prompt = SINGLE_SYSTEM + "\n\n" + "\n\n".join(c.render(max_chars=30_000) for c in catalogs.values())
        self.tool = {"type": "function", "function": {
            "name": "run_sql", "description": "Run one read-only SQLite query. Returns the columns, up to 10 rows and the row count, or the error.",
            "parameters": {"type": "object", "properties": {"database": {"type": "string", "enum": list(catalogs)}, "sql": {"type": "string"}},
                           "required": ["database", "sql"]}}}

    async def run(self, question: str, evidence: str = "", *, trial: int = 0, on_event=None, job_id: str | None = None, **_) -> Outcome:
        out = Outcome(system=self.name, question=question, evidence=evidence, trial=trial)
        tracker = BudgetTracker(self.budget)
        ctx = RunContext(question, evidence, tracker, Quarantine(enabled=False), self.tracer, trial, _emitter(on_event))
        with self.tracer.start_as_current_span("invoke_agent single_agent", attributes={"gen_ai.operation.name": "invoke_agent", "gen_ai.agent.name": "single_agent",
                                                                                         "app.job_id": job_id or "", "app.trial": trial}) as root:
            try:
                await self._run(ctx, out)
            except BudgetExceeded as err:
                out.status, out.stop_reason, out.error = "budget_exceeded", err.reason, str(err)
            except (LLMUnavailable, ReplayMiss) as err:
                out.status, out.error = "failed", f"{type(err).__name__}: {err}"
            finally:
                out.usage = tracker.usage()
                root.set_attributes({"app.status": out.status, "app.llm_calls": tracker.llm_calls})
        return out

    async def _run(self, ctx: RunContext, out: Outcome) -> None:
        messages = [{"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"Question: {ctx.question}\nHint from the business glossary (use it literally): {ctx.evidence or '(none)'}"}]
        last_ok: tuple[str, QueryResult] | None = None
        final_text = None
        for _step in range(self.max_steps):
            reply = await call_llm(self.llm, ctx, agent="single_agent", messages=messages, tools=[self.tool], max_tokens=1536)
            messages.append(reply.as_message())
            if not reply.tool_calls:
                final_text = reply.content or ""
                break
            for call in reply.tool_calls:
                try:
                    args = json.loads(call["arguments"])
                    db_id, sql = args["database"], str(args["sql"])
                    db = self.catalogs[db_id].db
                except (json.JSONDecodeError, KeyError, TypeError) as err:
                    content = json.dumps({"error": f"invalid arguments ({type(err).__name__}: {err})"})
                else:
                    result = await run_sql(db, ctx, sql)
                    out.attempts.append({"sql": result.sql, "database": db_id, "source": "tool", "error_kind": result.error_kind, "error": result.error,
                                         "n_rows": len(result.rows) if result.ok else None, "elapsed_s": result.elapsed_s})
                    if result.ok:
                        last_ok = (db_id, result)
                        if out.first_sql is None:
                            out.first_sql, out.first_result_obj = result.sql, result
                    content = json.dumps({"columns": result.columns, "rows": [list(r) for r in result.rows[:10]], "n_rows": len(result.rows),
                                          "error": result.error}, default=str, ensure_ascii=False)
                messages.append({"role": "tool", "tool_call_id": call["id"], "content": content})
        else:
            out.stop_reason = "max_steps"
        sql, db_id = parse_final_answer(final_text, list(self.catalogs)) if final_text else (None, None)
        if sql is None:
            if last_ok is None:
                out.status, out.error = "no_answer", "no final query and no successful run_sql call"
                return
            db_id, result = last_ok
            out.stop_reason = out.stop_reason or "no final query: used the last successful run_sql"
        else:
            db_id = db_id or (last_ok[0] if last_ok else None) or (out.attempts[-1]["database"] if out.attempts else next(iter(self.catalogs)))
            if last_ok and last_ok[0] == db_id and last_ok[1].sql == sql:
                result = last_ok[1]
            else:
                result = await run_sql(self.catalogs[db_id].db, ctx, sql, purpose="final_answer")
        out.database, out.sql, out.result, out.result_obj = db_id, result.sql, result.to_dict(self.limits.preview_rows), result
        if out.first_sql is None and result.ok:
            out.first_sql, out.first_result_obj = result.sql, result
        out.finding = (final_text or "").split("```")[-1].strip()[:500] or None
        out.status = "completed" if result.ok else "no_answer"
        if not result.ok:
            out.error = f"final query failed: {result.error_kind}: {result.error}"
