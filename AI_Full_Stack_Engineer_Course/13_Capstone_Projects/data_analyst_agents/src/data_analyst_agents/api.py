"""FastAPI service: `uvicorn data_analyst_agents.api:app --host 0.0.0.0 --port 8000`.

POST /jobs (202, Idempotency-Key) · GET /jobs/{id} · GET /jobs/{id}/events (Server-Sent Events, Last-Event-ID) ·
POST /jobs/{id}/cancel · POST /jobs/{id}/approvals (X-Reviewer-Token) · GET /jobs/{id}/report · /chart.png · /export.csv ·
GET /health (liveness) · GET /ready (readiness: databases, LLM, queue)
"""

from __future__ import annotations

import asyncio
import hmac
import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Path, Request, Response
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.sse import EventSourceResponse, ServerSentEvent
from pydantic import BaseModel, ConfigDict, Field

from data_analyst_agents import __version__, config
from data_analyst_agents.config import JobBudget
from data_analyst_agents.jobs import TERMINAL, Conflict, JobManager, JobStore, NotFound, QueueFull
from data_analyst_agents.llm import OpenAICompatibleLLM
from data_analyst_agents.pipeline import AnalystTeam, TeamSettings
from data_analyst_agents.schema import load_catalogs
from data_analyst_agents.tracing import Tracing

logger = logging.getLogger("uvicorn.error")
JobId = Annotated[str, Path(pattern=r"^[a-f0-9]{16}$")]


class JobRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str = Field(min_length=5, max_length=1000)
    evidence: str = Field(default="", max_length=2000, description="optional business-glossary hint, e.g. what a column code means")


class Decision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    approval_key: str = Field(min_length=1, max_length=80)
    decision: Literal["approve", "deny"]
    reviewer: str = Field(min_length=1, max_length=80)


def public_job(job: dict) -> dict:
    outcome = job["outcome"] or {}
    result = outcome.get("result") or {}
    base = f"/jobs/{job['id']}"
    return {
        "id": job["id"], "status": job["status"], "question": job["request"]["question"], "created_at": job["created_at"], "started_at": job["started_at"],
        "finished_at": job["finished_at"], "error": job["error"], "pending_approval": job["pending"],
        "result": {"database": outcome.get("database"), "sql": outcome.get("sql"), "columns": result.get("columns"), "n_rows": result.get("n_rows"),
                   "preview": (result.get("preview") or [])[:10], "masked_columns": result.get("masked_columns"), "finding": outcome.get("finding"),
                   "caveats": outcome.get("caveats"), "verification": {k: (outcome.get("verification") or {}).get(k) for k in ("flagged", "problems", "llm_verdict")},
                   "chart": (outcome.get("analysis") or {}).get("status"), "usage": outcome.get("usage"), "guardrail_events": outcome.get("guardrail_events")} if outcome else None,
        "links": {"self": base, "events": f"{base}/events", "report": f"{base}/report", "chart": f"{base}/chart.png", "export": f"{base}/export.csv"},
    }


# Dependencies live at module level: with `from __future__ import annotations`, FastAPI resolves annotations by name in
# the module's globals, so aliases defined inside create_app() would silently turn into query parameters.
def get_manager(request: Request) -> JobManager:
    if request.app.state.manager is None:
        raise HTTPException(status_code=503, detail=f"service not ready: {request.app.state.load_error}")
    return request.app.state.manager


Manager = Annotated[JobManager, Depends(get_manager)]


def existing_job(job_id: JobId, manager: Manager) -> dict:
    try:
        return manager.store.get(job_id)
    except NotFound as exc:
        raise HTTPException(status_code=404, detail="job not found") from exc


Job = Annotated[dict, Depends(existing_job)]


def create_app(*, llm=None, catalogs=None, team_settings: TeamSettings | None = None, budget: JobBudget | None = None, workers: int | None = None) -> FastAPI:
    """App factory. Tests pass a FakeLLM and a fixture catalog; production reads everything from the environment."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.manager, app.state.load_error = None, None
        app.state.llm = llm or OpenAICompatibleLLM()
        tracing = Tracing(config.traces_path())
        try:
            cats = catalogs if catalogs is not None else await asyncio.to_thread(load_catalogs)
            team = AnalystTeam(app.state.llm, cats, settings=team_settings or TeamSettings(), budget=budget or JobBudget.from_env(), tracer=tracing.tracer)
            manager = JobManager(team, JobStore(config.service_dir() / "jobs.sqlite"), workers=workers or int(os.environ.get("ANALYST_WORKERS", "2")),
                                 artifacts_dir=config.artifacts_dir(), exports_dir=config.exports_dir())
            await manager.start()
            app.state.manager = manager
            logger.info("analyst team ready: databases=%s model=%s", list(cats), app.state.llm.name)
        except Exception as exc:  # stay alive: /health answers, /ready explains
            app.state.load_error = f"{type(exc).__name__}: {str(exc)[:200]}"
            logger.error("service not ready: %s", app.state.load_error)
        yield
        if app.state.manager is not None:
            await app.state.manager.stop()
            app.state.manager.store.close()
        tracing.shutdown()

    app = FastAPI(title="AI Data Analyst Team", version=__version__, lifespan=lifespan)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/ready")
    async def ready(request: Request) -> JSONResponse:
        manager = request.app.state.manager
        checks = {"databases": f"ok ({', '.join(manager.team.catalogs)})" if manager else f"unavailable: {request.app.state.load_error}"}
        try:
            await asyncio.wait_for(request.app.state.llm.list_models(), timeout=5)
            checks["llm"] = "ok"
        except Exception as exc:
            checks["llm"] = f"unreachable: {type(exc).__name__}"
        checks["queue"] = f"ok ({manager.queue.qsize()}/{manager.queue_max} waiting)" if manager else "unavailable"
        is_ready = all(v.startswith("ok") for v in checks.values())
        return JSONResponse(status_code=200 if is_ready else 503, content={"status": "ready" if is_ready else "not ready", "checks": checks})

    @app.post("/jobs", status_code=202)
    async def create_job(body: JobRequest, manager: Manager, response: Response,
                         idempotency_key: Annotated[str | None, Header(max_length=128)] = None) -> dict:
        try:
            job, created = await manager.submit(body.question, body.evidence, idempotency_key)
        except Conflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except QueueFull as exc:
            raise HTTPException(status_code=429, detail=f"queue full ({exc}); retry later", headers={"Retry-After": "30"}) from exc
        if not created:
            response.status_code = 200  # the same request with the same key: the original job, no second run
        return public_job(job)

    @app.get("/jobs/{job_id}")
    def get_job(job: Job) -> dict:
        return public_job(job)

    @app.get("/jobs/{job_id}/events", response_class=EventSourceResponse)
    async def job_events(job: Job, manager: Manager, last_event_id: Annotated[str | None, Header()] = None) -> AsyncIterator[ServerSentEvent]:
        """Progress events as they happen; reconnecting clients resume after Last-Event-ID. Ends when the job is finished."""
        after = int(last_event_id) if last_event_id and last_event_id.isdigit() else 0
        while True:
            waiter = manager.waiter(job["id"])
            for event in manager.store.events(job["id"], after):
                after = event["seq"]
                yield ServerSentEvent(event=event["type"], id=str(event["seq"]), data=event["data"])
            if manager.store.get(job["id"])["status"] in TERMINAL and not manager.store.events(job["id"], after):
                return
            try:
                await asyncio.wait_for(waiter.wait(), timeout=15)
            except TimeoutError:
                yield ServerSentEvent(comment="keep-alive")  # keeps proxies from closing an idle stream

    @app.post("/jobs/{job_id}/cancel", status_code=202)
    async def cancel_job(job: Job, manager: Manager) -> dict:
        try:
            return public_job(await manager.cancel(job["id"]))
        except Conflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/jobs/{job_id}/approvals")
    async def decide(job: Job, body: Decision, manager: Manager, x_reviewer_token: Annotated[str | None, Header()] = None) -> dict:
        """A human approves or denies an expensive query or an export. Set REVIEWER_TOKEN to require X-Reviewer-Token."""
        expected = os.environ.get("REVIEWER_TOKEN")
        if expected and not hmac.compare_digest((x_reviewer_token or "").encode(), expected.encode()):
            raise HTTPException(status_code=401, detail="a valid X-Reviewer-Token header is required")
        try:
            return public_job(await manager.decide(job["id"], body.approval_key, body.decision == "approve", body.reviewer))
        except Conflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/jobs/{job_id}/report", response_class=PlainTextResponse)
    def report(job: Job) -> str:
        markdown = (job["outcome"] or {}).get("report_markdown")
        if not markdown:
            raise HTTPException(status_code=404, detail="no report for this job (yet)")
        return markdown

    @app.get("/jobs/{job_id}/chart.png")
    def chart(job: Job) -> FileResponse:
        path = config.artifacts_dir() / job["id"] / "chart.png"
        if not path.is_file():
            raise HTTPException(status_code=404, detail="no chart for this job")
        return FileResponse(path, media_type="image/png")

    @app.get("/jobs/{job_id}/export.csv")
    def export(job: Job, manager: Manager) -> FileResponse:
        record = manager.store.export(job["id"])
        if record is None:
            raise HTTPException(status_code=404, detail="no approved export for this job")
        return FileResponse(record["path"], media_type="text/csv", filename=f"{job['id']}.csv")

    return app


app = create_app()
