"""FastAPI service: `uvicorn research_agent.api:app --host 0.0.0.0 --port 8000`.

POST /chat · POST /chat/stream (Server-Sent Events) · GET/POST /approvals · GET /conversations/{id} · GET /runs/{id}/trace
GET /health (liveness) · GET /ready (readiness: tools, corpus, LLM)
"""

from __future__ import annotations

import asyncio
import hmac
import logging
import os
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Annotated, Any, Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Path, Request
from fastapi.responses import JSONResponse
from fastapi.sse import EventSourceResponse, ServerSentEvent

from research_agent import __version__, config
from research_agent.agent import GuardrailSettings, ResearchAgent, open_agent
from research_agent.config import Budget
from research_agent.corpus import PaperIndex
from research_agent.llm import ChatModel, OpenAICompatibleLLM
from research_agent.schemas import ApprovalDecision, ChatRequest, ChatResponse
from research_agent.store import ApprovalConflict
from research_agent.tracing import load_trace

logger = logging.getLogger("uvicorn.error")


def get_agent(request: Request) -> ResearchAgent:
    agent = request.app.state.agent
    if agent is None:
        raise HTTPException(status_code=503, detail=f"agent not ready: {request.app.state.load_error}")
    return agent


AgentDep = Annotated[ResearchAgent, Depends(get_agent)]


def create_app(*, llm: ChatModel | None = None, tool_server: Any = None, approval_key: str | None = None,
               budget: Budget | None = None, guardrails: GuardrailSettings | None = None) -> FastAPI:
    """App factory. Tests pass a stub LLM and an in-memory MCP server; production uses env settings and a stdio server."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.agent, app.state.load_error = None, None
        app.state.llm = llm or OpenAICompatibleLLM()
        async with AsyncExitStack() as stack:
            try:
                app.state.agent = await stack.enter_async_context(open_agent(
                    llm=app.state.llm, tool_server=tool_server, approval_key=approval_key, budget=budget or Budget.from_env(),
                    guardrails=guardrails))
                logger.info("agent ready: model=%s tools=%s", app.state.llm.name, [s["function"]["name"] for s in app.state.agent._specs])
            except Exception as exc:  # stay alive: /health answers, /ready explains the problem
                app.state.load_error = f"{type(exc).__name__}: {str(exc)[:200]}"
                logger.error("agent not ready: %s", app.state.load_error)
            yield
            app.state.agent = None

    app = FastAPI(title="Research Agent", version=__version__, lifespan=lifespan)

    @app.get("/health")
    def health() -> dict:
        """Liveness: the process is up."""
        return {"status": "ok"}

    @app.get("/ready")
    async def ready(request: Request) -> JSONResponse:
        """Readiness: MCP tools connected, corpus not empty, LLM server reachable."""
        checks = {"tools": "ok" if request.app.state.agent else f"unavailable: {request.app.state.load_error}"}
        try:
            n_papers = PaperIndex(config.corpus_path()).count()
            checks["corpus"] = f"ok ({n_papers} papers)" if n_papers else "empty: run python -m research_agent.ingest"
        except Exception as exc:
            checks["corpus"] = f"error: {type(exc).__name__}"
        model = request.app.state.llm
        if hasattr(model, "list_models"):
            try:
                await asyncio.wait_for(model.list_models(), timeout=5)
                checks["llm"] = "ok"
            except Exception as exc:
                checks["llm"] = f"unreachable: {type(exc).__name__}"
        else:
            checks["llm"] = "ok (test double)"
        is_ready = all(value.startswith("ok") for value in checks.values())
        return JSONResponse(status_code=200 if is_ready else 503, content={"status": "ready" if is_ready else "not ready", "checks": checks})

    @app.post("/chat", response_model=ChatResponse)
    async def chat(body: ChatRequest, agent: AgentDep) -> dict:
        return (await agent.run(body.message, body.conversation_id)).to_dict()

    @app.post("/chat/stream", response_class=EventSourceResponse)
    async def chat_stream(body: ChatRequest, agent: AgentDep) -> AsyncIterator[ServerSentEvent]:
        """The same run, streamed: run_started, llm_step, tool_call, approval_required, guardrail, budget_stop, final."""
        async for event in agent.stream(body.message, body.conversation_id):
            yield ServerSentEvent(event=event["type"], id=str(event["seq"]), data=event)

    @app.get("/conversations/{conversation_id}")
    def conversation(conversation_id: Annotated[str, Path(pattern=r"^[A-Za-z0-9_\-]{8,64}$")], agent: AgentDep) -> dict:
        if not agent.store.conversation_exists(conversation_id):
            raise HTTPException(status_code=404, detail="conversation not found")
        return {"conversation_id": conversation_id, "runs": agent.store.runs(conversation_id), "messages": agent.store.messages(conversation_id)}

    @app.get("/runs/{run_id}/trace")
    def trace(run_id: Annotated[str, Path(pattern=r"^[a-f0-9]{12}$")], agent: AgentDep) -> dict:
        path = config.traces_dir() / f"{run_id}.jsonl"
        if not path.is_file():
            raise HTTPException(status_code=404, detail="trace not found")
        return {"run_id": run_id, "events": load_trace(path)}

    @app.get("/approvals")
    def list_approvals(agent: AgentDep, status: Literal["pending", "approved", "denied"] | None = None) -> list[dict]:
        return agent.store.list_approvals(status)

    @app.post("/approvals/{approval_id}")
    async def decide(approval_id: Annotated[str, Path(pattern=r"^[a-f0-9]{16}$")], body: ApprovalDecision, agent: AgentDep,
                     x_reviewer_token: Annotated[str | None, Header()] = None) -> dict:
        """A human approves or denies a queued side effect. Set REVIEWER_TOKEN to require the X-Reviewer-Token header."""
        expected = os.environ.get("REVIEWER_TOKEN")
        if expected and not hmac.compare_digest((x_reviewer_token or "").encode(), expected.encode()):
            raise HTTPException(status_code=401, detail="a valid X-Reviewer-Token header is required")
        try:
            return await agent.decide(approval_id, approved=body.decision == "approve", reviewer=body.reviewer)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="approval not found") from exc
        except ApprovalConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    return app


app = create_app()
