"""FastAPI service: `uvicorn rag_assistant.api:app --host 0.0.0.0 --port 8000`.

POST /ask     question → cited answer (JSON), or Server-Sent Events when "stream": true
POST /ingest  incremental re-index of the corpus folder, then an atomic index swap (needs the X-API-Key header)
GET  /health  liveness · GET /ready readiness · GET /stats cache and status counters
"""

from __future__ import annotations

import json
import logging
import secrets
import threading
from contextlib import asynccontextmanager
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from rag_assistant import __version__
from rag_assistant.config import Settings
from rag_assistant.corpus import load_sources
from rag_assistant.guardrails import InputRejected
from rag_assistant.index import IndexStore, ingest
from rag_assistant.pipeline import RAGPipeline

logger = logging.getLogger("uvicorn.error")


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str = Field(min_length=1, max_length=20_000, description="Hard cap for the request body; the configurable limit answers 413.")
    stream: bool = Field(default=False, description="true → text/event-stream with meta, delta and final events")
    top_k: int | None = Field(default=None, ge=1, le=10, description="passages in the prompt (default from settings)")


class CitationOut(BaseModel):
    chunk_id: str
    quote: str
    url: str
    section: str


class AskResponse(BaseModel):
    request_id: str
    status: Literal["answered", "refused", "unsupported", "blocked_output", "invalid_output"]
    answer: str
    citations: list[CitationOut]
    cache: str
    index_version: str
    config: str
    timings_ms: dict[str, float]
    tokens: dict[str, int | None]


def get_pipeline(request: Request) -> RAGPipeline:
    pipeline = request.app.state.pipeline
    if pipeline is None:
        raise HTTPException(status_code=503, detail=f"not ready: {request.app.state.load_error}")
    return pipeline


PipelineDep = Annotated[RAGPipeline, Depends(get_pipeline)]


def sse_event(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def create_app(settings: Settings | None = None, *, pipeline_factory=None) -> FastAPI:
    """App factory: tests inject settings and a pipeline factory (stub LLM, test embedder); production reads the environment."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.settings = settings or Settings.from_env()
        app.state.pipeline, app.state.load_error, app.state.ingest_lock = None, None, threading.Lock()
        try:
            pipeline = (pipeline_factory or RAGPipeline.from_settings)(app.state.settings)
            pipeline.warm_up()  # load models at startup, not on the first user request
            app.state.pipeline = pipeline
            logger.info("index %s loaded: %d chunks", pipeline.index.version, len(pipeline.index.chunks))
        except Exception as err:  # noqa: BLE001 — stay alive so /health answers and /ready explains the problem
            app.state.load_error = f"{type(err).__name__}: {err}"
            logger.error("pipeline not loaded: %s", app.state.load_error)
        yield
        app.state.pipeline = None

    app = FastAPI(title="uv Docs Assistant", version=__version__, lifespan=lifespan)

    @app.get("/health")
    def health() -> dict:
        """Liveness: the process is up."""
        return {"status": "ok"}

    @app.get("/ready")
    def ready(request: Request):
        """Readiness: index and models are loaded. LLM reachability is reported but does not fail readiness."""
        pipeline = request.app.state.pipeline
        if pipeline is None:
            return JSONResponse(status_code=503, content={"status": "not ready", "reason": request.app.state.load_error})
        return {"status": "ready", "index_version": pipeline.index.version, "chunks": len(pipeline.index.chunks),
                "llm_reachable": pipeline.llm.ping(timeout=1.0)}

    # Plain `def`: retrieval is CPU work and the LLM client is blocking, so FastAPI runs these in its thread pool
    @app.post("/ask", response_model=None, responses={200: {"model": AskResponse, "content": {"text/event-stream": {}}}})
    def ask(body: AskRequest, pipeline: PipelineDep):
        events = pipeline.ask_events(body.question, stream=body.stream, top_k=body.top_k)
        try:
            first = next(events)  # runs the input guardrail before any byte is sent, so limits still get a proper status code
        except InputRejected as err:
            raise HTTPException(status_code=err.status_code, detail=err.reason) from None
        if body.stream:
            def event_stream():
                yield sse_event(*first)
                for event, data in events:
                    yield sse_event(event, data)

            return StreamingResponse(event_stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
        for event, data in events:
            if event == "error":
                raise HTTPException(status_code=503, detail=data["message"])
            if event == "final":
                return data
        raise HTTPException(status_code=500, detail="no answer produced")

    @app.post("/ingest")
    def ingest_corpus(request: Request, pipeline: PipelineDep, x_api_key: Annotated[str | None, Header()] = None) -> dict:
        settings = request.app.state.settings
        if not settings.ingest_api_key:
            raise HTTPException(status_code=503, detail="ingestion is disabled: set RAG_INGEST_API_KEY")
        if x_api_key is None or not secrets.compare_digest(x_api_key.encode(), settings.ingest_api_key.encode()):
            raise HTTPException(status_code=401, detail="invalid or missing API key", headers={"WWW-Authenticate": "X-API-Key"})
        lock = request.app.state.ingest_lock
        if not lock.acquire(blocking=False):
            raise HTTPException(status_code=409, detail="an ingestion is already running")
        try:
            store = IndexStore(settings.index_dir)
            report = ingest(load_sources(settings.corpus_dir), store, pipeline.embedder, chunk_tokens=settings.chunk_tokens,
                            chunk_overlap=settings.chunk_overlap)
            if report.wrote_new_version:
                pipeline.swap_index(store.load())
        finally:
            lock.release()
        return {**report.to_dict(), "active_version": pipeline.index.version}

    @app.get("/stats")
    def stats(pipeline: PipelineDep) -> dict:
        cache = pipeline.cache
        return {"index_version": pipeline.index.version, "requests_by_status": dict(pipeline.status_counts),
                "cache": {"mode": cache.mode, "entries": len(cache), "hit_rate": round(cache.hit_rate, 4), **cache.stats}}

    return app


app = create_app()
