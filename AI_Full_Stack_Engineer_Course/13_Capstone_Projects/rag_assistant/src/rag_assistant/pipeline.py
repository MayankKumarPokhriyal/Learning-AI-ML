"""The request pipeline shared by the API, the CLI and the evaluation harness:
guard input → cache lookup → retrieve → generate (streamed or not) → validate citations and links → cache → trace."""

from __future__ import annotations

import json
import threading
import time
import uuid
from collections import Counter
from collections.abc import Iterator
from dataclasses import replace

from rag_assistant.cache import ResponseCache
from rag_assistant.config import Settings
from rag_assistant.embeddings import CrossEncoderReranker, SentenceTransformerEmbedder
from rag_assistant.generation import ANSWER_SCHEMA, REFUSAL_TEXT, AnswerFieldStream, FinalAnswer, build_messages, finalize
from rag_assistant.guardrails import InputRejected, UrlHoldback, check_question, redact
from rag_assistant.index import Index, IndexStore
from rag_assistant.llm import LLMResult, OpenAICompatibleLLM
from rag_assistant.retrieval import RetrievalConfig, RetrievalResult, Retriever
from rag_assistant.tracing import TraceWriter, utc_now

CACHEABLE_STATUSES = {"answered", "refused"}


class RAGPipeline:
    def __init__(self, settings: Settings, index: Index, embedder, reranker, llm, *, cache: ResponseCache | None = None,
                 tracer: TraceWriter | None = None):
        self.settings, self.embedder, self.reranker, self.llm = settings, embedder, reranker, llm
        self.retriever = Retriever(index, embedder, reranker)
        self.cache = cache if cache is not None else ResponseCache(settings.cache_mode, settings.cache_threshold, settings.cache_max_entries,
                                                                   settings.cache_ttl_seconds)
        self.tracer = tracer if tracer is not None else TraceWriter(settings.trace_path)
        self.status_counts: Counter = Counter()
        self._lock = threading.Lock()

    @classmethod
    def from_settings(cls, settings: Settings, *, llm=None, embedder=None, reranker=None, cache=None, tracer=None) -> RAGPipeline:
        index = IndexStore(settings.index_dir).load()
        embedder = embedder or SentenceTransformerEmbedder(settings.embedding_model, device=settings.device, query_prompt=settings.query_prompt)
        if index.manifest["params"]["embedding_model"] != embedder.name:  # querying with a different model returns garbage silently
            raise ValueError(f"index {index.version} was built with {index.manifest['params']['embedding_model']}, not {embedder.name}; re-ingest")
        if reranker is None and settings.rerank:
            reranker = CrossEncoderReranker(settings.reranker_model, device=settings.device)
        llm = llm or OpenAICompatibleLLM(settings.llm_base_url, settings.llm_model, settings.llm_api_key, timeout=settings.llm_timeout,
                                         reasoning_effort=settings.reasoning_effort, record_dir=settings.llm_cache_dir,
                                         replay_realtime=settings.llm_replay_realtime)
        return cls(settings, index, embedder, reranker, llm, cache=cache, tracer=tracer)

    @property
    def index(self) -> Index:
        return self.retriever.index

    def warm_up(self) -> None:
        """Load the models now instead of on the first user request."""
        self.embedder.embed_query("warm up")
        if self.reranker is not None:
            self.reranker.score("warm up", ["warm up"])

    def release_models(self) -> None:
        self.embedder.release()
        if self.reranker is not None:
            self.reranker.release()

    def swap_index(self, index: Index) -> None:
        retriever = Retriever(index, self.embedder, self.reranker)  # build fully, then switch in one assignment
        with self._lock:
            self.retriever = retriever
        self.cache.clear()  # cached answers belong to the old index (the scope key would also miss)

    def retrieval_config(self, **overrides) -> RetrievalConfig:
        return replace(RetrievalConfig.from_settings(self.settings), **{k: v for k, v in overrides.items() if v is not None})

    def retrieve(self, question: str, config: RetrievalConfig | None = None, query_vector=None) -> RetrievalResult:
        return self.retriever.retrieve(question, config or self.retrieval_config(), query_vector)

    def _allowed_domains(self, output_url_check: bool | None = None):
        check = self.settings.output_url_check if output_url_check is None else output_url_check
        return self.settings.allowed_answer_domains if check else None

    def generate(self, question: str, retrieval: RetrievalResult, *, prompt_defense: bool | None = None, output_url_check: bool | None = None,
                 temperature: float | None = None) -> tuple[FinalAnswer, LLMResult]:
        """Non-streaming generation + answer policy (used by the evaluation harness and experiments)."""
        if not retrieval.selected:
            return FinalAnswer("refused", REFUSAL_TEXT, draft_answer="(no passages retrieved)"), LLMResult("", 0, 0, 0.0, None, "no_context")
        defense = self.settings.prompt_defense if prompt_defense is None else prompt_defense
        result = self.llm.complete(build_messages(question, retrieval.selected, defense), max_tokens=self.settings.max_output_tokens,
                                   temperature=self.settings.temperature if temperature is None else temperature,
                                   json_schema=ANSWER_SCHEMA, schema_name="grounded_answer")
        return finalize(result.text, retrieval.selected, self._allowed_domains(output_url_check)), result

    def ask(self, question: str, **kwargs) -> dict:
        """Non-streaming convenience wrapper around ask_events: returns the final payload."""
        for event, data in self.ask_events(question, stream=False, **kwargs):
            if event == "error":
                raise RuntimeError(data["message"])
            if event == "final":
                return data
        raise RuntimeError("pipeline ended without a final event")

    def ask_events(self, question: str, *, stream: bool = True, use_cache: bool = True, top_k: int | None = None) -> Iterator[tuple[str, dict]]:
        """Yields ("meta", …), zero or more ("delta", {"text"}), then ("final", payload) or ("error", …).
        Raises InputRejected (before yielding anything) when the question breaks the input limits."""
        started = time.perf_counter()
        retriever, settings = self.retriever, self.settings  # one index for the whole request, even if /ingest swaps it meanwhile
        config = self.retrieval_config(top_k=top_k)
        request_id = uuid.uuid4().hex[:16]
        trace = {"request_id": request_id, "timestamp": utc_now(), "index_version": retriever.index.version, "config": config.label,
                 "stream": stream, "prompt_defense": settings.prompt_defense}
        timings: dict[str, float] = {}

        def elapsed_ms(since: float) -> float:
            return (time.perf_counter() - since) * 1000

        try:
            q = check_question(question, settings.max_question_chars, settings.max_question_tokens)
        except InputRejected as err:
            trace.update(status="rejected", reason=err.reason, question=redact(question[:300])[0], question_chars=len(question))
            self._finish(trace, "rejected")
            raise
        timings["guard_input"] = elapsed_ms(started)
        redacted_question, redactions = redact(q)
        trace.update(question=redacted_question, question_chars=len(q), redactions=redactions)

        scope = f"{retriever.index.version}|{config.label}|defense={settings.prompt_defense}|quarantine={config.quarantine_flagged}"
        hit, vector = None, None
        if use_cache and self.cache.mode != "off":
            embed_seconds = 0.0

            def timed_embed(text: str):
                nonlocal embed_seconds
                t0 = time.perf_counter()
                v = retriever.embedder.embed_query(text)
                embed_seconds = time.perf_counter() - t0
                return v

            t = time.perf_counter()
            hit, vector = self.cache.lookup(q, scope, embed=timed_embed if config.mode != "bm25" else None)
            if embed_seconds:
                timings["embed"] = embed_seconds * 1000
            timings["cache_lookup"] = elapsed_ms(t) - embed_seconds * 1000
        if hit is not None:
            final: FinalAnswer = hit.value
            yield "meta", {"request_id": request_id, "cache": hit.kind, "index_version": retriever.index.version,
                           "sources": [{"chunk_id": c["chunk_id"], "section": c["section"], "url": c["url"]} for c in final.citations]}
            if stream:
                yield "delta", {"text": final.answer}
            timings["total"] = elapsed_ms(started)
            trace.update(cache=hit.kind, cache_similarity=round(hit.similarity, 4), status=final.status, answer=redact(final.answer)[0],
                         citations=[c["chunk_id"] for c in final.citations], tokens={"prompt": 0, "completion": 0}, timings_ms=_rounded(timings))
            self._finish(trace, final.status)
            yield "final", self._payload(request_id, final, hit.kind, retriever.index.version, config, timings, {"prompt": 0, "completion": 0})
            return

        result = retriever.retrieve(q, config, query_vector=vector)
        timings.update({stage: seconds * 1000 for stage, seconds in result.timings.items()})
        trace.update(retrieval=[c.to_trace() for c in result.candidates[:10]], quarantined=result.quarantined,
                     selected=[c.chunk_id for c in result.selected], context_tokens=result.context_tokens)
        cache_state = "miss" if use_cache and self.cache.mode != "off" else "bypass"
        yield "meta", {"request_id": request_id, "cache": cache_state, "index_version": retriever.index.version,
                       "sources": [{"chunk_id": c.chunk_id, "section": c.header, "url": c.url} for c in result.selected]}

        llm_kwargs = {"max_tokens": settings.max_output_tokens, "temperature": settings.temperature, "json_schema": ANSWER_SCHEMA,
                      "schema_name": "grounded_answer"}
        llm_result, ttft_ms, llm_wall = None, None, None
        try:
            if not result.selected:
                llm_result = LLMResult(json.dumps({"answer": REFUSAL_TEXT, "citations": [], "answerable": False}), 0, 0, 0.0, None, "no_context")
                if stream:
                    yield "delta", {"text": REFUSAL_TEXT}
            elif stream:
                extractor = AnswerFieldStream()
                holdback = UrlHoldback(settings.allowed_answer_domains) if settings.output_url_check else None
                for piece in self.llm.stream(build_messages(q, result.selected, settings.prompt_defense), **llm_kwargs):
                    if isinstance(piece, LLMResult):
                        llm_result = piece
                        continue
                    text = extractor.feed(piece)
                    text = holdback.feed(text) if holdback is not None else text
                    if text:
                        ttft_ms = elapsed_ms(started) if ttft_ms is None else ttft_ms
                        yield "delta", {"text": text}
                if holdback is not None and (rest := holdback.flush()):
                    yield "delta", {"text": rest}
            else:
                t_llm = time.perf_counter()
                llm_result = self.llm.complete(build_messages(q, result.selected, settings.prompt_defense), **llm_kwargs)
                llm_wall = time.perf_counter() - t_llm
        except Exception as err:  # noqa: BLE001 — the LLM server is down, timed out, or returned an error
            timings["total"] = elapsed_ms(started)
            trace.update(status="llm_error", error=f"{type(err).__name__}: {str(err)[:200]}", timings_ms=_rounded(timings))
            self._finish(trace, "llm_error")
            yield "error", {"request_id": request_id, "message": f"the language model is unavailable ({type(err).__name__})"}
            return

        timings["llm"] = llm_result.seconds * 1000
        if ttft_ms is not None:
            timings["ttft"] = ttft_ms
        t = time.perf_counter()
        final = finalize(llm_result.text, result.selected, self._allowed_domains())
        timings["validate"] = elapsed_ms(t)
        if use_cache and final.status in CACHEABLE_STATUSES:
            self.cache.store(q, scope, final, vector)
        timings["total"] = elapsed_ms(started)
        if llm_result.replayed and llm_wall is not None:  # a replayed call returns instantly: count its recorded duration instead
            timings["total"] += max(0.0, llm_result.seconds - llm_wall) * 1000
        tokens = {"prompt": llm_result.prompt_tokens, "completion": llm_result.completion_tokens}
        trace.update(cache=cache_state, status=final.status, answer=redact(final.answer)[0], citations=[c["chunk_id"] for c in final.citations],
                     invalid_citations=[{"chunk_id": c["chunk_id"], "reason": c["reason"]} for c in final.invalid_citations],
                     blocked_urls=final.blocked_urls, tokens=tokens, finish_reason=llm_result.finish_reason, llm_replayed=llm_result.replayed,
                     timings_ms=_rounded(timings))
        self._finish(trace, final.status)
        yield "final", self._payload(request_id, final, cache_state, retriever.index.version, config, timings, tokens)

    def _finish(self, trace: dict, status: str) -> None:
        with self._lock:
            self.status_counts[status] += 1
        self.tracer.write(trace)

    @staticmethod
    def _payload(request_id, final: FinalAnswer, cache: str, index_version: str, config: RetrievalConfig, timings: dict, tokens: dict) -> dict:
        return {"request_id": request_id, "status": final.status, "answer": final.answer, "citations": final.citations, "cache": cache,
                "index_version": index_version, "config": config.label, "timings_ms": _rounded(timings), "tokens": tokens}


def _rounded(timings: dict[str, float]) -> dict[str, float]:
    return {stage: round(ms, 1) for stage, ms in timings.items()}
