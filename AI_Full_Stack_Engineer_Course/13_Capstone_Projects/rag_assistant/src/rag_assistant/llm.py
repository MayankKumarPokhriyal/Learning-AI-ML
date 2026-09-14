"""LLM clients.

`OpenAICompatibleLLM` talks to any OpenAI-compatible server (llama.cpp, vLLM, Ollama, LM Studio, OpenAI). With `record_dir`
set it records every response and replays identical requests later (a development convenience, like a VCR cassette; replayed
results are marked `replayed=True`).

`StubLLM` is a deterministic TEST DOUBLE for unit and API tests. It is never used for evaluation results.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx


@dataclass
class LLMResult:
    text: str
    prompt_tokens: int | None
    completion_tokens: int | None
    seconds: float  # wall time of the call (the recorded time when replayed)
    ttft: float | None  # seconds to the first content token (streaming only)
    finish_reason: str | None
    replayed: bool = False


class OpenAICompatibleLLM:
    def __init__(self, base_url: str, model: str, api_key: str = "local", *, timeout: float = 180.0, max_retries: int = 2,
                 reasoning_effort: str | None = "low", record_dir: Path | None = None, replay_realtime: bool = False):
        from openai import OpenAI

        self.base_url, self.model, self.api_key = base_url.rstrip("/"), model, api_key
        self.reasoning_effort = None if reasoning_effort in (None, "", "none") else reasoning_effort
        self.client = OpenAI(base_url=self.base_url, api_key=api_key, timeout=timeout, max_retries=max_retries)
        self.record_dir = Path(record_dir) if record_dir else None
        self.replay_realtime = replay_realtime  # wait the recorded time when replaying, so end-to-end latencies stay realistic
        self._occurrence: dict[str, int] = {}
        self._lock = threading.Lock()
        self.stats = {"live_calls": 0, "replayed_calls": 0}

    @property
    def name(self) -> str:
        return self.model

    def ping(self, timeout: float = 2.0) -> bool:
        try:
            response = httpx.get(f"{self.base_url}/models", timeout=timeout, headers={"Authorization": f"Bearer {self.api_key}"})
            return response.status_code == 200
        except httpx.HTTPError:
            return False

    def _params(self, messages, max_tokens, temperature, json_schema, schema_name, stream) -> dict:
        params = {"model": self.model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature}
        if json_schema is not None:
            params["response_format"] = {"type": "json_schema", "json_schema": {"name": schema_name, "strict": True, "schema": json_schema}}
        if self.reasoning_effort:  # gpt-oss chat template option; reasoning tokens count toward max_tokens
            params["extra_body"] = {"chat_template_kwargs": {"reasoning_effort": self.reasoning_effort}}
        if stream:
            params.update(stream=True, stream_options={"include_usage": True})
        return params

    def _record_path(self, kind: str, params: dict) -> Path | None:
        if self.record_dir is None:
            return None
        digest = hashlib.sha256(json.dumps({"kind": kind, **params}, sort_keys=True, default=str).encode()).hexdigest()[:32]
        with self._lock:  # the n-th identical request in this process maps to the n-th recording
            n = self._occurrence[digest] = self._occurrence.get(digest, 0) + 1
        return self.record_dir / f"{digest}-{n}.json"

    def _count(self, key: str) -> None:
        with self._lock:
            self.stats[key] += 1

    def complete(self, messages: list[dict], *, max_tokens: int = 1024, temperature: float = 0.0, json_schema: dict | None = None,
                 schema_name: str = "response") -> LLMResult:
        params = self._params(messages, max_tokens, temperature, json_schema, schema_name, stream=False)
        path = self._record_path("complete", params)
        if path is not None and path.exists():
            self._count("replayed_calls")
            result = LLMResult(**json.loads(path.read_text())["result"], replayed=True)
            if self.replay_realtime:
                time.sleep(result.seconds)
            return result
        start = time.perf_counter()
        response = self.client.chat.completions.create(**params)
        choice, usage = response.choices[0], response.usage
        result = LLMResult(choice.message.content or "", usage.prompt_tokens if usage else None, usage.completion_tokens if usage else None,
                           time.perf_counter() - start, None, choice.finish_reason)
        self._count("live_calls")
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"result": {k: v for k, v in asdict(result).items() if k != "replayed"}}))
        return result

    def stream(self, messages: list[dict], *, max_tokens: int = 1024, temperature: float = 0.0, json_schema: dict | None = None,
               schema_name: str = "response") -> Iterator[str | LLMResult]:
        """Yields content deltas (str) as they arrive, then one final LLMResult."""
        params = self._params(messages, max_tokens, temperature, json_schema, schema_name, stream=True)
        path = self._record_path("stream", params)
        if path is not None and path.exists():
            saved = json.loads(path.read_text())
            self._count("replayed_calls")
            start = time.perf_counter()
            for offset, piece in saved["deltas"]:
                delay = offset - (time.perf_counter() - start)
                if delay > 0:
                    time.sleep(delay)  # replay at the recorded pace, so streaming behaves like the original call
                yield piece
            yield LLMResult(**saved["result"], replayed=True)
            return
        start, deltas, parts, usage, finish, ttft = time.perf_counter(), [], [], None, None, None
        for event in self.client.chat.completions.create(**params):
            if event.usage:
                usage = event.usage
            if not event.choices:
                continue
            choice = event.choices[0]
            finish = choice.finish_reason or finish
            piece = choice.delta.content
            if piece:  # hidden reasoning arrives in delta.reasoning_content and is not shown to users
                elapsed = time.perf_counter() - start
                ttft = elapsed if ttft is None else ttft
                deltas.append([round(elapsed, 4), piece])
                parts.append(piece)
                yield piece
        result = LLMResult("".join(parts), usage.prompt_tokens if usage else None, usage.completion_tokens if usage else None,
                           time.perf_counter() - start, ttft, finish)
        self._count("live_calls")
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"deltas": deltas, "result": {k: v for k, v in asdict(result).items() if k != "replayed"}}))
        yield result


PASSAGE = re.compile(r'<passage id="([^"]+)"[^>]*>\n(.*?)\n</passage>', re.S)
PLAIN_PASSAGE = re.compile(r"^\[([^\]\n]+)\] [^\n]*\n(.*?)(?=\n\n\[[^\]\n]+\] |\n\nQuestion: |\Z)", re.S | re.M)
STOPWORDS = frozenset("what which when where does how the and for with that this from into your uv use using can should about there".split())


class StubLLM:
    """TEST DOUBLE — a deterministic fake LLM for unit and API tests. It answers with the first sentence of the passage that
    shares the most words with the question (quoting it as the citation), or refuses when nothing overlaps."""

    name = "test-stub"

    def __init__(self, responder=None, chunk_size: int = 7):
        self.responder, self.chunk_size = responder or self.default_responder, chunk_size
        self.calls: list[list[dict]] = []
        self.stats = {"live_calls": 0, "replayed_calls": 0}

    @staticmethod
    def default_responder(messages: list[dict]) -> str:
        user = messages[-1]["content"]
        question = user.rsplit("Question:", 1)[-1]
        words = {w for w in re.findall(r"[a-z0-9_]+", question.lower()) if len(w) > 3 and w not in STOPWORDS}
        passages = PASSAGE.findall(user) or PLAIN_PASSAGE.findall(user)
        best, best_overlap = None, 0
        for chunk_id, text in passages:
            overlap = len(words & set(re.findall(r"[a-z0-9_]+", text.lower())))
            if overlap > best_overlap:
                best, best_overlap = (chunk_id, text), overlap
        if best is None:
            return json.dumps({"answer": "I can't find the answer to that in the uv documentation.", "citations": [], "answerable": False})
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", best[1]) if len(s.strip()) >= 12 and not s.lstrip().startswith("#")]
        sentence = max(sentences, key=lambda s: len(words & set(re.findall(r"[a-z0-9_]+", s.lower())))) if sentences else best[1].strip()
        return json.dumps({"answer": sentence, "citations": [{"chunk_id": best[0], "quote": sentence}], "answerable": True})

    def ping(self, timeout: float = 2.0) -> bool:
        return True

    def complete(self, messages, **kwargs) -> LLMResult:
        self.calls.append(messages)
        text = self.responder(messages)
        return LLMResult(text, sum(len(m["content"]) // 4 for m in messages), len(text) // 4, 0.0, None, "stop")

    def stream(self, messages, **kwargs) -> Iterator[str | LLMResult]:
        result = self.complete(messages)
        for i in range(0, len(result.text), self.chunk_size):
            yield result.text[i:i + self.chunk_size]
        result.ttft = 0.0
        yield result
