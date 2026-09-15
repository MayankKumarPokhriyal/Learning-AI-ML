"""LLM access with retries, timeouts, token accounting, and record/replay.

- OpenAICompatibleLLM: any OpenAI-compatible server (llama.cpp, vLLM, Ollama, LM Studio, OpenAI). Transient errors are
  retried with jittered exponential backoff. Replies can be **recorded** (JSONL, keyed by a hash of the exact request)
  and **replayed offline** with replay_only=True, which is how CI runs the evaluation gate without a model.
- FakeLLM: a SCRIPTED TEST DOUBLE for unit tests. It is not a language model: never report its outputs as results.
"""

from __future__ import annotations

import asyncio
import gzip
import hashlib
import io
import json
import random
import re
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from data_analyst_agents.config import LLMSettings


class LLMFormatError(Exception):
    """The reply was not the structured JSON we asked for (e.g. reasoning used up max_tokens)."""


class ReplayMiss(Exception):
    """Replay-only mode met a request that was never recorded: a prompt, model or setting changed."""


class LLMUnavailable(Exception):
    """The server kept failing after all retries."""


@dataclass
class LLMReply:
    content: str | None
    tool_calls: list[dict] = field(default_factory=list)  # [{"id", "name", "arguments": JSON string}]
    finish_reason: str = "stop"
    prompt_tokens: int = 0
    completion_tokens: int = 0  # includes hidden reasoning tokens for reasoning models
    latency_s: float = 0.0  # measured when the call was really made (replays keep the original)
    cached: bool = False
    attempts: int = 1
    key: str = ""  # hash of the request (not stored in recordings: it IS the record's key)

    def json(self) -> dict:
        text = (self.content or "").strip()
        fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
        if fenced:
            text = fenced.group(1).strip()
        try:
            value = json.loads(text)
        except json.JSONDecodeError as err:
            raise LLMFormatError(f"reply is not valid JSON ({err.msg}; finish_reason={self.finish_reason}, {len(text)} chars)") from err
        if not isinstance(value, dict):
            raise LLMFormatError("reply JSON is not an object")
        return value

    def as_message(self) -> dict:
        message: dict[str, Any] = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            message["tool_calls"] = [{"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": c["arguments"]}}
                                     for c in self.tool_calls]
        return message


_REPLY_FIELDS = {f.name for f in fields(LLMReply)} - {"cached", "key"}


def request_key(request: dict, extra_body: dict | None, trial: int) -> str:
    """Hash of everything that changes the reply. The base URL is excluded so recordings replay against any endpoint."""
    return hashlib.sha256(json.dumps([request, extra_body, trial], sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


def load_records(path: Path) -> dict[str, dict]:
    if not path.is_file():
        return {}
    raw = path.read_bytes()
    text = gzip.decompress(raw).decode() if path.suffix == ".gz" else raw.decode()
    records = {}
    for line in text.splitlines():
        if line.strip():
            record = json.loads(line)
            records[record.pop("key")] = record
    return records


def write_records(path: Path, records: dict[str, dict]) -> Path:
    """Deterministic bytes (sorted keys, gzip mtime=0), so re-recording identical replies doesn't change the file."""
    body = "".join(json.dumps({"key": key, **records[key]}, sort_keys=True, ensure_ascii=False) + "\n" for key in sorted(records))
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".gz":
        buffer = io.BytesIO()
        with gzip.GzipFile(fileobj=buffer, mode="wb", mtime=0, filename="") as gz:
            gz.write(body.encode())
        path.write_bytes(buffer.getvalue())
    else:
        path.write_text(body)
    return path


class OpenAICompatibleLLM:
    def __init__(self, settings: LLMSettings | None = None, *, client: Any = None, sleep: Callable = asyncio.sleep, seed: int = 0):
        self.settings = settings or LLMSettings.from_env()
        self.name = self.settings.model
        if client is None and not self.settings.replay_only:
            from openai import AsyncOpenAI

            # max_retries=0: retries happen in _call below, so they are counted, traced and use our backoff policy
            client = AsyncOpenAI(base_url=self.settings.base_url, api_key=self.settings.api_key, timeout=self.settings.timeout_s, max_retries=0)
        self._client = client
        self._sleep = sleep
        self._rng = random.Random(seed)
        self._lock = threading.Lock()
        self._semaphores: dict[int, asyncio.Semaphore] = {}
        path = self.settings.cache_path
        self._cache: dict[str, dict] = load_records(path) if path is not None else {}
        self.used_keys: list[str] = []
        self.stats = {"live_calls": 0, "replayed_calls": 0, "retries": 0}

    def _semaphore(self) -> asyncio.Semaphore:
        loop_id = id(asyncio.get_running_loop())  # a semaphore belongs to one event loop
        if loop_id not in self._semaphores:
            self._semaphores[loop_id] = asyncio.Semaphore(self.settings.max_concurrency)
        return self._semaphores[loop_id]

    async def complete(self, messages: list[dict], *, agent: str, schema: dict | None = None, schema_name: str = "answer",
                       tools: list[dict] | None = None, max_tokens: int = 1024, trial: int = 0, timeout_s: float | None = None) -> LLMReply:
        request: dict[str, Any] = {"model": self.settings.model, "messages": messages, "max_tokens": max_tokens, "temperature": self.settings.temperature}
        if tools:
            request["tools"] = tools
        if schema:
            request["response_format"] = {"type": "json_schema", "json_schema": {"name": schema_name, "strict": True, "schema": schema}}
        effort = self.settings.reasoning_effort
        extra_body = {"chat_template_kwargs": {"reasoning_effort": effort}} if effort else None
        key = request_key(request, extra_body, trial)
        with self._lock:
            self.used_keys.append(key)
            record = self._cache.get(key)
        if record is not None:
            self.stats["replayed_calls"] += 1
            return LLMReply(**{k: v for k, v in record.items() if k in _REPLY_FIELDS}, cached=True, key=key)
        if self.settings.replay_only:
            raise ReplayMiss(f"no recorded reply for this {agent} request (key {key[:12]}); the prompt, model or settings changed — re-record live")
        reply = await self._call(request, extra_body, timeout_s)
        reply.key = key
        self.stats["live_calls"] += 1
        record = {"agent": agent, **{k: v for k, v in asdict(reply).items() if k in _REPLY_FIELDS}}
        with self._lock:
            self._cache[key] = record
            if self.settings.cache_path is not None and self.settings.cache_path.suffix != ".gz":
                self.settings.cache_path.parent.mkdir(parents=True, exist_ok=True)
                with self.settings.cache_path.open("a") as f:
                    f.write(json.dumps({"key": key, **record}, ensure_ascii=False) + "\n")
        return reply

    async def _call(self, request: dict, extra_body: dict | None, timeout_s: float | None) -> LLMReply:
        import openai

        transient = (openai.APIConnectionError, openai.RateLimitError, openai.InternalServerError, TimeoutError)  # APITimeoutError ⊂ APIConnectionError
        limit = max(1.0, min(timeout_s or self.settings.timeout_s, self.settings.timeout_s))
        for attempt in range(1, self.settings.max_attempts + 1):
            try:
                async with self._semaphore():
                    start = time.perf_counter()
                    response = await asyncio.wait_for(self._client.chat.completions.create(**request, extra_body=extra_body), timeout=limit)
                    latency = time.perf_counter() - start
                break
            except transient as err:
                if attempt == self.settings.max_attempts:
                    raise LLMUnavailable(f"{type(err).__name__} after {attempt} attempts") from err
                self.stats["retries"] += 1
                await self._sleep(min(20.0, 2.0 ** (attempt - 1)) * (0.5 + self._rng.random()))  # full jitter around 1, 2, 4 … s
        choice, usage = response.choices[0], response.usage
        return LLMReply(
            content=choice.message.content,
            tool_calls=[{"id": c.id, "name": c.function.name, "arguments": c.function.arguments or "{}"} for c in (choice.message.tool_calls or [])],
            finish_reason=choice.finish_reason or "stop",
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            latency_s=round(latency, 3),
            attempts=attempt,
        )

    def export_recording(self, path: Path, keys: list[str]) -> dict:
        """Write the replies for `keys` (e.g. every request one evaluation made) as a small, deterministic replay file."""
        wanted = {k: self._cache[k] for k in dict.fromkeys(keys) if k in self._cache}
        write_records(path, wanted)
        return {"path": path, "records": len(wanted), "bytes": path.stat().st_size}

    async def list_models(self) -> list[str]:
        if self._client is None:
            return [self.settings.model + " (replay only)"]
        return [m.id for m in (await self._client.models.list()).data]


ScriptItem = dict | str | LLMReply | BaseException


class FakeLLM:
    """SCRIPTED TEST DOUBLE — not a model. `script` maps an agent name to a list of replies consumed in order, or to a
    function(messages) → reply. A dict becomes JSON content, a str becomes text, an LLMReply is returned as-is, and an
    exception is raised."""

    name = "fake-llm (scripted test double, not a model)"

    def __init__(self, script: dict[str, list[ScriptItem] | Callable[[list[dict]], ScriptItem]] | None = None, *,
                 prompt_tokens: int = 100, completion_tokens: int = 20, delay_s: float = 0.0):
        self.script = {agent: (spec if callable(spec) else list(spec)) for agent, spec in (script or {}).items()}
        self.prompt_tokens, self.completion_tokens, self.delay_s = prompt_tokens, completion_tokens, delay_s
        self.calls: list[dict] = []
        self.used_keys: list[str] = []
        self.stats = {"live_calls": 0, "replayed_calls": 0, "retries": 0}

    async def complete(self, messages: list[dict], *, agent: str, schema: dict | None = None, schema_name: str = "answer",
                       tools: list[dict] | None = None, max_tokens: int = 1024, trial: int = 0, timeout_s: float | None = None) -> LLMReply:
        self.calls.append({"agent": agent, "messages": [dict(m) for m in messages], "structured": schema is not None,
                           "tools": [t["function"]["name"] for t in tools or []], "trial": trial})
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        spec = self.script.get(agent)
        if spec is None or (not callable(spec) and not spec):
            raise AssertionError(f"FakeLLM: no scripted reply left for agent {agent!r}")
        item = spec(messages) if callable(spec) else spec.pop(0)
        self.stats["live_calls"] += 1
        if isinstance(item, BaseException):
            raise item
        if isinstance(item, LLMReply):
            return item
        content = json.dumps(item) if isinstance(item, dict) else str(item)
        return LLMReply(content=content, prompt_tokens=self.prompt_tokens, completion_tokens=self.completion_tokens, latency_s=0.01)

    async def list_models(self) -> list[str]:
        return [self.name]


def tool_call_reply(name: str, arguments: dict, call_id: str = "call_1") -> LLMReply:
    """Helper for tests: a reply that requests one tool call."""
    return LLMReply(content=None, tool_calls=[{"id": call_id, "name": name, "arguments": json.dumps(arguments)}], finish_reason="tool_calls",
                    prompt_tokens=100, completion_tokens=20)
