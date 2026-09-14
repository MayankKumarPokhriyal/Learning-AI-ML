"""LLM access.

- OpenAICompatibleLLM: any OpenAI-compatible server (llama.cpp, vLLM, Ollama, LM Studio, OpenAI) with token/latency
  accounting, a concurrency limit, and an optional JSONL reply cache for reproducible re-runs.
- StubLLM: a DETERMINISTIC TEST DOUBLE for unit tests. It is not a language model: it replays scripted replies so the
  loop, budgets, and guardrails can be tested exactly and offline. Never report its outputs as model results.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

from research_agent.config import LLMSettings


@dataclass
class LLMReply:
    content: str | None
    tool_calls: list[dict] = field(default_factory=list)  # [{"id", "name", "arguments": JSON string}]
    finish_reason: str = "stop"  # "stop" | "tool_calls" | "length"
    prompt_tokens: int = 0
    completion_tokens: int = 0  # includes hidden reasoning tokens for reasoning models
    latency_s: float = 0.0
    cached: bool = False

    def as_message(self) -> dict:
        message: dict[str, Any] = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            message["tool_calls"] = [{"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": c["arguments"]}}
                                     for c in self.tool_calls]
        return message


class ChatModel(Protocol):
    name: str

    async def complete(self, messages: list[dict], *, tools: list[dict] | None = None, response_format: dict | None = None,
                       max_tokens: int = 1024) -> LLMReply: ...


class OpenAICompatibleLLM:
    def __init__(self, settings: LLMSettings | None = None):
        from openai import AsyncOpenAI  # imported here so unit tests with the stub don't need a client

        self.settings = settings or LLMSettings.from_env()
        self.name = self.settings.model
        self._client = AsyncOpenAI(base_url=self.settings.base_url, api_key=self.settings.api_key,
                                   timeout=self.settings.timeout_s, max_retries=self.settings.max_retries)
        self._semaphores: dict[int, asyncio.Semaphore] = {}
        self._cache: dict[str, dict] = {}
        self._cache_lock = threading.Lock()
        path = self.settings.cache_path
        if path is not None and path.is_file():
            for line in path.read_text().splitlines():
                record = json.loads(line)
                self._cache[record.pop("key")] = record

    def _semaphore(self) -> asyncio.Semaphore:
        loop_id = id(asyncio.get_running_loop())  # a semaphore belongs to one event loop
        if loop_id not in self._semaphores:
            self._semaphores[loop_id] = asyncio.Semaphore(self.settings.max_concurrency)
        return self._semaphores[loop_id]

    def _extra_body(self) -> dict | None:
        effort = self.settings.reasoning_effort
        return {"chat_template_kwargs": {"reasoning_effort": effort}} if effort else None

    async def complete(self, messages: list[dict], *, tools: list[dict] | None = None, response_format: dict | None = None,
                       max_tokens: int = 1024) -> LLMReply:
        request: dict[str, Any] = {"model": self.settings.model, "messages": messages, "max_tokens": max_tokens,
                                   "temperature": self.settings.temperature}
        if tools:
            request["tools"] = tools
        if response_format:
            request["response_format"] = response_format
        extra_body = self._extra_body()
        key = hashlib.sha256(json.dumps([self.settings.base_url, request, extra_body], sort_keys=True, default=str).encode()).hexdigest()
        if self.settings.cache_path is not None and key in self._cache:
            return LLMReply(**self._cache[key], cached=True)
        async with self._semaphore():
            start = time.perf_counter()
            response = await self._client.chat.completions.create(**request, extra_body=extra_body)
            latency = time.perf_counter() - start
        choice, usage = response.choices[0], response.usage
        reply = LLMReply(
            content=choice.message.content,
            tool_calls=[{"id": c.id, "name": c.function.name, "arguments": c.function.arguments or "{}"}
                        for c in (choice.message.tool_calls or [])],
            finish_reason=choice.finish_reason or "stop",
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            latency_s=round(latency, 3),
        )
        if self.settings.cache_path is not None:
            record = {k: v for k, v in asdict(reply).items() if k != "cached"}
            with self._cache_lock:
                self._cache[key] = record
                self.settings.cache_path.parent.mkdir(parents=True, exist_ok=True)
                with self.settings.cache_path.open("a") as f:
                    f.write(json.dumps({"key": key, **record}) + "\n")
        return reply

    async def list_models(self) -> list[str]:
        return [m.id for m in (await self._client.models.list()).data]

    async def aclose(self) -> None:
        await self._client.close()


ScriptStep = LLMReply | Callable[[list[dict]], LLMReply]


class StubLLM:
    """DETERMINISTIC TEST DOUBLE — not a model. Replays `script` for tool-calling steps; answers structured-output
    requests with `final_answer` (a dict or a function of the messages)."""

    name = "stub-llm (scripted test double, not a model)"

    def __init__(self, script: list[ScriptStep] | None = None, *, final_answer: dict | Callable[[list[dict]], dict] | None = None,
                 delay_s: float = 0.0):
        self.script = list(script or [])
        self.final_answer = final_answer
        self.delay_s = delay_s
        self.calls: list[dict] = []

    @staticmethod
    def tool_call(name: str, arguments: dict | str, *, call_id: str | None = None, prompt_tokens: int = 200,
                  completion_tokens: int = 30) -> LLMReply:
        args = arguments if isinstance(arguments, str) else json.dumps(arguments)
        return LLMReply(content=None, tool_calls=[{"id": call_id or f"call_{name}_{abs(hash(args)) % 10**6}", "name": name, "arguments": args}],
                        finish_reason="tool_calls", prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)

    @staticmethod
    def text(content: str, *, prompt_tokens: int = 300, completion_tokens: int = 40, finish_reason: str = "stop") -> LLMReply:
        return LLMReply(content=content, finish_reason=finish_reason, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)

    async def complete(self, messages: list[dict], *, tools: list[dict] | None = None, response_format: dict | None = None,
                       max_tokens: int = 1024) -> LLMReply:
        self.calls.append({"messages": [dict(m) for m in messages], "tools": [t["function"]["name"] for t in tools or []],
                           "structured": response_format is not None, "max_tokens": max_tokens})
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        if response_format is not None:
            answer = self.final_answer(messages) if callable(self.final_answer) else self.final_answer
            answer = answer or {"answer": "stub answer", "citations": [], "outcome": "answered"}
            return LLMReply(content=json.dumps(answer), prompt_tokens=150, completion_tokens=40)
        if not self.script:
            return self.text("The stub script is exhausted.")
        step = self.script.pop(0)
        return step(messages) if callable(step) else step
