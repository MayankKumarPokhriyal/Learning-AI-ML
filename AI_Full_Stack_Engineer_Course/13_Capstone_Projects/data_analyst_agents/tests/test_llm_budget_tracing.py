"""Retries, record/replay, budgets and OpenTelemetry spans — all offline."""

import asyncio
from types import SimpleNamespace as NS

import httpx
import openai
import pytest

from conftest import OK, PAID_SQL, PLAN
from data_analyst_agents.budget import BudgetExceeded, BudgetTracker
from data_analyst_agents.config import JobBudget, LLMSettings
from data_analyst_agents.llm import FakeLLM, LLMFormatError, LLMReply, LLMUnavailable, OpenAICompatibleLLM, ReplayMiss, load_records, request_key, write_records
from data_analyst_agents.pipeline import AnalystTeam, TeamSettings
from data_analyst_agents.tracing import Tracing, agent_usage, load_spans

REQUEST = httpx.Request("POST", "http://llm.test/v1/chat/completions")
MESSAGES = [{"role": "user", "content": "q"}]


def completion(content='{"ok": true}'):
    return NS(choices=[NS(message=NS(content=content, tool_calls=None), finish_reason="stop")], usage=NS(prompt_tokens=11, completion_tokens=7))


def make_llm(outcomes, **settings):
    calls = []

    async def create(**kwargs):
        calls.append(kwargs)
        item = outcomes.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    sleeps = []

    async def sleep(seconds):
        sleeps.append(seconds)

    llm = OpenAICompatibleLLM(LLMSettings(**settings), client=NS(chat=NS(completions=NS(create=create))), sleep=sleep)
    return llm, calls, sleeps


def test_transient_errors_are_retried_with_backoff():
    llm, calls, sleeps = make_llm([openai.APITimeoutError(request=REQUEST), openai.APIConnectionError(request=REQUEST), completion()])
    reply = asyncio.run(llm.complete(MESSAGES, agent="t"))
    assert reply.json() == {"ok": True} and reply.attempts == 3 and len(calls) == 3 and len(sleeps) == 2 and llm.stats["retries"] == 2
    assert calls[0]["extra_body"] == {"chat_template_kwargs": {"reasoning_effort": "low"}}


def test_gives_up_after_the_last_attempt_and_never_retries_bad_requests():
    llm, calls, _ = make_llm([openai.APIConnectionError(request=REQUEST)] * 3)
    with pytest.raises(LLMUnavailable):
        asyncio.run(llm.complete(MESSAGES, agent="t"))
    assert len(calls) == 3
    bad = openai.BadRequestError("bad request", response=httpx.Response(400, request=REQUEST), body=None)
    llm, calls, _ = make_llm([bad, completion()])
    with pytest.raises(openai.BadRequestError):
        asyncio.run(llm.complete(MESSAGES, agent="t"))
    assert len(calls) == 1


def test_record_then_replay_offline(tmp_path):
    cache = tmp_path / "cache.jsonl"
    llm, _, _ = make_llm([completion('{"n": 1}')], cache_path=cache)
    first = asyncio.run(llm.complete(MESSAGES, agent="t"))
    replay = OpenAICompatibleLLM(LLMSettings(cache_path=cache, replay_only=True))
    again = asyncio.run(replay.complete(MESSAGES, agent="t"))
    assert again.cached and again.json() == first.json() and again.latency_s == first.latency_s
    with pytest.raises(ReplayMiss):
        asyncio.run(replay.complete(MESSAGES, agent="t", trial=1))  # another trial is another request
    with pytest.raises(ReplayMiss):
        asyncio.run(replay.complete([{"role": "user", "content": "a changed prompt"}], agent="t"))


def test_recordings_are_byte_for_byte_deterministic(tmp_path):
    records = {"b": {"content": "2"}, "a": {"content": "1"}}
    one = write_records(tmp_path / "one.jsonl.gz", records).read_bytes()
    two = write_records(tmp_path / "two.jsonl.gz", dict(reversed(records.items()))).read_bytes()
    assert one == two and load_records(tmp_path / "one.jsonl.gz") == records


def test_reply_parsing_and_request_keys():
    assert LLMReply(content='```json\n{"a": 1}\n```').json() == {"a": 1}
    with pytest.raises(LLMFormatError):
        LLMReply(content="", finish_reason="length").json()
    request = {"model": "m", "messages": MESSAGES}
    assert request_key(request, None, 0) == request_key(dict(request), None, 0) != request_key(request, None, 1)


def test_budget_tracker_limits():
    now = [0.0]
    tracker = BudgetTracker(JobBudget(max_llm_calls=1, max_wall_s=10, max_sql_runs=1), clock=lambda: now[0])
    tracker.before_llm()
    tracker.after_llm(LLMReply(content="x", prompt_tokens=5, completion_tokens=5, latency_s=1.5))
    with pytest.raises(BudgetExceeded, match="max_llm_calls"):
        tracker.before_llm()
    tracker.before_sql()
    with pytest.raises(BudgetExceeded, match="max_sql_runs"):
        tracker.before_sql()
    now[0] = 11
    with pytest.raises(BudgetExceeded, match="max_wall_s"):
        tracker.check_time()
    assert tracker.usage()["total_tokens"] == 10 and tracker.usage()["llm_latency_s"] == 1.5


def test_spans_follow_the_genai_conventions_and_add_up(catalogs, tmp_path):
    tracing = Tracing(tmp_path / "spans.jsonl")
    llm = FakeLLM({"planner": [PLAN], "sql_writer": [{"sql": PAID_SQL}], "verifier": [OK]})
    out = asyncio.run(AnalystTeam(llm, catalogs, settings=TeamSettings(profile="answer"), tracer=tracing.tracer).run("What is the total amount of paid orders?"))
    tracing.shutdown()
    spans = load_spans(tmp_path / "spans.jsonl")
    names = {s["name"] for s in spans}
    assert {"invoke_workflow data-analyst-team", "invoke_agent planner", "invoke_agent sql_writer", "invoke_agent verifier", "execute_tool run_sql"} <= names
    root = next(s for s in spans if s["name"].startswith("invoke_workflow"))
    assert root["parent_id"] is None and all(s["trace_id"] == root["trace_id"] for s in spans)
    usage = {row["agent"]: row for row in agent_usage(spans)}
    assert set(usage) == {"planner", "sql_writer", "verifier"} and sum(r["input_tokens"] for r in usage.values()) == out.usage["prompt_tokens"]
