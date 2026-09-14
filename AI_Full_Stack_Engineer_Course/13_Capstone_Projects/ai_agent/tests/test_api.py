"""HTTP API tests with FastAPI's TestClient, a StubLLM test double, and an in-memory MCP server."""

import json

import pytest
from fastapi.testclient import TestClient
from mcp import StdioServerParameters

from conftest import APPROVAL_KEY
from research_agent.api import create_app
from research_agent.llm import StubLLM
from research_agent.mcp_server import build_server

SAVE = StubLLM.tool_call("save_report", {"filename": "react.md", "title": "ReAct", "content": "ReAct summary (2210.03629).", "citations": ["2210.03629"]})


@pytest.fixture
def make_client(tool_ctx, index, monkeypatch):
    monkeypatch.setenv("AGENT_CORPUS_PATH", str(index.path))

    def factory(llm, tool_server=None):
        return TestClient(create_app(llm=llm, tool_server=tool_server or build_server(tool_ctx), approval_key=APPROVAL_KEY))

    return factory


def sse_events(text: str) -> list[dict]:
    events = []
    for block in text.strip().split("\n\n"):
        fields = dict(line.split(": ", 1) for line in block.splitlines() if ": " in line)
        if "event" in fields:
            events.append({"event": fields["event"], "data": json.loads(fields["data"])})
    return events


def test_health_and_ready(make_client):
    with make_client(StubLLM()) as client:
        assert client.get("/health").json() == {"status": "ok"}
        ready = client.get("/ready")
        assert ready.status_code == 200
        assert ready.json()["checks"] == {"tools": "ok", "corpus": "ok (2 papers)", "llm": "ok (test double)"}


def test_not_ready_without_a_corpus(make_client, monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_CORPUS_PATH", str(tmp_path / "empty.sqlite"))
    with make_client(StubLLM()) as client:
        assert client.get("/health").status_code == 200
        ready = client.get("/ready")
        assert ready.status_code == 503 and ready.json()["checks"]["corpus"].startswith("empty")


def test_service_without_tools_is_alive_but_not_ready(make_client):
    with make_client(StubLLM(), tool_server=StdioServerParameters(command="/nonexistent/python", args=[])) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/ready").status_code == 503
        assert client.post("/chat", json={"message": "hi"}).status_code == 503


def test_chat_returns_a_structured_answer_usage_trace_and_history(make_client):
    llm = StubLLM([StubLLM.tool_call("get_paper", {"arxiv_id": "2210.03629"}), StubLLM.text("ReAct interleaves reasoning and acting.")],
                  final_answer={"answer": "ReAct interleaves reasoning and acting.", "citations": ["2210.03629"], "outcome": "answered"})
    with make_client(llm) as client:
        body = client.post("/chat", json={"message": "What is ReAct?"}).json()
        assert body["status"] == "completed" and body["answer"]["citations"] == ["2210.03629"]
        assert body["usage"]["steps"] == 2 and body["usage"]["tool_calls"] == 1
        trace = client.get(f"/runs/{body['run_id']}/trace").json()["events"]
        assert trace[0]["kind"] == "run_started" and trace[-1]["kind"] == "final"
        history = client.get(f"/conversations/{body['conversation_id']}").json()
        assert [run["question"] for run in history["runs"]] == ["What is ReAct?"]
        assert client.get("/conversations/doesnotexist").status_code == 404


@pytest.mark.parametrize("payload", [{"message": ""}, {"message": "x" * 2001}, {"message": "hi", "extra": 1},
                                     {"message": "hi", "conversation_id": "../../etc"}, {}])
def test_chat_input_validation(make_client, payload):
    with make_client(StubLLM()) as client:
        assert client.post("/chat", json=payload).status_code == 422


def test_stream_sends_server_sent_events(make_client):
    llm = StubLLM([StubLLM.tool_call("search_papers", {"query": "reasoning acting"}), StubLLM.text("ReAct.")])
    with make_client(llm) as client, client.stream("POST", "/chat/stream", json={"message": "What is ReAct?"}) as response:
        assert response.headers["content-type"].startswith("text/event-stream")
        events = sse_events(response.read().decode())
    names = [e["event"] for e in events]
    assert names[0] == "run_started" and names[-1] == "final" and "llm_step" in names and "tool_call" in names
    assert events[-1]["data"]["status"] == "completed"


def test_approval_endpoints(make_client, tool_ctx):
    with make_client(StubLLM([SAVE, StubLLM.text("Waiting for approval.")])) as client:
        chat = client.post("/chat", json={"message": "Please save a report about ReAct"}).json()
        pending = client.get("/approvals", params={"status": "pending"}).json()
        assert chat["answer"]["outcome"] == "awaiting_approval" and [a["id"] for a in pending] == chat["answer"]["approval_ids"]
        assert not (tool_ctx.reports_dir / "react.md").exists()
        approval_id = pending[0]["id"]
        decided = client.post(f"/approvals/{approval_id}", json={"decision": "approve", "reviewer": "alice"})
        assert decided.status_code == 200 and decided.json()["result"]["executed"] is True
        assert (tool_ctx.reports_dir / "react.md").exists()
        assert client.post(f"/approvals/{approval_id}", json={"decision": "deny", "reviewer": "alice"}).status_code == 409
        assert client.post("/approvals/0123456789abcdef", json={"decision": "deny", "reviewer": "x"}).status_code == 404
        assert client.post("/approvals/not-an-id", json={"decision": "deny", "reviewer": "x"}).status_code == 422


def test_reviewer_token_is_required_when_configured(make_client, monkeypatch):
    monkeypatch.setenv("REVIEWER_TOKEN", "s3cret-reviewer-token")
    with make_client(StubLLM([SAVE, StubLLM.text("Waiting.")])) as client:
        client.post("/chat", json={"message": "Save a report about ReAct"})
        approval_id = client.get("/approvals").json()[0]["id"]
        decision = {"decision": "deny", "reviewer": "alice"}
        assert client.post(f"/approvals/{approval_id}", json=decision).status_code == 401
        assert client.post(f"/approvals/{approval_id}", json=decision, headers={"X-Reviewer-Token": "wrong"}).status_code == 401
        assert client.post(f"/approvals/{approval_id}", json=decision, headers={"X-Reviewer-Token": "s3cret-reviewer-token"}).status_code == 200
