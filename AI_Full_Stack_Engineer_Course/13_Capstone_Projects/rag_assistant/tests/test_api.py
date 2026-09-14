import json

import pytest
from fastapi.testclient import TestClient

from conftest import FIXTURE_DOCS, make_pipeline, make_settings
from rag_assistant.api import create_app
from rag_assistant.llm import StubLLM


@pytest.fixture
def client(test_settings, built_index):
    with TestClient(create_app(test_settings, pipeline_factory=make_pipeline)) as test_client:  # `with` runs the lifespan
        yield test_client


def parse_sse(text: str) -> list[tuple[str, dict]]:
    events = []
    for block in text.strip().split("\n\n"):
        fields = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((fields["event"], json.loads(fields["data"])))
    return events


def test_health_and_ready(client, built_index):
    assert client.get("/health").json() == {"status": "ok"}
    ready = client.get("/ready")
    assert ready.status_code == 200
    assert ready.json() == {"status": "ready", "index_version": built_index.index_version, "chunks": built_index.chunks_total, "llm_reachable": True}


def test_ask_returns_a_cited_answer_and_caches_it(client):
    first = client.post("/ask", json={"question": "How do I upgrade a tool?"})
    assert first.status_code == 200
    body = first.json()
    assert body["status"] == "answered" and body["cache"] == "miss"
    assert body["citations"][0]["chunk_id"].startswith("guides/tools#") and body["citations"][0]["url"].startswith("https://example.com/docs/")
    assert {"retrieval", "llm", "total"} & set(body["timings_ms"])
    again = client.post("/ask", json={"question": "how do i upgrade a tool"})
    assert again.json()["cache"] == "exact" and again.json()["answer"] == body["answer"]
    stats = client.get("/stats").json()
    assert stats["cache"]["exact_hits"] == 1 and stats["requests_by_status"] == {"answered": 2}


def test_ask_streams_server_sent_events(client):
    with client.stream("POST", "/ask", json={"question": "What is the uvx command?", "stream": True}) as response:
        assert response.status_code == 200 and response.headers["content-type"].startswith("text/event-stream")
        events = parse_sse(response.read().decode())
    names = [name for name, _ in events]
    assert names[0] == "meta" and names[-1] == "final" and names.count("delta") >= 2
    final = events[-1][1]
    assert "".join(data["text"] for name, data in events if name == "delta") == final["answer"]
    assert final["status"] == "answered" and events[0][1]["sources"]


@pytest.mark.parametrize(("body", "status"), [
    ({"question": "x" * 1001}, 413),  # longer than max_question_chars
    ({"question": "   "}, 422),  # empty after stripping
    ({"question": ""}, 422),  # schema: min_length
    ({"question": "ok", "colour": "blue"}, 422),  # schema: unknown field
    ({"question": "ok", "top_k": 50}, 422),  # schema: out of range
])
def test_invalid_requests_are_rejected(client, body, status):
    assert client.post("/ask", json=body).status_code == status


def test_limits_apply_to_streaming_requests_before_the_stream_starts(client):
    assert client.post("/ask", json={"question": "x" * 1001, "stream": True}).status_code == 413


def test_ingest_requires_the_api_key(client):
    assert client.post("/ingest").status_code == 401
    assert client.post("/ingest", headers={"X-API-Key": "wrong"}).status_code == 401
    ok = client.post("/ingest", headers={"X-API-Key": "test-key"})
    assert ok.status_code == 200 and ok.json()["chunks_embedded"] == 0 and len(ok.json()["unchanged"]) == 3


def test_ingest_swaps_to_the_new_index_and_clears_the_cache(client, test_settings):
    client.post("/ask", json={"question": "How do I uninstall a tool?"})
    new_section = "\n## Uninstalling tools\n\nRun `uv tool uninstall black` to remove a tool.\n"
    (test_settings.corpus_dir / "guides/tools.md").write_text(FIXTURE_DOCS["guides/tools.md"] + new_section)
    before = client.get("/ready").json()["index_version"]
    report = client.post("/ingest", headers={"X-API-Key": "test-key"}).json()
    assert report["updated"] == ["guides/tools"] and report["active_version"] != before
    answer = client.post("/ask", json={"question": "How do I uninstall a tool?"}).json()
    assert answer["cache"] == "miss" and answer["index_version"] == report["active_version"] and "uninstall" in answer["answer"]


def test_ingest_is_disabled_without_a_configured_key(tmp_path, corpus_dir, built_index):
    settings = make_settings(tmp_path, corpus_dir, ingest_api_key=None)
    with TestClient(create_app(settings, pipeline_factory=make_pipeline)) as client:
        assert client.post("/ingest", headers={"X-API-Key": "anything"}).status_code == 503


def test_not_ready_without_an_index(tmp_path, corpus_dir):
    settings = make_settings(tmp_path / "empty", corpus_dir)
    with TestClient(create_app(settings, pipeline_factory=make_pipeline)) as client:
        assert client.get("/health").status_code == 200
        ready = client.get("/ready")
        assert ready.status_code == 503 and "ingest" in ready.json()["reason"]
        assert client.post("/ask", json={"question": "hello"}).status_code == 503


class BrokenLLM(StubLLM):
    def complete(self, messages, **kwargs):
        raise ConnectionError("LLM server down")

    def stream(self, messages, **kwargs):
        raise ConnectionError("LLM server down")
        yield  # pragma: no cover


def test_llm_outage_returns_503_or_an_error_event(test_settings, built_index):
    with TestClient(create_app(test_settings, pipeline_factory=lambda s: make_pipeline(s, llm=BrokenLLM()))) as client:
        assert client.post("/ask", json={"question": "How do I upgrade a tool?"}).status_code == 503
        with client.stream("POST", "/ask", json={"question": "How do I upgrade a tool?", "stream": True}) as response:
            events = parse_sse(response.read().decode())
        assert [name for name, _ in events] == ["meta", "error"]
