"""The HTTP service with FastAPI's TestClient and FakeLLM (a scripted test double, not a model)."""

import json
import time

import pytest
from fastapi.testclient import TestClient

from conftest import OK, PAID_SQL, PLAN, make_catalogs
from data_analyst_agents.api import create_app
from data_analyst_agents.llm import FakeLLM

QUESTION = "What is the total amount of paid orders?"


def team_llm(delay_s: float = 0.0) -> FakeLLM:
    return FakeLLM({"planner": [PLAN] * 3, "sql_writer": [{"sql": PAID_SQL}] * 3, "verifier": [OK] * 3,
                    "report_writer": [{"finding": "Paid orders total 540.", "caveats": []}] * 3}, delay_s=delay_s)


def sse_events(text: str) -> list[dict]:
    events = []
    for block in text.strip().split("\n\n"):
        fields = dict(line.split(": ", 1) for line in block.splitlines() if ": " in line and not line.startswith(":"))
        if "event" in fields:
            events.append({"event": fields["event"], "id": fields.get("id"), "data": json.loads(fields["data"])})
    return events


def wait_for(client, job_id, statuses, timeout=20):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/jobs/{job_id}").json()
        if job["status"] in statuses:
            return job
        time.sleep(0.05)
    raise AssertionError(f"job still {job['status']}, expected one of {statuses}")


@pytest.fixture
def make_client(catalogs):
    def factory(llm, cats=None):
        return TestClient(create_app(llm=llm, catalogs=cats or catalogs, workers=1))

    return factory


def test_health_ready_and_input_validation(make_client):
    with make_client(team_llm()) as client:
        assert client.get("/health").json() == {"status": "ok"}
        ready = client.get("/ready").json()
        assert ready["status"] == "ready" and ready["checks"]["databases"] == "ok (shop)"
        assert client.post("/jobs", json={"question": "hi"}).status_code == 422
        assert client.post("/jobs", json={"question": QUESTION, "sql": "DROP TABLE orders"}).status_code == 422
        assert client.get("/jobs/0123456789abcdef").status_code == 404


def test_a_job_streams_progress_and_finishes_with_a_report(make_client):
    with make_client(team_llm()) as client:
        created = client.post("/jobs", json={"question": QUESTION})
        assert created.status_code == 202
        job_id = created.json()["id"]
        with client.stream("GET", f"/jobs/{job_id}/events") as response:
            assert response.headers["content-type"].startswith("text/event-stream")
            events = sse_events(response.read().decode())
        kinds = [e["event"] for e in events]
        assert kinds[:2] == ["queued", "started"] and "plan" in kinds and "report_ready" in kinds and kinds[-1] == "job_status"
        job = client.get(f"/jobs/{job_id}").json()
        assert job["status"] == "completed" and job["result"]["preview"] == [[540.0]] and job["result"]["usage"]["llm_calls"] == 4
        assert f"```sql\n{PAID_SQL}\n```" in client.get(f"/jobs/{job_id}/report").text
        with client.stream("GET", f"/jobs/{job_id}/events", headers={"Last-Event-ID": events[-2]["id"]}) as again:  # a reconnecting client
            assert [e["event"] for e in sse_events(again.read().decode())] == ["job_status"]


def test_idempotency_keys(make_client):
    with make_client(team_llm()) as client:
        first = client.post("/jobs", json={"question": QUESTION}, headers={"Idempotency-Key": "abc-123"})
        again = client.post("/jobs", json={"question": QUESTION}, headers={"Idempotency-Key": "abc-123"})
        assert first.status_code == 202 and again.status_code == 200 and again.json()["id"] == first.json()["id"]
        assert client.post("/jobs", json={"question": "Something else entirely?"}, headers={"Idempotency-Key": "abc-123"}).status_code == 409
        wait_for(client, first.json()["id"], {"completed"})


def test_an_export_needs_an_authorised_reviewer_and_happens_once(make_client, monkeypatch):
    monkeypatch.setenv("REVIEWER_TOKEN", "letmein")
    with make_client(team_llm()) as client:
        job_id = client.post("/jobs", json={"question": "Export the total amount of paid orders to CSV"}).json()["id"]
        job = wait_for(client, job_id, {"awaiting_approval"})
        assert job["pending_approval"]["kind"] == "export" and client.get(f"/jobs/{job_id}/export.csv").status_code == 404
        decision = {"approval_key": "export", "decision": "approve", "reviewer": "qa"}
        assert client.post(f"/jobs/{job_id}/approvals", json=decision).status_code == 401
        approved = client.post(f"/jobs/{job_id}/approvals", json=decision, headers={"X-Reviewer-Token": "letmein"})
        assert approved.status_code == 200 and approved.json()["status"] == "completed"
        assert client.get(f"/jobs/{job_id}/export.csv").text.splitlines() == ["SUM(amount)", "540.0"]
        assert client.post(f"/jobs/{job_id}/approvals", json=decision, headers={"X-Reviewer-Token": "letmein"}).status_code == 409


def test_an_approved_expensive_query_resumes_the_job(make_client, shop_path, descriptions):
    llm = team_llm()
    with make_client(llm, make_catalogs(shop_path, descriptions, expensive_scan_rows=5)) as client:
        job_id = client.post("/jobs", json={"question": QUESTION}).json()["id"]
        pending = wait_for(client, job_id, {"awaiting_approval"})["pending_approval"]
        assert pending["kind"] == "expensive_query" and pending["sql"] == PAID_SQL
        response = client.post(f"/jobs/{job_id}/approvals", json={"approval_key": pending["key"], "decision": "approve", "reviewer": "qa"})
        assert response.status_code == 200
        job = wait_for(client, job_id, {"completed"})
        assert job["result"]["sql"] == PAID_SQL and [c["agent"] for c in llm.calls].count("sql_writer") == 1


def test_cancel_a_running_job(make_client):
    with make_client(team_llm(delay_s=3.0)) as client:
        job_id = client.post("/jobs", json={"question": QUESTION}).json()["id"]
        wait_for(client, job_id, {"running"})
        cancelled = client.post(f"/jobs/{job_id}/cancel")
        assert cancelled.status_code == 202 and cancelled.json()["status"] == "cancelled"
        assert client.post(f"/jobs/{job_id}/cancel").status_code == 409
