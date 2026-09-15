"""Job queue for analysis requests.

SQLite holds jobs, their progress events, approval decisions and exports, so state survives a restart. asyncio workers
take jobs from a bounded queue (backpressure → HTTP 429), support cancellation, idempotency keys (a retried POST never
starts a second job), human approvals (atomic: one decision per approval), resumable expensive queries (checkpoint),
and idempotent exports (a retried approval never writes twice).
"""

from __future__ import annotations

import asyncio
import csv
import hashlib
import json
import os
import sqlite3
import threading
import time
import uuid
from pathlib import Path

from data_analyst_agents.pipeline import AnalystTeam

TERMINAL = frozenset({"completed", "failed", "cancelled", "budget_exceeded", "no_answer", "denied"})
JSON_FIELDS = ("request", "outcome", "checkpoint", "pending")

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, idempotency_key TEXT UNIQUE, request_hash TEXT NOT NULL, request TEXT NOT NULL,
    status TEXT NOT NULL, created_at REAL NOT NULL, started_at REAL, finished_at REAL, outcome TEXT, checkpoint TEXT, pending TEXT, error TEXT);
CREATE TABLE IF NOT EXISTS events (job_id TEXT NOT NULL, seq INTEGER NOT NULL, type TEXT NOT NULL, data TEXT NOT NULL, ts REAL NOT NULL,
    PRIMARY KEY (job_id, seq));
CREATE TABLE IF NOT EXISTS decisions (job_id TEXT NOT NULL, approval_key TEXT NOT NULL, decision TEXT NOT NULL, reviewer TEXT NOT NULL,
    decided_at REAL NOT NULL, PRIMARY KEY (job_id, approval_key));
CREATE TABLE IF NOT EXISTS exports (job_id TEXT PRIMARY KEY, path TEXT NOT NULL, rows INTEGER NOT NULL, sha256 TEXT NOT NULL, created_at REAL NOT NULL);
"""


class Conflict(Exception):
    pass


class QueueFull(Exception):
    pass


class NotFound(KeyError):
    pass


class JobStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._con = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self._con.row_factory = sqlite3.Row
        self._con.execute("PRAGMA journal_mode = WAL")
        self._con.executescript(SCHEMA)
        self._lock = threading.RLock()

    def close(self) -> None:
        self._con.close()

    def _rows(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._con.execute(sql, params).fetchall()

    def _write(self, sql: str, params: tuple = ()) -> int:
        with self._lock:
            return self._con.execute(sql, params).rowcount

    @staticmethod
    def _job(row: sqlite3.Row) -> dict:
        job = dict(row)
        for name in JSON_FIELDS:
            job[name] = json.loads(job[name]) if job[name] else None
        return job

    def create(self, request: dict, idempotency_key: str | None = None) -> tuple[dict, bool]:
        digest = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
        with self._lock:
            if idempotency_key:
                existing = self._rows("SELECT * FROM jobs WHERE idempotency_key = ?", (idempotency_key,))
                if existing:
                    if existing[0]["request_hash"] != digest:
                        raise Conflict("this Idempotency-Key was already used with a different request")
                    return self._job(existing[0]), False
            job_id = uuid.uuid4().hex[:16]
            self._write("INSERT INTO jobs (id, idempotency_key, request_hash, request, status, created_at) VALUES (?, ?, ?, ?, 'queued', ?)",
                        (job_id, idempotency_key, digest, json.dumps(request), time.time()))
        return self.get(job_id), True

    def key_exists(self, idempotency_key: str | None) -> bool:
        return bool(idempotency_key) and bool(self._rows("SELECT 1 FROM jobs WHERE idempotency_key = ?", (idempotency_key,)))

    def get(self, job_id: str) -> dict:
        rows = self._rows("SELECT * FROM jobs WHERE id = ?", (job_id,))
        if not rows:
            raise NotFound(job_id)
        return self._job(rows[0])

    def update(self, job_id: str, **fields) -> None:
        values = [json.dumps(v, default=str) if k in JSON_FIELDS and v is not None else v for k, v in fields.items()]
        self._write(f"UPDATE jobs SET {', '.join(f'{k} = ?' for k in fields)} WHERE id = ?", (*values, job_id))

    def transition(self, job_id: str, from_statuses: tuple[str, ...], to_status: str, **fields) -> bool:
        """Atomic compare-and-set on the status: exactly one concurrent caller wins."""
        values = [json.dumps(v, default=str) if k in JSON_FIELDS and v is not None else v for k, v in fields.items()]
        assignments = ", ".join(["status = ?", *(f"{k} = ?" for k in fields)])
        marks = ", ".join("?" for _ in from_statuses)
        return self._write(f"UPDATE jobs SET {assignments} WHERE id = ? AND status IN ({marks})", (to_status, *values, job_id, *from_statuses)) == 1

    def add_event(self, job_id: str, kind: str, data: dict) -> dict:
        with self._lock:
            seq = self._rows("SELECT COALESCE(MAX(seq), 0) + 1 FROM events WHERE job_id = ?", (job_id,))[0][0]
            ts = time.time()
            self._write("INSERT INTO events VALUES (?, ?, ?, ?, ?)", (job_id, seq, kind, json.dumps(data, default=str, ensure_ascii=False), ts))
        return {"seq": seq, "type": kind, "data": data, "ts": ts}

    def events(self, job_id: str, after: int = 0) -> list[dict]:
        return [{"seq": r["seq"], "type": r["type"], "data": json.loads(r["data"]), "ts": r["ts"]}
                for r in self._rows("SELECT * FROM events WHERE job_id = ? AND seq > ? ORDER BY seq", (job_id, after))]

    def record_decision(self, job_id: str, key: str, decision: str, reviewer: str) -> bool:
        return self._write("INSERT OR IGNORE INTO decisions VALUES (?, ?, ?, ?, ?)", (job_id, key, decision, reviewer, time.time())) == 1

    def decisions(self, job_id: str) -> dict[str, str]:
        return {r["approval_key"]: r["decision"] for r in self._rows("SELECT * FROM decisions WHERE job_id = ?", (job_id,))}

    def record_export(self, job_id: str, path: Path, rows: int, digest: str) -> bool:
        return self._write("INSERT OR IGNORE INTO exports VALUES (?, ?, ?, ?, ?)", (job_id, str(path), rows, digest, time.time())) == 1

    def export(self, job_id: str) -> dict | None:
        rows = self._rows("SELECT * FROM exports WHERE job_id = ?", (job_id,))
        return dict(rows[0]) if rows else None

    def unfinished(self) -> list[str]:
        return [r["id"] for r in self._rows("SELECT id FROM jobs WHERE status IN ('queued', 'running') ORDER BY created_at")]


class JobManager:
    def __init__(self, team: AnalystTeam, store: JobStore, *, workers: int = 2, queue_max: int = 32, artifacts_dir: Path, exports_dir: Path):
        self.team, self.store = team, store
        self.n_workers, self.queue_max = workers, queue_max
        self.artifacts_dir, self.exports_dir = Path(artifacts_dir), Path(exports_dir)
        self._running: dict[str, asyncio.Task] = {}
        self._waiters: dict[str, asyncio.Event] = {}
        self._workers: list[asyncio.Task] = []

    async def start(self) -> None:
        self.queue: asyncio.Queue[str] = asyncio.Queue(maxsize=self.queue_max)
        for job_id in self.store.unfinished():  # at-least-once after a restart: the pipeline is read-only and exports are idempotent
            self.store.update(job_id, status="queued")
            self.emit(job_id, "requeued_after_restart", {})
            self.queue.put_nowait(job_id)
        self._workers = [asyncio.create_task(self._worker(i)) for i in range(self.n_workers)]

    async def stop(self) -> None:
        for worker in self._workers:
            worker.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)

    # ---------- events ----------
    def waiter(self, job_id: str) -> asyncio.Event:
        """Create BEFORE reading events, so a notification between the read and the wait is never lost."""
        return self._waiters.setdefault(job_id, asyncio.Event())

    def emit(self, job_id: str, kind: str, data: dict) -> dict:
        event = self.store.add_event(job_id, kind, data)
        waiter = self._waiters.pop(job_id, None)
        if waiter is not None:
            waiter.set()
        return event

    # ---------- commands ----------
    async def submit(self, question: str, evidence: str = "", idempotency_key: str | None = None) -> tuple[dict, bool]:
        if self.queue.full() and not self.store.key_exists(idempotency_key):
            raise QueueFull(f"{self.queue.qsize()} jobs are waiting")
        job, created = self.store.create({"question": question, "evidence": evidence}, idempotency_key)
        if created:
            self.emit(job["id"], "queued", {"position": self.queue.qsize() + 1})
            self.queue.put_nowait(job["id"])
        return job, created

    async def cancel(self, job_id: str) -> dict:
        job = self.store.get(job_id)
        if job["status"] in TERMINAL:
            raise Conflict(f"the job is already {job['status']}")
        if self.store.transition(job_id, ("queued", "awaiting_approval"), "cancelled", finished_at=time.time(), pending=None):
            self.emit(job_id, "cancelled", {"while": job["status"]})
            return self.store.get(job_id)
        task = self._running.get(job_id)
        if task is not None:
            task.cancel()
            await asyncio.wait({task}, timeout=10)
            for _ in range(100):  # the worker records the cancellation right after the task ends
                if self.store.get(job_id)["status"] != "running":
                    break
                await asyncio.sleep(0.01)
        return self.store.get(job_id)

    async def decide(self, job_id: str, key: str, approve: bool, reviewer: str) -> dict:
        job = self.store.get(job_id)
        pending = job["pending"] or {}
        if job["status"] != "awaiting_approval" or pending.get("key") != key:
            raise Conflict(f"no pending approval {key!r} (job status: {job['status']})")
        if not self.store.record_decision(job_id, key, "approve" if approve else "deny", reviewer):
            raise Conflict("this approval was already decided")
        self.emit(job_id, "approval_decided", {"key": key, "decision": "approve" if approve else "deny", "reviewer": reviewer})
        if pending["kind"] == "expensive_query":
            if approve and self.store.transition(job_id, ("awaiting_approval",), "queued", pending=None):
                self.queue.put_nowait(job_id)  # resumes from the checkpoint and runs exactly the approved SQL
            elif not approve and self.store.transition(job_id, ("awaiting_approval",), "denied", pending=None, finished_at=time.time(),
                                                       error="a reviewer denied the expensive query"):
                self.emit(job_id, "job_status", {"status": "denied"})
        else:
            export = await asyncio.to_thread(self._export, self.store.get(job_id)) if approve else None
            if self.store.transition(job_id, ("awaiting_approval",), "completed", pending=None, finished_at=time.time()):
                self.emit(job_id, "export_written" if export else "export_denied", {"rows": export["rows"], "sha256": export["sha256"]} if export else {})
                self.emit(job_id, "job_status", {"status": "completed"})
        return self.store.get(job_id)

    def _export(self, job: dict) -> dict:
        existing = self.store.export(job["id"])
        if existing:
            return existing  # idempotent: the side effect happens at most once per job
        outcome = job["outcome"]
        db = self.team.catalogs[outcome["database"]].db
        result = db.execute(outcome["sql"])  # the exact SQL the reviewer saw, re-run read-only (masked columns stay NULL)
        if not result.ok:
            raise RuntimeError(f"export query failed: {result.error}")
        self.exports_dir.mkdir(parents=True, exist_ok=True)
        path = self.exports_dir / f"{job['id']}.csv"
        partial = path.with_suffix(".csv.part")
        with partial.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(result.columns)
            writer.writerows(result.rows)
        digest = hashlib.sha256(partial.read_bytes()).hexdigest()
        os.replace(partial, path)  # atomic
        self.store.record_export(job["id"], path, len(result.rows), digest)
        return self.store.export(job["id"])

    # ---------- workers ----------
    async def _worker(self, index: int) -> None:
        while True:
            job_id = await self.queue.get()
            try:
                await self._process(job_id, index)
            finally:
                self.queue.task_done()

    async def _process(self, job_id: str, index: int) -> None:
        job = self.store.get(job_id)
        if job["status"] != "queued":
            return  # cancelled while waiting
        self.store.update(job_id, status="running", started_at=job["started_at"] or time.time())
        self.emit(job_id, "started", {"worker": index, "resumed": bool(job["checkpoint"])})
        approvals = frozenset(k for k, d in self.store.decisions(job_id).items() if d == "approve")

        async def on_event(kind: str, data: dict) -> None:
            self.emit(job_id, kind, data)

        task = asyncio.create_task(self.team.run(job["request"]["question"], job["request"].get("evidence", ""), on_event=on_event,
                                                 checkpoint=job["checkpoint"], approvals=approvals, job_id=job_id))
        self._running[job_id] = task
        try:
            outcome = await task
        except asyncio.CancelledError:
            if asyncio.current_task().cancelling():  # the server is shutting down: leave the job queued for the next start
                task.cancel()
                self.store.update(job_id, status="queued")
                raise
            self.store.update(job_id, status="cancelled", finished_at=time.time())
            self.emit(job_id, "cancelled", {"while": "running"})
            return
        except Exception as exc:  # a bug, not a model failure: record it and keep the worker alive
            self.store.update(job_id, status="failed", finished_at=time.time(), error=f"{type(exc).__name__}: {str(exc)[:300]}")
            self.emit(job_id, "job_status", {"status": "failed", "error": f"{type(exc).__name__}"})
            return
        finally:
            self._running.pop(job_id, None)
        folder = self.artifacts_dir / job_id
        if outcome.artifacts or outcome.report_markdown:
            folder.mkdir(parents=True, exist_ok=True)
            for name, data in outcome.artifacts.items():
                (folder / name).write_bytes(data)
            if outcome.report_markdown:
                (folder / "report.md").write_text(outcome.report_markdown)
        fields = {"outcome": outcome.to_dict(), "pending": outcome.pending_approval, "checkpoint": outcome.checkpoint, "error": outcome.error}
        if outcome.status in TERMINAL:
            fields["finished_at"] = time.time()
        self.store.update(job_id, status=outcome.status, **fields)
        self.emit(job_id, "job_status", {"status": outcome.status, "usage": outcome.usage, "pending_approval": outcome.pending_approval})
