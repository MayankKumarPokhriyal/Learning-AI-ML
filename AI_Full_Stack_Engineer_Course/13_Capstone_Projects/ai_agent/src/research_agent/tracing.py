"""Structured JSONL traces: one redacted JSON line per event (llm step, tool call, guardrail, budget stop, approval)."""

from __future__ import annotations

import json
import threading
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from research_agent.guardrails import redact


class JsonlTracer:
    """Appends events to `<traces_dir>/<run_id>.jsonl` (or keeps them in memory only when traces_dir is None)."""

    def __init__(self, traces_dir: str | Path | None, run_id: str, secrets: Iterable[str] = ()):
        self.run_id = run_id
        self.secrets = tuple(secrets)
        self.path = Path(traces_dir) / f"{run_id}.jsonl" if traces_dir is not None else None
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.events: list[dict] = []
        # appending to an existing trace (e.g. an approval decided later) continues its sequence numbers
        self._offset = sum(1 for _ in self.path.open()) if self.path is not None and self.path.is_file() else 0
        self._lock = threading.Lock()

    def event(self, kind: str, **fields: Any) -> dict:
        record = {"ts": datetime.now(UTC).isoformat(timespec="milliseconds"), "run_id": self.run_id, "seq": self._offset + len(self.events),
                  "kind": kind, **redact(fields, self.secrets)}
        with self._lock:
            self.events.append(record)
            if self.path is not None:
                with self.path.open("a") as f:
                    f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        return record


def load_trace(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
