"""Structured request traces: one JSON object per request, appended to a JSONL file (ship it to any log pipeline)."""

from __future__ import annotations

import json
import threading
from datetime import UTC, datetime
from pathlib import Path


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


class TraceWriter:
    def __init__(self, path: Path | None):
        self.path = Path(path) if path else None
        self._lock = threading.Lock()

    def write(self, record: dict) -> None:
        """Callers must pass already-redacted text: traces are logs, and logs leak."""
        if self.path is None:
            return
        line = json.dumps(record, ensure_ascii=False, default=str)
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")


def read_traces(path: Path) -> list[dict]:
    path = Path(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if path.exists() else []
