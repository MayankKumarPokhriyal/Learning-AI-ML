"""OpenTelemetry tracing to JSONL, named after the GenAI semantic conventions (invoke_agent · chat · execute_tool).

One trace per job: `invoke_workflow data-analyst-team` → `invoke_agent <role>` → `chat <model>` / `execute_tool <tool>`.
Spans carry gen_ai.usage.* token counts and app.* attributes (agent, cached, latency). The JSONL file can be shipped to
any OTLP backend later; here it is also the source for per-agent cost tables.
"""

from __future__ import annotations

import json
import os
import threading
from collections import defaultdict
from pathlib import Path

from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult
from opentelemetry.trace import NoOpTracer

from data_analyst_agents import __version__, config

PROVIDER_NAME = os.environ.get("LLM_PROVIDER_NAME", "openai-compatible")


class JsonlSpanExporter(SpanExporter):
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def export(self, spans) -> SpanExportResult:
        lines = []
        for s in spans:
            lines.append(json.dumps({
                "name": s.name, "trace_id": format(s.context.trace_id, "032x"), "span_id": format(s.context.span_id, "016x"),
                "parent_id": format(s.parent.span_id, "016x") if s.parent else None, "kind": s.kind.name,
                "start": s.start_time / 1e9, "duration_s": round((s.end_time - s.start_time) / 1e9, 4), "status": s.status.status_code.name,
                "attributes": {k: (list(v) if isinstance(v, tuple) else v) for k, v in (s.attributes or {}).items()},
                "events": [{"name": e.name, "attributes": dict(e.attributes or {})} for e in s.events],
            }, ensure_ascii=False, default=str))
        with self._lock, self.path.open("a") as f:
            f.write("".join(line + "\n" for line in lines))
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        return None


class Tracing:
    """A private TracerProvider (not the global one), so tests and notebooks can create several without warnings."""

    def __init__(self, path: Path | None = None, *, service_name: str = "data-analyst-agents"):
        self.path = Path(path) if path is not None else None
        self.provider = TracerProvider(resource=Resource.create({"service.name": service_name, "service.version": __version__}))
        if self.path is not None:
            self.provider.add_span_processor(SimpleSpanProcessor(JsonlSpanExporter(self.path)))  # synchronous: fine for a local file
        self.tracer = self.provider.get_tracer("data_analyst_agents", __version__)

    def shutdown(self) -> None:
        self.provider.shutdown()


NOOP_TRACER = NoOpTracer()


def load_spans(path: Path) -> list[dict]:
    path = Path(path)
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.is_file() else []


def agent_usage(spans: list[dict], prices: dict = config.EXAMPLE_PRICES_PER_MILLION) -> list[dict]:
    """Per-agent LLM calls, tokens, latency and example cost from `chat` spans."""
    groups: dict[str, dict] = defaultdict(lambda: {"calls": 0, "replayed": 0, "input_tokens": 0, "output_tokens": 0, "latency_s": 0.0})
    for span in spans:
        a = span["attributes"]
        if a.get("gen_ai.operation.name") != "chat":
            continue
        g = groups[a.get("app.agent", "?")]
        g["calls"] += 1
        g["replayed"] += int(bool(a.get("app.cached")))
        g["input_tokens"] += int(a.get("gen_ai.usage.input_tokens", 0))
        g["output_tokens"] += int(a.get("gen_ai.usage.output_tokens", 0))
        g["latency_s"] += float(a.get("app.latency_s", 0.0))
    rows = []
    for agent, g in sorted(groups.items(), key=lambda kv: -kv[1]["input_tokens"] - kv[1]["output_tokens"]):
        cost = (g["input_tokens"] * prices["input"] + g["output_tokens"] * prices["output"]) / 1e6
        rows.append({"agent": agent, **g, "latency_s": round(g["latency_s"], 2), "mean_latency_s": round(g["latency_s"] / g["calls"], 2),
                     "example_cost_usd": round(cost, 6)})
    return rows
