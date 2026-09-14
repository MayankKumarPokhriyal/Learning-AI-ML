"""Evaluation harness: labeled tasks → agent runs → deterministic checks → quality, steps, tokens, latency, example cost.

    python -m research_agent.evaluate --out data/eval/report.json      # needs an LLM server; arXiv replays snapshots
    python -m research_agent.gate --report data/eval/report.json       # the CI regression gate
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import re
import statistics
import time
import unicodedata
from collections import Counter
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from research_agent import __version__, config
from research_agent.agent import ResearchAgent, RunResult, open_agent
from research_agent.guardrails import ARXIV_ID_IN_TEXT

DEFAULT_TASKS = config.PROJECT_ROOT / "evals" / "tasks.jsonl"


def load_tasks(path: str | Path = DEFAULT_TASKS) -> list[dict]:
    tasks = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip() and not line.startswith("//")]
    ids = [t["id"] for t in tasks]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate task ids")
    return tasks


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "").lower()
    text = text.replace("’", "'").replace("–", "-").replace("—", "-")
    return re.sub(r"\s+", " ", text).strip()


NUMBER_WORDS = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve".split())}


def numbers_in(text: str) -> list[float]:
    """Standalone numbers in normalized text ('1,000' -> 1000, 'v9' -> 9, 'eight' -> 8); digits inside arXiv ids are ignored."""
    text = re.sub(r"\bv(\d+)\b", r" \1", ARXIV_ID_IN_TEXT.sub(" ", text))
    found = [float(m.replace(",", "")) for m in re.findall(r"(?<![\w.,])\d{1,3}(?:,\d{3})+(?:\.\d+)?(?![\w]|\.\d)|(?<![\w.,])\d+(?:\.\d+)?(?![\w]|\.\d)", text)]
    return found + [float(NUMBER_WORDS[w]) for w in re.findall(r"[a-z]+", text) if w in NUMBER_WORDS]


def run_checks(task: dict, result: RunResult) -> list[dict]:
    """Deterministic checks on the user-visible answer and on what the agent did. Every check must pass."""
    answer = result.answer or {}
    text, citations = normalize(answer.get("answer", "")), set(answer.get("citations", []))
    rows = []
    for spec in task["checks"]:
        kind, value = spec["type"], spec.get("value")
        values = value if isinstance(value, list) else [value]
        if kind == "number":
            found = numbers_in(text)
            passed = all(any(abs(n - float(v)) <= spec.get("tol", 0) + 1e-9 for n in found) for v in values)
        elif kind == "mentions_at_least":
            passed = sum(normalize(v) in text for v in values) >= spec["n"]
        elif kind == "cites":
            passed = all(v in citations for v in values)
        elif kind == "cites_any":
            passed = any(v in citations for v in values)
        elif kind == "mentions_any":
            passed = any(normalize(v) in text for v in values)
        elif kind == "mentions_all":
            passed = all(normalize(v) in text for v in values)
        elif kind == "not_mentions":
            passed = not any(normalize(v) in text or v in citations for v in values)
        elif kind == "outcome":
            passed = answer.get("outcome") in values
        elif kind == "approval_requested":
            passed = any(a["tool"] == value for a in result.approvals)
        elif kind == "no_approvals":
            passed = not result.approvals
        elif kind == "no_citations":
            passed = not citations
        elif kind == "max_steps":
            passed = result.steps <= value
        else:
            raise ValueError(f"unknown check type {kind!r} in task {task['id']}")
        rows.append({"check": kind, "value": value, "passed": bool(passed)})
    return rows


def verify_evidence(tasks: list[dict], index) -> list[dict]:
    """Check every task's gold facts against the paper corpus (the arXiv snapshot) — gold labels never come from an LLM."""
    rows = []
    for task in tasks:
        for fact in task.get("evidence", []):
            paper = index.get(fact["arxiv_id"])
            for key, expected in fact.items():
                if key == "arxiv_id":
                    continue
                if paper is None:
                    actual = None
                elif key == "title_contains":
                    actual = expected if expected.lower() in paper.title.lower() else paper.title
                elif key == "first_author_contains":
                    actual = expected if expected.lower() in paper.authors[0].lower() else paper.authors[0]
                elif key == "abstract_contains":
                    actual = expected if expected in paper.abstract else "(not in abstract)"
                elif key == "primary_category":
                    actual = paper.primary_category
                elif key == "year":
                    actual = paper.year
                elif key == "n_authors":
                    actual = len(paper.authors)
                elif key == "version":
                    actual = paper.version
                else:
                    raise ValueError(f"unknown evidence key {key!r} in task {task['id']}")
                rows.append({"task": task["id"], "arxiv_id": fact["arxiv_id"], "fact": key, "expected": expected, "actual": actual,
                             "ok": actual == expected})
    return rows


def task_row(task: dict, result: RunResult) -> dict:
    checks = run_checks(task, result)
    answer = result.answer or {}
    return {"id": task["id"], "category": task["category"], "passed": result.status == "completed" and all(c["passed"] for c in checks),
            "failed_checks": [c for c in checks if not c["passed"]], "status": result.status, "stop_reason": result.stop_reason,
            "steps": result.steps, "tool_calls": result.tool_calls, "tool_errors": result.tool_errors, "tools_used": result.tools_used,
            "prompt_tokens": result.prompt_tokens, "completion_tokens": result.completion_tokens, "total_tokens": result.total_tokens,
            "latency_s": result.latency_s, "example_cost_usd": result.cost_usd(), "approvals": len(result.approvals),
            "guardrail_events": len(result.guardrail_events), "outcome": answer.get("outcome"), "citations": answer.get("citations", []),
            "answer": answer.get("answer"), "error": result.error, "run_id": result.run_id}


async def run_tasks(agent: ResearchAgent, tasks: list[dict], *, concurrency: int = 2,
                    on_row: Callable[[dict], None] | None = None) -> list[dict]:
    slots = asyncio.Semaphore(concurrency)

    async def one(task: dict) -> dict:
        async with slots:
            result = await agent.run(task["question"])
        row = task_row(task, result)
        if on_row:
            on_row(row)
        return row

    return list(await asyncio.gather(*(one(task) for task in tasks)))


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q * len(ordered)) - 1)] if ordered else 0.0  # nearest-rank


def summarize(rows: list[dict], prices: dict = config.EXAMPLE_PRICES_PER_MILLION) -> dict:
    n = len(rows)
    by_category = {}
    for category in sorted({r["category"] for r in rows}):
        group = [r for r in rows if r["category"] == category]
        by_category[category] = {"n": len(group), "success_rate": sum(r["passed"] for r in group) / len(group),
                                 "mean_steps": statistics.mean(r["steps"] for r in group),
                                 "mean_total_tokens": statistics.mean(r["total_tokens"] for r in group)}
    tool_calls = sum(r["tool_calls"] for r in rows)
    costs = [(r["prompt_tokens"] * prices["input"] + r["completion_tokens"] * prices["output"]) / 1e6 for r in rows]
    return {
        "n_tasks": n,
        "passed": sum(r["passed"] for r in rows),
        "success_rate": sum(r["passed"] for r in rows) / n if n else 0.0,
        "by_category": by_category,
        "mean_steps": statistics.mean(r["steps"] for r in rows) if n else 0.0,
        "mean_tool_calls": tool_calls / n if n else 0.0,
        "tool_error_rate": sum(r["tool_errors"] for r in rows) / tool_calls if tool_calls else 0.0,
        "mean_total_tokens": statistics.mean(r["total_tokens"] for r in rows) if n else 0.0,
        "prompt_token_share": sum(r["prompt_tokens"] for r in rows) / max(1, sum(r["total_tokens"] for r in rows)),
        "p50_latency_s": statistics.median(r["latency_s"] for r in rows) if n else 0.0,
        "p95_latency_s": _percentile([r["latency_s"] for r in rows], 0.95),
        "total_example_cost_usd": sum(costs),
        "mean_example_cost_usd": sum(costs) / n if n else 0.0,
        "stop_reasons": dict(Counter(r["stop_reason"] for r in rows)),
        "not_completed_rate": sum(r["status"] != "completed" for r in rows) / n if n else 0.0,
        "prices_per_million_tokens": prices,
        "prices_note": "EXAMPLE prices for cost arithmetic only, not a real price list",
    }


def write_report(rows: list[dict], summary: dict, meta: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"summary": summary, "meta": meta, "tasks": rows}, indent=1, ensure_ascii=False, default=str))
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tasks", default=str(DEFAULT_TASKS))
    parser.add_argument("--out", default=str(config.data_dir() / "eval" / "report.json"))
    parser.add_argument("--ids", nargs="*", help="only these task ids")
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--arxiv-mode", default="replay", choices=["replay", "record", "live"])
    args = parser.parse_args(argv)
    tasks = [t for t in load_tasks(args.tasks) if not args.ids or t["id"] in args.ids]

    async def go() -> tuple[list[dict], dict]:
        async with open_agent(arxiv_mode=args.arxiv_mode) as agent:
            start = time.perf_counter()
            rows = await run_tasks(agent, tasks, concurrency=args.concurrency,
                                   on_row=lambda r: print(f"{r['id']:5s} {'PASS' if r['passed'] else 'FAIL'} steps={r['steps']} "
                                                          f"tokens={r['total_tokens']} {r['latency_s']:.1f}s", flush=True))
            meta = {"model": agent.llm.name, "agent_version": __version__, "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
                    "wall_clock_s": round(time.perf_counter() - start, 1), "budget": agent.budget.__dict__, "arxiv_mode": args.arxiv_mode}
        return rows, meta

    rows, meta = asyncio.run(go())
    summary = summarize(rows)
    path = write_report(rows, summary, meta, args.out)
    print(json.dumps({k: summary[k] for k in ("n_tasks", "success_rate", "mean_steps", "mean_total_tokens", "p95_latency_s")}, indent=1))
    print(f"report written to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
