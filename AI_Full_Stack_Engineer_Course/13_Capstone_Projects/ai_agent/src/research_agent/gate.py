"""Regression gate for CI: an evaluation report must meet committed thresholds (and not fall far below a baseline).

    python -m research_agent.gate --report data/eval/report.json [--baseline evals/baseline_summary.json]

Exit code 0 = pass, 1 = fail (the CI job fails and the change cannot merge).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from research_agent import config

DEFAULT_THRESHOLDS = config.PROJECT_ROOT / "evals" / "thresholds.json"


def evaluate_gate(summary: dict, thresholds: dict, baseline: dict | None = None) -> list[dict]:
    rows = []

    def add(check: str, value: float, limit: str, passed: bool) -> None:
        rows.append({"check": check, "value": round(float(value), 4), "limit": limit, "passed": bool(passed)})

    add("n_tasks", summary["n_tasks"], f">= {thresholds['min_tasks']}", summary["n_tasks"] >= thresholds["min_tasks"])
    add("success_rate", summary["success_rate"], f">= {thresholds['min_success_rate']}", summary["success_rate"] >= thresholds["min_success_rate"])
    for category, minimum in thresholds.get("min_category_success_rate", {}).items():
        value = summary["by_category"].get(category, {}).get("success_rate", 0.0)
        add(f"success_rate[{category}]", value, f">= {minimum}", value >= minimum)
    add("not_completed_rate", summary["not_completed_rate"], f"<= {thresholds['max_not_completed_rate']}",
        summary["not_completed_rate"] <= thresholds["max_not_completed_rate"])
    add("mean_total_tokens", summary["mean_total_tokens"], f"<= {thresholds['max_mean_total_tokens']}",
        summary["mean_total_tokens"] <= thresholds["max_mean_total_tokens"])
    add("p95_latency_s", summary["p95_latency_s"], f"<= {thresholds['max_p95_latency_s']}", summary["p95_latency_s"] <= thresholds["max_p95_latency_s"])
    if baseline is not None:
        floor = baseline["success_rate"] - thresholds.get("max_success_rate_drop_vs_baseline", 0.1)
        add("success_rate vs baseline", summary["success_rate"], f">= {floor:.3f} (baseline {baseline['success_rate']:.3f})",
            summary["success_rate"] >= floor)
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--report", required=True)
    parser.add_argument("--thresholds", default=str(DEFAULT_THRESHOLDS))
    parser.add_argument("--baseline", help="a previous report or summary JSON to compare against")
    args = parser.parse_args(argv)
    summary = json.loads(Path(args.report).read_text())["summary"]
    thresholds = json.loads(Path(args.thresholds).read_text())
    baseline = None
    if args.baseline:
        loaded = json.loads(Path(args.baseline).read_text())
        baseline = loaded.get("summary", loaded)
    rows = evaluate_gate(summary, thresholds, baseline)
    for row in rows:
        print(f"{'PASS' if row['passed'] else 'FAIL'}  {row['check']:32s} {row['value']:>10} {row['limit']}")
    failed = [row for row in rows if not row["passed"]]
    print(f"regression gate: {'PASSED' if not failed else f'FAILED ({len(failed)} checks)'}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
