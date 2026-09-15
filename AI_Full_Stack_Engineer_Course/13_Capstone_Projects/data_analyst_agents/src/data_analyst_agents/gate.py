"""Regression gate for CI: an evaluation report must meet committed thresholds and not drop far below the baseline.

    python -m data_analyst_agents.gate --report data/eval/replay.json [--baseline evals/baseline_summary.json]

Exit code 0 = pass, 1 = fail (the CI job fails and the change cannot merge).

Latency is only gated on LIVE reports: in a replayed report the latencies are historical measurements that no code
change can move, so the replay gate (every pull request) marks them SKIP and the nightly live gate enforces them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from data_analyst_agents import config

DEFAULT_THRESHOLDS = config.EVALS_DIR / "thresholds.json"


def evaluate_gate(report: dict, thresholds: dict, baseline: dict | None = None) -> list[dict]:
    rows: list[dict] = []
    summary = report["systems"][thresholds["system"]]["summary"]
    replayed = report.get("meta", {}).get("mode") == "replay"

    def add(check: str, value: float, limit: str, passed: bool, skipped: bool = False) -> None:
        rows.append({"check": check, "value": round(float(value), 4), "limit": limit, "passed": bool(passed), "skipped": skipped})

    add("n_questions", summary["n"], f">= {thresholds['min_questions']}", summary["n"] >= thresholds["min_questions"])
    add("execution_accuracy", summary["accuracy"], f">= {thresholds['min_execution_accuracy']}", summary["accuracy"] >= thresholds["min_execution_accuracy"])
    for difficulty, minimum in thresholds.get("min_accuracy_by_difficulty", {}).items():
        value = summary["by_difficulty"].get(difficulty, {}).get("accuracy", 0.0)
        add(f"accuracy[{difficulty}]", value, f">= {minimum}", value >= minimum)
    add("routing_accuracy", summary["routing_accuracy"], f">= {thresholds['min_routing_accuracy']}", summary["routing_accuracy"] >= thresholds["min_routing_accuracy"])
    add("no_sql_rate", summary["no_sql_rate"], f"<= {thresholds['max_no_sql_rate']}", summary["no_sql_rate"] <= thresholds["max_no_sql_rate"])
    add("mean_llm_calls", summary["mean_llm_calls"], f"<= {thresholds['max_mean_llm_calls']}", summary["mean_llm_calls"] <= thresholds["max_mean_llm_calls"])
    add("mean_total_tokens", summary["mean_total_tokens"], f"<= {thresholds['max_mean_total_tokens']}", summary["mean_total_tokens"] <= thresholds["max_mean_total_tokens"])
    if replayed:
        add("p95_latency_s", summary["p95_latency_s"], f"<= {thresholds['max_p95_latency_s']} (replayed latencies are historical: checked by the live gate)", True, skipped=True)
    else:
        add("p95_latency_s", summary["p95_latency_s"], f"<= {thresholds['max_p95_latency_s']}", summary["p95_latency_s"] <= thresholds["max_p95_latency_s"])
    if baseline is not None:
        floor = baseline["accuracy"] - thresholds["max_accuracy_drop_vs_baseline"]
        add("accuracy vs baseline", summary["accuracy"], f">= {floor:.3f} (baseline {baseline['accuracy']:.3f})", summary["accuracy"] >= floor - 1e-9)
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--report", required=True)
    parser.add_argument("--thresholds", default=str(DEFAULT_THRESHOLDS))
    parser.add_argument("--baseline", help="a summary JSON with an 'accuracy' field (e.g. evals/baseline_summary.json)")
    args = parser.parse_args(argv)
    report = json.loads(Path(args.report).read_text())
    thresholds = json.loads(Path(args.thresholds).read_text())
    baseline = json.loads(Path(args.baseline).read_text()) if args.baseline else None
    rows = evaluate_gate(report, thresholds, baseline)
    for row in rows:
        status = "SKIP" if row["skipped"] else "PASS" if row["passed"] else "FAIL"
        print(f"{status}  {row['check']:24s} {row['value']:>10} {row['limit']}")
    failed = [row for row in rows if not row["passed"]]
    mode = report.get("meta", {}).get("mode", "live")
    print(f"regression gate ({thresholds['system']}, {mode} report): {'PASSED' if not failed else f'FAILED ({len(failed)} checks)'}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
