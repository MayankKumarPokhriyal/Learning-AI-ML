"""Evaluation: execution accuracy on BIRD mini-dev questions — the multi-agent team vs the single-agent baseline.

    python -m data_analyst_agents.evaluate --mode live --record --out data/eval/live.json     # real LLM; writes evals/recordings/
    python -m data_analyst_agents.evaluate --mode replay --out data/eval/replay.json          # offline, from the recordings (CI)

Execution accuracy follows BIRD's official evaluation: the predicted and gold result sets are equal as SETS of rows
(order and duplicates ignored). Both systems get the question and BIRD's evidence hint; neither is told the database.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np

from data_analyst_agents import __version__, config
from data_analyst_agents.config import JobBudget, LLMSettings
from data_analyst_agents.db import QueryResult
from data_analyst_agents.llm import OpenAICompatibleLLM
from data_analyst_agents.pipeline import AnalystTeam, Outcome, SingleAgentBaseline, TeamSettings
from data_analyst_agents.questions import load_eval_questions, load_rule
from data_analyst_agents.schema import load_catalogs
from data_analyst_agents.tracing import NOOP_TRACER

SYSTEMS = ("multi_agent", "single_agent")


def recording_path(system: str) -> Path:
    return config.EVALS_DIR / "recordings" / f"{system}.jsonl.gz"


def execution_match(predicted: list[tuple], gold: list[tuple]) -> bool:
    return set(map(tuple, predicted)) == set(map(tuple, gold))


def pass_at_k(successes: list[int], n: int, k: int) -> float:
    """Unbiased pass@k (Chen et al., 2021): the chance that at least one of k trials drawn from n succeeds."""
    return float(np.mean([1 - math.comb(n - c, k) / math.comb(n, k) for c in successes]))


def pass_hat_k(successes: list[int], n: int, k: int) -> float:
    """pass^k (τ-bench, Yao et al., 2024): the chance that ALL k trials drawn from n succeed."""
    return float(np.mean([math.comb(c, k) / math.comb(n, k) for c in successes]))


def paired_bootstrap(a: list[float], b: list[float], *, n_resamples: int = 10_000, seed: int = 0) -> dict:
    """95% percentile CI of mean(a - b), resampling QUESTIONS (both systems answered the same ones)."""
    a_arr, b_arr = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    diffs = a_arr - b_arr
    rng = np.random.default_rng(seed)
    boot = diffs[rng.integers(0, len(diffs), size=(n_resamples, len(diffs)))].mean(axis=1)
    low, high = np.percentile(boot, [2.5, 97.5])
    return {"n": len(diffs), "delta": float(diffs.mean()), "ci95": [float(low), float(high)], "share_of_resamples_above_zero": float((boot > 0).mean()),
            "a_only": int(((a_arr == 1) & (b_arr == 0)).sum()), "b_only": int(((a_arr == 0) & (b_arr == 1)).sum())}


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return float(ordered[max(0, math.ceil(q * len(ordered)) - 1)]) if ordered else 0.0  # nearest rank


def classify_failure(outcome: Outcome, gold_db: str, predicted: QueryResult | None, gold: QueryResult) -> str:
    """One category per wrong answer, assigned by rules in this order (computed from the saved results, never by an LLM)."""
    if outcome.status == "budget_exceeded":
        return "budget stop"
    if outcome.status == "failed":
        return "infrastructure error"
    if outcome.database != gold_db:
        return "routed to the wrong database"
    if predicted is None or not predicted.ok:
        return "no executable SQL"
    if predicted.truncated:
        return "result truncated by the row limit"
    if not predicted.rows and gold.rows:
        return "empty result (filter value or join)"
    if len(predicted.columns) != len(gold.columns):
        return "wrong projection (extra or missing columns)"
    if {tuple(sorted(map(str, r))) for r in predicted.rows} == {tuple(sorted(map(str, r))) for r in gold.rows}:
        return "column order differs"
    if len(set(predicted.rows)) != len(set(gold.rows)):
        return "wrong row set (filter, join or grouping)"
    if len(gold.rows) == 1 and len(gold.columns) == 1:
        p, g = predicted.rows[0][0], gold.rows[0][0]
        if isinstance(p, int | float) and isinstance(g, int | float):
            if g and (math.isclose(p, g * 100, rel_tol=1e-3) or math.isclose(p * 100, g, rel_tol=1e-3)):
                return "percentage vs ratio (×100)"
            if math.isclose(p, g, rel_tol=0.02):
                return "rounding or precision"
            return "wrong value (aggregation or formula)"
    return "wrong values (same shape)"


def score(question: dict, outcome: Outcome, gold: QueryResult) -> dict:
    predicted = outcome.result_obj
    correct = bool(predicted is not None and predicted.ok and not predicted.truncated and execution_match(predicted.rows, gold.rows))
    first = outcome.first_result_obj
    first_correct = bool(first is not None and first.ok and not first.truncated and execution_match(first.rows, gold.rows))
    errors = sum(1 for a in outcome.attempts if a.get("error_kind"))
    u = outcome.usage
    return {
        "question_id": question["question_id"], "db_id": question["db_id"], "difficulty": question["difficulty"], "trial": outcome.trial,
        "system": outcome.system, "status": outcome.status, "routed_db": outcome.database, "routing_correct": outcome.database == question["db_id"],
        "correct": correct, "first_correct": first_correct, "has_sql": predicted is not None and predicted.ok,
        "verifier_flagged_first": (outcome.first_verification or {}).get("flagged"), "verifier_repairs": outcome.verifier_repairs,
        "sql_attempts": len(outcome.attempts), "sql_errors": errors, "error_kinds": [a["error_kind"] for a in outcome.attempts if a.get("error_kind")],
        "recovered_from_error": errors > 0 and predicted is not None and predicted.ok,
        "llm_calls": u.get("llm_calls", 0), "replayed_calls": u.get("replayed_calls", 0), "prompt_tokens": u.get("prompt_tokens", 0),
        "completion_tokens": u.get("completion_tokens", 0), "total_tokens": u.get("total_tokens", 0), "latency_s": u.get("service_time_s", 0.0),
        "example_cost_usd": u.get("example_cost_usd", 0.0), "would_need_approval": len(outcome.would_need_approval),
        "failure": None if correct else classify_failure(outcome, question["db_id"], predicted, gold),
        "sql": outcome.sql, "gold_sql": question["SQL"], "question": question["question"], "evidence": question["evidence"],
        "pred_columns": predicted.columns if predicted else None, "pred_rows": len(predicted.rows) if predicted and predicted.ok else None,
        "pred_preview": [list(map(str, r))[:4] for r in predicted.rows[:3]] if predicted and predicted.ok else None,
        "gold_rows": len(gold.rows), "gold_preview": [list(map(str, r))[:4] for r in gold.rows[:3]],
        "verifier_problems": (outcome.first_verification or {}).get("problems"), "error": outcome.error,
    }


async def run_system(system, questions: list[dict], golds: dict[int, QueryResult], *, trials=(0,), concurrency: int = 2, on_row=None) -> list[dict]:
    slots = asyncio.Semaphore(concurrency)

    async def one(question: dict, trial: int) -> dict:
        async with slots:
            outcome = await system.run(question["question"], question["evidence"], trial=trial)
        row = score(question, outcome, golds[question["question_id"]])
        if on_row:
            on_row(row)
        return row

    rows = await asyncio.gather(*(one(q, t) for t in trials for q in questions))
    return sorted(rows, key=lambda r: (r["trial"], r["question_id"]))


def summarize(rows: list[dict], prices: dict = config.EXAMPLE_PRICES_PER_MILLION) -> dict:
    n = len(rows)

    def acc(group):
        return {"n": len(group), "accuracy": round(sum(r["correct"] for r in group) / len(group), 4)} if group else {"n": 0, "accuracy": 0.0}

    first_wrong = [r for r in rows if r["verifier_flagged_first"] is not None and r["has_sql"] and not r["first_correct"]]
    first_right = [r for r in rows if r["verifier_flagged_first"] is not None and r["first_correct"]]
    repaired = [r for r in rows if r["verifier_repairs"]]
    attempts = sum(r["sql_attempts"] for r in rows)
    return {
        "n": n, "accuracy": acc(rows)["accuracy"],
        "by_difficulty": {d: acc([r for r in rows if r["difficulty"] == d]) for d in ("simple", "moderate", "challenging")},
        "by_database": {db: acc([r for r in rows if r["db_id"] == db]) for db in sorted({r["db_id"] for r in rows})},
        "routing_accuracy": round(sum(r["routing_correct"] for r in rows) / n, 4),
        "no_sql_rate": round(sum(not r["has_sql"] for r in rows) / n, 4),
        "sql_error_rate": round(sum(r["sql_errors"] for r in rows) / attempts, 4) if attempts else 0.0,
        "questions_with_sql_error": sum(r["sql_errors"] > 0 for r in rows),
        "recovered_after_sql_error": sum(r["recovered_from_error"] for r in rows),
        "first_query_accuracy": round(sum(r["first_correct"] for r in rows) / n, 4),
        "verifier": {"first_wrong": len(first_wrong), "caught": sum(bool(r["verifier_flagged_first"]) for r in first_wrong),
                     "catch_rate": round(sum(bool(r["verifier_flagged_first"]) for r in first_wrong) / len(first_wrong), 4) if first_wrong else None,
                     "first_right": len(first_right), "false_alarms": sum(bool(r["verifier_flagged_first"]) for r in first_right),
                     "false_alarm_rate": round(sum(bool(r["verifier_flagged_first"]) for r in first_right) / len(first_right), 4) if first_right else None,
                     "repairs": len(repaired), "repairs_fixed": sum(not r["first_correct"] and r["correct"] for r in repaired),
                     "repairs_broke": sum(r["first_correct"] and not r["correct"] for r in repaired)},
        "mean_llm_calls": round(float(np.mean([r["llm_calls"] for r in rows])), 3),
        "mean_total_tokens": round(float(np.mean([r["total_tokens"] for r in rows])), 1),
        "mean_prompt_tokens": round(float(np.mean([r["prompt_tokens"] for r in rows])), 1),
        "p50_latency_s": round(_percentile([r["latency_s"] for r in rows], 0.5), 2), "p95_latency_s": round(_percentile([r["latency_s"] for r in rows], 0.95), 2),
        "mean_example_cost_usd": round(float(np.mean([r["example_cost_usd"] for r in rows])), 6),
        "statuses": dict(Counter(r["status"] for r in rows)), "failures": dict(Counter(r["failure"] for r in rows if r["failure"])),
        "would_need_approval": sum(r["would_need_approval"] > 0 for r in rows), "prices_per_million": prices,
    }


def reliability(rows: list[dict], question_ids: list[int], k: int) -> dict:
    by_question = {qid: [r for r in rows if r["question_id"] == qid and r["trial"] < k] for qid in question_ids}
    complete = {qid: rs for qid, rs in by_question.items() if len(rs) == k}
    if not complete:
        return {}
    successes = [sum(r["correct"] for r in rs) for rs in complete.values()]
    return {"k": k, "n_questions": len(complete), "pass@1": round(pass_hat_k(successes, k, 1), 4), f"pass@{k}": round(pass_at_k(successes, k, k), 4),
            f"pass^{k}": round(pass_hat_k(successes, k, k), 4), "flaky_questions": sorted(q for q, s in zip(complete, successes, strict=True) if 0 < s < k)}


def build_system(name: str, llm, catalogs, tracer=NOOP_TRACER):
    budget = JobBudget.from_env()  # the same per-question budget for both systems (ANALYST_MAX_* variables)
    if name == "multi_agent":
        return AnalystTeam(llm, catalogs, settings=TeamSettings(profile="answer", approval_mode="record"), budget=budget, tracer=tracer)
    if name == "single_agent":
        return SingleAgentBaseline(llm, catalogs, budget=budget, tracer=tracer)
    raise ValueError(f"unknown system {name!r}")


def compare(report: dict) -> dict:
    ma, sa = (sorted((r for r in report["systems"][s]["rows"] if r["trial"] == 0), key=lambda r: r["question_id"]) for s in SYSTEMS)
    assert [r["question_id"] for r in ma] == [r["question_id"] for r in sa], "systems must answer the same questions"
    ms, ss = report["systems"]["multi_agent"]["summary"], report["systems"]["single_agent"]["summary"]
    return {"accuracy_delta": paired_bootstrap([r["correct"] for r in ma], [r["correct"] for r in sa]),
            "token_ratio": round(ms["mean_total_tokens"] / ss["mean_total_tokens"], 3) if ss["mean_total_tokens"] else None,
            "llm_call_ratio": round(ms["mean_llm_calls"] / ss["mean_llm_calls"], 3) if ss["mean_llm_calls"] else None,
            "latency_p50_ratio": round(ms["p50_latency_s"] / ss["p50_latency_s"], 3) if ss["p50_latency_s"] else None}


async def evaluate(systems=SYSTEMS, *, mode: str = "live", pass_k: int = 2, concurrency: int = 2, record: bool = False, llm_factory=None,
                   catalogs=None, tracer=NOOP_TRACER, log=print) -> dict:
    """Run the committed question set through each system (trial 0), then extra trials on the pass^k subset."""
    rule = load_rule()
    catalogs = catalogs or load_catalogs(rule["databases"])
    questions = load_eval_questions(rule)
    golds = {q["question_id"]: catalogs[q["db_id"]].db.execute(q["SQL"]) for q in questions}
    subset = [q for q in questions if q["question_id"] in rule["pass_k_question_ids"]]
    report = {"meta": {"version": __version__, "mode": mode, "pass_k": pass_k, "question_ids": rule["question_ids"], "pass_k_question_ids": rule["pass_k_question_ids"],
                       "started_at": time.strftime("%Y-%m-%dT%H:%M:%S")}, "systems": {}}
    for name in systems:
        if llm_factory is not None:
            llm = llm_factory(name)
        else:
            settings = LLMSettings.from_env()
            if mode == "replay":
                settings = dataclasses.replace(settings, replay_only=True, cache_path=recording_path(name))
            llm = OpenAICompatibleLLM(settings)
        system = build_system(name, llm, catalogs, tracer)
        first_key, started = len(llm.used_keys), time.perf_counter()
        rows = await run_system(system, questions, golds, trials=(0,), concurrency=concurrency)
        if pass_k > 1 and subset:
            rows += await run_system(system, subset, golds, trials=tuple(range(1, pass_k)), concurrency=concurrency)
        entry = {"summary": summarize([r for r in rows if r["trial"] == 0]), "reliability": reliability(rows, rule["pass_k_question_ids"], pass_k),
                 "rows": rows, "wall_clock_s": round(time.perf_counter() - started, 1), "llm_stats": dict(llm.stats), "model": llm.name}
        if record and mode == "live" and hasattr(llm, "export_recording"):
            info = llm.export_recording(recording_path(name), llm.used_keys[first_key:])
            entry["recording"] = {"path": str(Path(info["path"]).relative_to(config.PROJECT_ROOT)) if Path(info["path"]).is_relative_to(config.PROJECT_ROOT) else str(info["path"]),
                                  "records": info["records"], "bytes": info["bytes"]}
        report["systems"][name] = entry
        s = entry["summary"]
        log(f"{name}: execution accuracy {s['accuracy']:.1%} on {s['n']} questions · mean {s['mean_llm_calls']:.1f} LLM calls, "
            f"{s['mean_total_tokens']:,.0f} tokens · {entry['wall_clock_s']:.0f} s wall clock · {llm.stats}")
    if all(s in report["systems"] for s in SYSTEMS):
        report["comparison"] = compare(report)
    return report


def write_report(report: dict, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=1, default=str, ensure_ascii=False))
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--system", choices=[*SYSTEMS, "both"], default="both")
    parser.add_argument("--mode", choices=["live", "replay"], default="live")
    parser.add_argument("--pass-k", type=int, default=2)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--record", action="store_true", help="live mode: write evals/recordings/<system>.jsonl.gz for offline replay")
    parser.add_argument("--out", default=str(config.data_dir() / "eval" / "report.json"))
    args = parser.parse_args(argv)
    systems = SYSTEMS if args.system == "both" else (args.system,)
    report = asyncio.run(evaluate(systems, mode=args.mode, pass_k=args.pass_k, concurrency=args.concurrency, record=args.record))
    print(f"report → {write_report(report, Path(args.out))}")
    if "comparison" in report:
        d = report["comparison"]["accuracy_delta"]
        print(f"multi-agent − single-agent accuracy: {d['delta']:+.1%} (95% CI {d['ci95'][0]:+.1%} … {d['ci95'][1]:+.1%}, n={d['n']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
