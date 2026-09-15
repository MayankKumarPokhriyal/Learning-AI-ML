"""Scoring, reliability statistics, the question-selection rule and the regression gate."""

import itertools
import json
import math

from data_analyst_agents.db import QueryResult
from data_analyst_agents.evaluate import classify_failure, execution_match, paired_bootstrap, pass_at_k, pass_hat_k, reliability, summarize
from data_analyst_agents.gate import evaluate_gate
from data_analyst_agents.gate import main as gate_main
from data_analyst_agents.pipeline import Outcome
from data_analyst_agents.questions import apply_rule

THRESHOLDS = {"system": "multi_agent", "min_questions": 3, "min_execution_accuracy": 0.5, "min_accuracy_by_difficulty": {"simple": 0.5},
              "min_routing_accuracy": 0.9, "max_no_sql_rate": 0.1, "max_mean_llm_calls": 6, "max_mean_total_tokens": 15000, "max_p95_latency_s": 120,
              "max_accuracy_drop_vs_baseline": 0.05}


def result(rows, columns=("x",)):
    return QueryResult(sql="SELECT …", columns=list(columns), rows=rows)


def row(qid, correct, trial=0, first_correct=None, flagged=None, repairs=0, **overrides):
    base = {"question_id": qid, "db_id": "db", "difficulty": "simple", "trial": trial, "system": "multi_agent", "status": "completed", "routed_db": "db",
            "routing_correct": True, "correct": correct, "first_correct": correct if first_correct is None else first_correct, "has_sql": True,
            "verifier_flagged_first": flagged, "verifier_repairs": repairs, "sql_attempts": 1, "sql_errors": 0, "recovered_from_error": False,
            "llm_calls": 3, "replayed_calls": 0, "prompt_tokens": 1000, "completion_tokens": 200, "total_tokens": 1200, "latency_s": 5.0,
            "example_cost_usd": 0.00027, "would_need_approval": 0, "failure": None if correct else "wrong values (same shape)"}
    return {**base, **overrides}


def test_execution_match_uses_birds_set_semantics():
    assert execution_match([(1, "a"), (2, "b")], [(2, "b"), (1, "a"), (1, "a")])
    assert not execution_match([("a", 1)], [(1, "a")])


def test_pass_estimators_match_brute_force_enumeration():
    n, k = 4, 2
    for c in range(n + 1):
        subsets = list(itertools.combinations([1] * c + [0] * (n - c), k))
        assert math.isclose(pass_hat_k([c], n, k), sum(all(s) for s in subsets) / len(subsets))
        assert math.isclose(pass_at_k([c], n, k), sum(any(s) for s in subsets) / len(subsets))


def test_paired_bootstrap():
    same = paired_bootstrap([1, 0, 1, 1], [1, 0, 1, 1])
    assert same["delta"] == 0 and same["ci95"] == [0.0, 0.0]
    better = paired_bootstrap([1] * 20, [0] * 10 + [1] * 10)
    assert better["delta"] == 0.5 and better["ci95"][0] > 0 and better["a_only"] == 10 and better["b_only"] == 0


def test_failure_taxonomy():
    out = Outcome(system="multi_agent", question="q", database="db", status="completed")
    cases = [
        (result([(50.0,)]), result([(0.5,)]), "percentage vs ratio (×100)"),
        (result([(1, 2)], ("a", "b")), result([(1,)]), "wrong projection (extra or missing columns)"),
        (result([]), result([(1,)]), "empty result (filter value or join)"),
        (result([(1,), (2,)]), result([(1,)]), "wrong row set (filter, join or grouping)"),
        (result([(0.3334,)]), result([(0.3333,)]), "rounding or precision"),
        (result([("b", 1)], ("n", "v")), result([(1, "b")], ("v", "n")), "column order differs"),
    ]
    for predicted, gold, expected in cases:
        assert classify_failure(out, "db", predicted, gold) == expected
    assert classify_failure(out, "another_db", result([(1,)]), result([(1,)])) == "routed to the wrong database"


def test_summary_counts_verifier_catches_false_alarms_and_repairs():
    rows = [row(1, True, flagged=False), row(2, True, first_correct=False, flagged=True, repairs=1), row(3, False, flagged=False), row(4, True, flagged=True, repairs=1)]
    v = summarize(rows)["verifier"]
    assert summarize(rows)["accuracy"] == 0.75
    assert (v["first_wrong"], v["caught"], v["first_right"], v["false_alarms"], v["repairs_fixed"], v["repairs_broke"]) == (2, 1, 2, 1, 1, 0)


def test_reliability_pass_hat_k():
    rows = [row(1, True), row(1, True, trial=1), row(2, True), row(2, False, trial=1)]
    assert reliability(rows, [1, 2], 2) == {"k": 2, "n_questions": 2, "pass@1": 0.75, "pass@2": 1.0, "pass^2": 0.5, "flaky_questions": [2]}


def test_the_question_selection_rule(shop_db):
    specs = [("simple", "SELECT 1"), ("simple", "SELECT email FROM customers"), ("moderate", "SELECT 1"), ("moderate", "SELECT 2"),
             ("moderate", "SELECT 3"), ("challenging", "SELECT nope"), ("challenging", "SELECT 4")]
    questions = [{"question_id": i, "db_id": "shop", "difficulty": d, "question": "q", "evidence": "", "SQL": sql} for i, (d, sql) in enumerate(specs)]
    rule = {"databases": ["shop"], "per_database": {"simple": 2, "moderate": 1, "challenging": 1}, "seed": 1, "max_gold_seconds": 5, "max_gold_rows": 100}
    selected, excluded = apply_rule(questions, rule, {"shop": shop_db})
    assert {q["question_id"] for q in excluded} == {1, 5}  # reads masked PII · gold SQL fails
    assert sorted(q["difficulty"] for q in selected) == ["challenging", "moderate", "moderate", "simple"]  # the missing simple → moderate
    again = apply_rule(questions, rule, {"shop": shop_db})[0]
    assert [q["question_id"] for q in selected] == [q["question_id"] for q in again]  # deterministic (timings inside the dicts vary)


def test_the_gate_passes_and_blocks(tmp_path):
    good = {"systems": {"multi_agent": {"summary": summarize([row(i, True) for i in range(4)])}}}
    bad = {"systems": {"multi_agent": {"summary": summarize([row(i, i == 0) for i in range(4)])}}}
    assert all(r["passed"] for r in evaluate_gate(good, THRESHOLDS, {"accuracy": 1.0}))
    assert {r["check"] for r in evaluate_gate(bad, THRESHOLDS, {"accuracy": 1.0}) if not r["passed"]} == {"execution_accuracy", "accuracy[simple]", "accuracy vs baseline"}
    (tmp_path / "t.json").write_text(json.dumps(THRESHOLDS))
    (tmp_path / "r.json").write_text(json.dumps(bad))
    assert gate_main(["--report", str(tmp_path / "r.json"), "--thresholds", str(tmp_path / "t.json")]) == 1


def test_the_replay_gate_skips_latency_and_the_live_gate_enforces_it():
    slow = summarize([row(i, True, latency_s=500.0) for i in range(4)])
    live = evaluate_gate({"meta": {"mode": "live"}, "systems": {"multi_agent": {"summary": slow}}}, THRESHOLDS)
    replay = evaluate_gate({"meta": {"mode": "replay"}, "systems": {"multi_agent": {"summary": slow}}}, THRESHOLDS)
    assert [(r["passed"], r["skipped"]) for r in live if r["check"] == "p95_latency_s"] == [(False, False)]
    assert [(r["passed"], r["skipped"]) for r in replay if r["check"] == "p95_latency_s"] == [(True, True)]
