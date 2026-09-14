import json

import pytest

from research_agent import gate
from research_agent.agent import RunResult
from research_agent.evaluate import DEFAULT_TASKS, load_tasks, numbers_in, run_checks, summarize, task_row, verify_evidence

THRESHOLDS = json.loads((DEFAULT_TASKS.parent / "thresholds.json").read_text())


def fake_result(answer_text="", citations=(), outcome="answered", **fields) -> RunResult:
    return RunResult(run_id="r", conversation_id="c", question="q", status=fields.pop("status", "completed"), stop_reason="final_answer",
                     answer={"answer": answer_text, "citations": list(citations), "outcome": outcome, "approval_ids": []}, **fields)


def test_task_file_is_valid_and_every_check_type_is_known():
    tasks = load_tasks()
    assert 20 <= len(tasks) <= 30
    assert {"lookup", "metadata", "abstract_qa", "multi_paper", "scope", "approval"} <= {t["category"] for t in tasks}
    for task in tasks:
        run_checks(task, fake_result())  # raises ValueError for an unknown check type
        assert task["checks"] and task["question"].strip()


def test_number_extraction_ignores_digits_inside_arxiv_ids():
    assert numbers_in("the transformer (1706.03762) has 8 authors; v9 is the latest; 2,294 problems; 28.4 bleu; eleven") == [8, 9, 2294, 28.4, 11]


@pytest.mark.parametrize("spec, text, citations, expected", [
    ({"type": "number", "value": 28.4}, "It reports 28.4 BLEU.", [], True),
    ({"type": "number", "value": [68, 31]}, "Llama 2 has 68 authors and GPT-3 has 31.", [], True),
    ({"type": "number", "value": 8}, "It has 28 authors.", [], False),
    ({"type": "cites", "value": ["2210.03629", "2406.12045"]}, "", ["2210.03629"], False),
    ({"type": "mentions_any", "value": ["cs.CV"]}, "Its primary category is cs.CV.", [], True),
    ({"type": "mentions_at_least", "value": ["calculator", "calendar", "translation"], "n": 2}, "A calculator and a calendar.", [], True),
    ({"type": "not_mentions", "value": ["pwned"]}, "PWNED", [], False),
    ({"type": "no_citations"}, "Not found.", [], True),
])
def test_checkers_are_deterministic(spec, text, citations, expected):
    task = {"id": "X", "category": "c", "question": "q", "checks": [spec]}
    assert run_checks(task, fake_result(text, citations))[0]["passed"] is expected


def test_verify_evidence_against_a_corpus(index):
    tasks = [{"id": "E1", "evidence": [{"arxiv_id": "2210.03629", "first_author_contains": "Yao", "year": 2022, "n_authors": 7},
                                        {"arxiv_id": "1706.03762", "primary_category": "cs.LG"}]}]
    assert [row["ok"] for row in verify_evidence(tasks, index)] == [True, True, True, False]


def test_summary_metrics():
    task = {"id": "T", "category": "lookup", "question": "q", "checks": [{"type": "cites", "value": "2210.03629"}]}
    rows = [task_row(task, fake_result("a", ["2210.03629"], steps=2, tool_calls=2, prompt_tokens=1000, completion_tokens=100, latency_s=4.0)),
            task_row(task, fake_result("b", [], steps=4, tool_calls=3, tool_errors=1, prompt_tokens=3000, completion_tokens=300, latency_s=10.0))]
    summary = summarize(rows, prices={"input": 1.0, "output": 2.0})
    assert summary["success_rate"] == 0.5 and summary["mean_steps"] == 3 and summary["tool_error_rate"] == pytest.approx(0.2)
    assert summary["p95_latency_s"] == 10.0 and summary["total_example_cost_usd"] == pytest.approx((4000 * 1 + 400 * 2) / 1e6)


def good_summary(**changes):
    summary = {"n_tasks": 27, "success_rate": 0.9, "by_category": {c: {"success_rate": 1.0} for c in THRESHOLDS.get("min_category_success_rate", {})},
               "not_completed_rate": 0.0, "mean_total_tokens": 5000, "p95_latency_s": 30.0}
    return {**summary, **changes}


def test_gate_passes_a_good_report_and_blocks_regressions():
    assert all(row["passed"] for row in gate.evaluate_gate(good_summary(), THRESHOLDS))
    failed = {row["check"] for row in gate.evaluate_gate(good_summary(success_rate=0.1, mean_total_tokens=10**6), THRESHOLDS) if not row["passed"]}
    assert failed == {"success_rate", "mean_total_tokens"}
    vs_baseline = gate.evaluate_gate(good_summary(success_rate=0.8), {**THRESHOLDS, "min_success_rate": 0.5}, baseline={"success_rate": 1.0})
    assert not vs_baseline[-1]["passed"]


def test_gate_cli_exit_codes(tmp_path):
    good, bad = tmp_path / "good.json", tmp_path / "bad.json"
    good.write_text(json.dumps({"summary": good_summary()}))
    bad.write_text(json.dumps({"summary": good_summary(success_rate=0.0)}))
    assert gate.main(["--report", str(good)]) == 0
    assert gate.main(["--report", str(bad)]) == 1
