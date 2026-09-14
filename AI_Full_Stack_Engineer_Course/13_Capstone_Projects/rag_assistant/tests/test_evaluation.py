import json
import re

import pytest

from rag_assistant import config as cfg
from rag_assistant.cli import main
from rag_assistant.evaluation import (
    checkable_terms,
    choose_config,
    cohen_kappa,
    evaluate,
    gate_checks,
    load_questions,
    ranking_metrics,
    rule_correct,
    rule_faithful,
)
from rag_assistant.llm import StubLLM


def test_ranking_metrics_hand_computed():
    assert ranking_metrics(["a", "b", "c", "d"], {"c"}) == {"recall@1": 0.0, "recall@3": 1.0, "recall@5": 1.0, "recall@10": 1.0, "mrr@10": 1 / 3}
    assert ranking_metrics(["a"], {"z"})["mrr@10"] == 0.0


def test_rule_correctness_needs_every_rule_group_and_an_answer():
    item = {"kind": "answerable", "rules": [["--locked"], ["error", "fail"]]}
    assert rule_correct(item, "answered", "`--locked` raises an error")
    assert not rule_correct(item, "answered", "`--locked` checks the lockfile")
    assert not rule_correct(item, "refused", "`--locked` raises an error")
    assert rule_correct({"kind": "unanswerable", "rules": None}, "unsupported", "")
    timeout = {"kind": "answerable", "rules": [["5 min"], ["dependency confusion"]]}
    assert rule_correct(timeout, "answered", "Waits 5 minutes to avoid dependency‑confusion issues")  # model typography


def test_rule_faithfulness_checks_exact_terms_against_the_context():
    answer = "Set `UV_CACHE_DIR` or pass `--cache-dir <path>`; see pyproject.toml."
    assert checkable_terms(answer) == ["--cache-dir", "UV_CACHE_DIR", "pyproject.toml"]
    assert rule_faithful(answer, ["Use --cache-dir, UV_CACHE_DIR, or tool.uv.cache-dir in pyproject.toml."])
    assert not rule_faithful(answer.replace("UV_CACHE_DIR", "UV_CACHE_PATH"), ["Use --cache-dir, UV_CACHE_DIR in pyproject.toml."])


def test_cohen_kappa_matches_the_formula():
    rater_a = [True] * 6 + [False] * 4
    rater_b = [True] * 5 + [False] + [True] + [False] * 3  # 8 of 10 agree; p_e = .6*.6 + .4*.4 = .52
    assert cohen_kappa(rater_a, rater_b) == pytest.approx((0.8 - 0.52) / (1 - 0.52))
    assert cohen_kappa([True, True], [True, True]) is None  # undefined when both raters give one label


THRESHOLDS = {"_comment": "ignored", "retrieval": {"recall@5": [">=", 0.8]}, "operations": {"total_s_p95": ["<=", 10.0]}}


def test_gate_checks_and_cli_exit_codes(tmp_path, capsys):
    good = {"name": "good", "summary": {"recall@5": 0.9, "total_s_p95": 4.0}}
    bad = {"name": "bad", "summary": {"recall@5": 0.7, "total_s_p95": None}}
    assert all(c["passed"] for c in gate_checks(good["summary"], THRESHOLDS))
    thresholds = tmp_path / "gate.json"
    thresholds.write_text(json.dumps(THRESHOLDS))
    for report, code in [(good, 0), (bad, 1)]:
        path = tmp_path / f"{report['name']}.json"
        path.write_text(json.dumps(report))
        assert main(["gate", str(path), "--thresholds", str(thresholds)]) == code
    assert "FAIL  retrieval  recall@5" in capsys.readouterr().out
    assert main(["gate", str(tmp_path / "bad.json"), "--thresholds", str(thresholds), "--retrieval-only"]) == 1


def test_choose_config_prefers_the_faster_configuration_within_noise():
    def report(name, correct, faithful, p95):
        return {"name": name, "summary": {"n_questions": 20, "answer_correctness": correct, "faithfulness": faithful, "total_s_p95": p95}}

    thresholds = {"answers": {"answer_correctness": [">=", 0.7]}}
    result = choose_config([report("slow", 0.90, 0.95, 9.0), report("fast", 0.86, 0.95, 3.0), report("weak", 0.60, 1.0, 1.0)], thresholds)
    assert result["choice"] == "fast"  # 0.86 is within one question (0.05) of 0.90; "weak" fails the gate
    assert [row["passes_gate"] for row in result["table"]] == [True, True, False]


def judge_stub(messages):
    """TEST DOUBLE for the judge: marks an answer correct and faithful if it repeats the reference's first word."""
    user = messages[-1]["content"]
    reference = re.search(r"Reference answer: (\S+)", user).group(1).strip("`.")
    answer = user.rsplit("System answer:", 1)[1]
    ok = reference.lower() in answer.lower()
    return json.dumps({"reason": "stub", "correct": ok, "faithful": ok})


def test_evaluate_end_to_end_with_test_doubles(pipeline):
    questions = [
        {"qid": "q1", "kind": "answerable", "question": "Which command is uvx an alias for?", "answer": "uv tool run",
         "evidence": [{"doc_id": "guides/tools", "quote": "The `uvx` command is an alias for `uv tool run`."}], "rules": [["uv tool run"]]},
        {"qid": "q2", "kind": "answerable", "question": "Which environment variable sets the cache directory?", "answer": "UV_CACHE_DIR",
         "evidence": [{"doc_id": "concepts/cache", "quote": "The cache directory can be set with the `UV_CACHE_DIR` environment variable."}],
         "rules": [["UV_CACHE_DIR"]]},
        {"qid": "q3", "kind": "unanswerable", "question": "zebra quantum marmalade?", "answer": None, "evidence": [], "rules": None},
    ]
    report = evaluate(pipeline, questions, name="stub run", judge_llm=StubLLM(responder=judge_stub), concurrency=1)
    s = report["summary"]
    assert s["recall@5"] == 1.0 and s["n_questions"] == 3
    assert {r["qid"]: r["status"] for r in report["rows"]} == {"q1": "answered", "q2": "answered", "q3": "refused"}
    assert s["refusal_accuracy"] == 1.0 and s["citation_validity"] == 1.0 and s["citation_accuracy"] == 1.0
    assert s["judge_n"] == 2 and s["judge_rule_agreement_correct"] == 1.0
    assert evaluate(pipeline, questions, name="retrieval", retrieval_only=True)["summary"].keys() >= {"recall@1", "mrr@10", "evidence_in_context"}


def test_repository_evaluation_set_is_well_formed():
    questions = load_questions(cfg.PROJECT_DIR / "eval" / "questions.jsonl")
    assert 30 <= len(questions) <= 40 and len({q["qid"] for q in questions}) == len(questions)
    answerable = [q for q in questions if q["kind"] == "answerable"]
    unanswerable = [q for q in questions if q["kind"] == "unanswerable"]
    assert len(unanswerable) >= 6 and len(answerable) + len(unanswerable) == len(questions)
    for q in answerable:
        assert q["answer"] and q["evidence"] and q["rules"], q["qid"]
        for group in q["rules"]:
            for pattern in group:
                re.compile(pattern)
    assert all(q["absent_terms"] and not q["evidence"] for q in unanswerable)
    assert set(json.loads((cfg.PROJECT_DIR / "eval" / "gate.json").read_text())) >= {"retrieval", "answers", "operations"}
