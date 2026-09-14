"""Evaluation harness: retrieval metrics, answer quality (rules + an LLM judge checked for agreement), citations, refusals,
latency and tokens; configuration comparison with a pre-registered rule; and a regression gate CI can run."""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from rag_assistant.generation import format_passages, normalize_for_match, normalize_unicode
from rag_assistant.tracing import utc_now

REFUSED_STATUSES = {"refused", "unsupported"}
KS = (1, 3, 5, 10)

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {"reason": {"type": "string"}, "correct": {"type": "boolean"}, "faithful": {"type": "boolean"}},
    "required": ["reason", "correct", "faithful"],
    "additionalProperties": False,
}
JUDGE_SYSTEM = (
    "You grade one answer from a documentation assistant. Be strict and brief.\n"
    "reason: one short sentence explaining the verdicts (write it first).\n"
    "correct: true if the system answer states the key facts of the reference answer (wording may differ; extra correct detail is fine). "
    "If the reference says NOT ANSWERABLE, correct is true only if the system refused.\n"
    "faithful: true only if every factual claim in the system answer (including commands, options, file names and numbers) is supported "
    "by the context passages. Judge faithfulness against the passages only, not against your own knowledge or the reference answer."
)


# ── labels ─────────────────────────────────────────────────────────────────────────────────────────────────────
def load_questions(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def find_quote_spans(text: str, quote: str) -> list[tuple[int, int]]:
    pattern = r"\s+".join(re.escape(word) for word in quote.split())  # tolerate re-wrapped lines
    return [m.span() for m in re.finditer(pattern, text)]


def gold_chunk_ids(index, item: dict, min_cover: float = 0.5) -> set[str]:
    """Chunks that contain at least half of a gold evidence span. Labels are text spans, so they survive re-chunking."""
    gold = set()
    for evidence in item.get("evidence", []):
        spans = find_quote_spans(index.documents.get(evidence["doc_id"], ""), evidence["quote"])
        for chunk in index.chunks:
            if chunk.doc_id == evidence["doc_id"] and any(min(chunk.end, e) - max(chunk.start, s) >= min_cover * (e - s) for s, e in spans):
                gold.add(chunk.chunk_id)
    return gold


# ── metrics ────────────────────────────────────────────────────────────────────────────────────────────────────
def ranking_metrics(ranked_ids: list[str], gold: set[str], ks: tuple[int, ...] = KS) -> dict:
    hits = [chunk_id in gold for chunk_id in ranked_ids[:max(ks)]]
    metrics = {f"recall@{k}": float(any(hits[:k])) for k in ks}
    metrics["mrr@10"] = next((1.0 / rank for rank, hit in enumerate(hits[:10], start=1) if hit), 0.0)
    return metrics


def rule_correct(item: dict, status: str, answer: str) -> bool:
    """Every rule group needs one matching pattern. Matching tolerates typographic spaces/hyphens and "word-word" vs "word word"."""
    if item["kind"] == "unanswerable":
        return status in REFUSED_STATUSES
    if status != "answered":
        return False
    text = re.sub(r"\s+", " ", normalize_unicode(answer))
    variants = (text, re.sub(r"(?<=[A-Za-z])-(?=[A-Za-z])", " ", text))
    return all(any(re.search(pattern, variant, flags=re.I) for pattern in group for variant in variants) for group in item["rules"])


def checkable_terms(answer: str) -> list[str]:
    """Exact strings an answer asserts: tokens inside `code`, --options, ENV_VARS, and file names. Each must appear in the context."""
    terms = set()
    for m in re.finditer(r"`([^`]+)`", answer):
        for token in m.group(1).split():
            token = token.strip(",.;:()[]{}\"'")
            if len(token) >= 3 and re.search(r"[A-Za-z]", token) and not re.fullmatch(r"<[^>]*>|\.\.\.|\$", token):
                terms.add(token)
    terms.update(re.findall(r"(?<![\w-])(--[a-z][a-z0-9-]*)", answer))
    terms.update(re.findall(r"\b([A-Z][A-Z0-9]*_[A-Z0-9_]+)\b", answer))
    terms.update(re.findall(r"\b([\w-]+(?:\.[\w-]+)*\.(?:toml|lock|txt|cfg|py|ya?ml|json|in))\b", answer))
    return sorted(terms)


def rule_faithful(answer: str, context_texts: list[str]) -> bool:
    context = normalize_for_match("\n".join(context_texts))
    return all(normalize_for_match(term) in context for term in checkable_terms(answer))


def cohen_kappa(a: list[bool], b: list[bool]) -> float | None:
    a_arr, b_arr = np.asarray(a, dtype=bool), np.asarray(b, dtype=bool)
    if len(a_arr) == 0:
        return None
    p_observed = float(np.mean(a_arr == b_arr))
    p_a, p_b = a_arr.mean(), b_arr.mean()
    p_expected = float(p_a * p_b + (1 - p_a) * (1 - p_b))
    return None if p_expected >= 1.0 else (p_observed - p_expected) / (1 - p_expected)


def paired_bootstrap(a: list[float], b: list[float], n_resamples: int = 5000, seed: int = 0) -> tuple[float, float, float]:
    """Mean of (a - b) over the same questions with a 95% percentile bootstrap interval (questions resampled together)."""
    diffs = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    idx = np.random.default_rng(seed).integers(0, len(diffs), size=(n_resamples, len(diffs)))
    means = diffs[idx].mean(axis=1)
    return float(diffs.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def judge_answer(llm, question: str, reference: str, answer: str, passages: str) -> dict | None:
    user = f"Question: {question}\nReference answer: {reference}\n\nContext passages:\n{passages}\n\nSystem answer: {answer}"
    result = llm.complete([{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": user}], max_tokens=1024,
                          temperature=0.0, json_schema=JUDGE_SCHEMA, schema_name="judgement")
    try:
        verdict = json.loads(result.text)
        return {"correct": bool(verdict["correct"]), "faithful": bool(verdict["faithful"]), "reason": str(verdict["reason"])[:300]}
    except (json.JSONDecodeError, KeyError, TypeError):
        return None


# ── the harness ────────────────────────────────────────────────────────────────────────────────────────────────
def evaluate(pipeline, questions: list[dict], *, name: str, config=None, judge_llm=None, concurrency: int = 2,
             retrieval_only: bool = False) -> dict:
    config = config or pipeline.retrieval_config()
    index = pipeline.index
    prepared = []
    for item in questions:  # retrieval runs sequentially: the models are local
        gold = gold_chunk_ids(index, item) if item["kind"] == "answerable" else set()
        if item["kind"] == "answerable" and not gold:
            raise ValueError(f"{item['qid']}: gold evidence not found in index {index.version}")
        retrieval = pipeline.retrieve(item["question"], config)
        ranked = [c.chunk_id for c in retrieval.candidates]
        row = {"qid": item["qid"], "kind": item["kind"], "question": item["question"], "gold_chunks": sorted(gold), "top10": ranked[:10],
               "selected": [c.chunk_id for c in retrieval.selected], "context_tokens": retrieval.context_tokens,
               "retrieval_ms": 1000 * sum(retrieval.timings.values()), "quarantined": retrieval.quarantined}
        if item["kind"] == "answerable":
            row.update(ranking_metrics(ranked, gold))
            row["evidence_in_context"] = bool(gold & set(row["selected"]))
        prepared.append((item, retrieval, row))

    if not retrieval_only:
        with ThreadPoolExecutor(concurrency) as pool:  # LLM calls run concurrently, bounded to protect a shared server
            outputs = list(pool.map(lambda entry: pipeline.generate(entry[0]["question"], entry[1]), prepared))
        for (item, retrieval, row), (final, llm) in zip(prepared, outputs, strict=True):
            row.update(status=final.status, answer=final.answer, draft_answer=final.draft_answer, citations=[c["chunk_id"] for c in final.citations],
                       invalid_citations=final.invalid_citations, n_citations=len(final.citations) + len(final.invalid_citations),
                       n_valid_citations=len(final.citations), rule_correct=rule_correct(item, final.status, final.answer),
                       llm_s=llm.seconds, prompt_tokens=llm.prompt_tokens, completion_tokens=llm.completion_tokens,
                       finish_reason=llm.finish_reason, replayed=llm.replayed, total_s=row["retrieval_ms"] / 1000 + llm.seconds)
            if final.status == "answered":
                row["rule_faithful"] = rule_faithful(final.answer, [c.text for c in retrieval.selected])
                if item["kind"] == "answerable":
                    row["citation_hit"] = bool(set(row["gold_chunks"]) & set(row["citations"]))
        if judge_llm is not None:
            to_judge = [entry for entry in prepared if entry[2]["status"] == "answered"]  # refusals are scored by rule

            def run_judge(entry):
                item, retrieval, row = entry
                reference = item["answer"] if item["kind"] == "answerable" else "NOT ANSWERABLE from the documentation; the system should refuse."
                return judge_answer(judge_llm, item["question"], reference, row["answer"], format_passages(retrieval.selected, spotlight=False))

            with ThreadPoolExecutor(concurrency) as pool:
                verdicts = list(pool.map(run_judge, to_judge))
            for (_, _, row), verdict in zip(to_judge, verdicts, strict=True):
                if verdict is not None:
                    row.update(judge_correct=verdict["correct"], judge_faithful=verdict["faithful"], judge_reason=verdict["reason"])

    rows = [row for _, _, row in prepared]
    return {"name": name, "config": {**config.to_dict(), "label": config.label, "prompt_defense": pipeline.settings.prompt_defense},
            "index_version": index.version, "llm_model": getattr(pipeline.llm, "name", None), "created_at": utc_now(),
            "summary": summarize(rows, judged=judge_llm is not None, retrieval_only=retrieval_only), "rows": rows}


def _mean(values) -> float | None:
    values = [float(v) for v in values if v is not None]
    return float(np.mean(values)) if values else None


def _percentile(values, q: float) -> float | None:
    values = [float(v) for v in values if v is not None]
    return float(np.percentile(values, q)) if values else None


def summarize(rows: list[dict], *, judged: bool = False, retrieval_only: bool = False) -> dict:
    answerable = [r for r in rows if r["kind"] == "answerable"]
    unanswerable = [r for r in rows if r["kind"] == "unanswerable"]
    s = {"n_questions": len(rows), "n_answerable": len(answerable), "n_unanswerable": len(unanswerable)}
    for k in KS:
        s[f"recall@{k}"] = _mean(r[f"recall@{k}"] for r in answerable)
    s["mrr@10"] = _mean(r["mrr@10"] for r in answerable)
    s["evidence_in_context"] = _mean(r["evidence_in_context"] for r in answerable)
    s["retrieval_ms_p50"], s["retrieval_ms_p95"] = _percentile([r["retrieval_ms"] for r in rows], 50), _percentile([r["retrieval_ms"] for r in rows], 95)
    if retrieval_only:
        return s
    answered = [r for r in rows if r["status"] == "answered"]
    n_citations = sum(r["n_citations"] for r in rows)
    s.update({
        "answer_correctness": _mean(r["rule_correct"] for r in rows),
        "answerable_correctness": _mean(r["rule_correct"] for r in answerable),
        "refusal_accuracy": _mean(r["status"] in REFUSED_STATUSES for r in unanswerable),
        "false_refusal_rate": _mean(r["status"] != "answered" for r in answerable),
        "citation_validity": sum(r["n_valid_citations"] for r in rows) / n_citations if n_citations else None,
        "citation_accuracy": _mean(r.get("citation_hit") for r in answerable if r["status"] == "answered"),
        "rule_faithfulness": _mean(r.get("rule_faithful") for r in answered),
        "n_answered": len(answered),
        "n_unsupported": sum(r["status"] == "unsupported" for r in rows),
        "n_blocked_output": sum(r["status"] == "blocked_output" for r in rows),
        "n_invalid_output": sum(r["status"] == "invalid_output" for r in rows),
    })
    if judged:
        judged_rows = [r for r in answered if "judge_correct" in r]
        s.update({
            "judge_n": len(judged_rows),
            "faithfulness": _mean(r["judge_faithful"] for r in judged_rows),
            "judge_correctness": _mean(r.get("judge_correct", r["rule_correct"]) for r in rows),
            "judge_rule_agreement_correct": _mean(r["judge_correct"] == r["rule_correct"] for r in judged_rows),
            "judge_rule_kappa_correct": cohen_kappa([r["rule_correct"] for r in judged_rows], [r["judge_correct"] for r in judged_rows]),
            "judge_rule_agreement_faithful": _mean(r["judge_faithful"] == r["rule_faithful"] for r in judged_rows),
            "judge_rule_kappa_faithful": cohen_kappa([r["rule_faithful"] for r in judged_rows], [r["judge_faithful"] for r in judged_rows]),
        })
    s.update({
        "llm_s_p50": _percentile([r["llm_s"] for r in rows if r["prompt_tokens"]], 50),
        "llm_s_p95": _percentile([r["llm_s"] for r in rows if r["prompt_tokens"]], 95),
        "total_s_p50": _percentile([r["total_s"] for r in rows], 50),
        "total_s_p95": _percentile([r["total_s"] for r in rows], 95),
        "prompt_tokens_mean": _mean(r["prompt_tokens"] for r in rows if r["prompt_tokens"]),
        "completion_tokens_mean": _mean(r["completion_tokens"] for r in rows if r["prompt_tokens"]),
        "replayed_share": _mean(r["replayed"] for r in rows if r["prompt_tokens"]),
    })
    return s


# ── comparison and gate ────────────────────────────────────────────────────────────────────────────────────────
def load_thresholds(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def gate_checks(summary: dict, thresholds: dict, sections: tuple[str, ...] | None = None) -> list[dict]:
    checks = []
    for section, metrics in thresholds.items():
        if section.startswith("_") or (sections is not None and section not in sections):
            continue
        for metric, (op, limit) in metrics.items():
            value = summary.get(metric)
            passed = value is not None and (value >= limit if op == ">=" else value <= limit)
            checks.append({"section": section, "metric": metric, "value": value, "op": op, "threshold": limit, "passed": passed})
    return checks


def choose_config(reports: list[dict], thresholds: dict) -> dict:
    """Pre-registered rule: among configurations that pass the gate, keep those within one question of the best answer
    correctness, then within one question of the best faithfulness, then pick the lowest p95 latency."""
    table = []
    for report in reports:
        checks = gate_checks(report["summary"], thresholds)
        s = report["summary"]
        failed = [c["metric"] for c in checks if not c["passed"]]
        table.append({"config": report["name"], "passes_gate": not failed, "failed_checks": failed, "answer_correctness": s["answer_correctness"],
                      "faithfulness": s.get("faithfulness"), "total_s_p95": s["total_s_p95"]})
    one_question = 1 / reports[0]["summary"]["n_questions"]
    pool = [row for row in table if row["passes_gate"]] or table
    best = max(row["answer_correctness"] for row in pool)
    pool = [row for row in pool if row["answer_correctness"] >= best - one_question - 1e-9]
    best_faithful = max(row["faithfulness"] or 0.0 for row in pool)
    pool = [row for row in pool if (row["faithfulness"] or 0.0) >= best_faithful - one_question - 1e-9]
    choice = min(pool, key=lambda row: row["total_s_p95"])
    any_pass = any(row["passes_gate"] for row in table)
    return {"choice": choice["config"], "table": table, "any_config_passes_gate": any_pass,
            "reason": (f"{choice['config']}: {'passes the gate' if choice['passes_gate'] else 'no configuration passes the gate'}; "
                       f"answer correctness {choice['answer_correctness']:.3f} (best {best:.3f}), faithfulness "
                       f"{(choice['faithfulness'] or 0):.3f}, p95 latency {choice['total_s_p95']:.1f}s; ties within one question "
                       f"({one_question:.3f}) go to faithfulness, then to latency")}


def write_report(report: dict, directory: Path) -> tuple[Path, Path]:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "-", report["name"].lower()).strip("-")
    json_path, md_path = directory / f"eval_{slug}.json", directory / f"eval_{slug}.md"
    json_path.write_text(json.dumps(report, indent=2, default=str) + "\n")
    lines = [f"# Evaluation: {report['name']}", "", f"- config: {report['config']['label']}", f"- index: {report['index_version']}",
             f"- model: {report['llm_model']}", f"- created: {report['created_at']}", "", "| metric | value |", "|---|---|"]
    lines += [f"| {k} | {v:.3f} |" if isinstance(v, float) else f"| {k} | {v} |" for k, v in report["summary"].items()]
    md_path.write_text("\n".join(lines) + "\n")
    return json_path, md_path
