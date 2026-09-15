"""The evaluation question set: a selection RULE committed in evals/question_set.json, applied to BIRD mini-dev.

    python -m data_analyst_agents.questions           # show the selection and check it matches the committed ids
    python -m data_analyst_agents.questions --write   # (only when the rule changes) write the ids into question_set.json

Only question ids are committed. The questions, gold SQL and databases stay in data/ (CC BY-SA 4.0, fetched by `make data`).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from data_analyst_agents import config
from data_analyst_agents.data import load_questions, sha256_file
from data_analyst_agents.db import ReadOnlyDatabase

RULE_PATH = config.EVALS_DIR / "question_set.json"
DIFFICULTIES = ("simple", "moderate", "challenging")


def load_rule(path: Path | None = None) -> dict:
    return json.loads(Path(path or RULE_PATH).read_text())


def rank(question_id: int, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{question_id}".encode()).hexdigest()


def gold_check(db: ReadOnlyDatabase, question: dict, max_rows: int) -> dict:
    result = db.execute(question["SQL"], max_rows=max_rows)
    return {"ok": result.ok, "error": result.error, "elapsed_s": result.elapsed_s, "n_rows": len(result.rows), "truncated": result.truncated,
            "masked": result.masked_columns}


def apply_rule(rows: list[dict], rule: dict, dbs: dict[str, ReadOnlyDatabase]) -> tuple[list[dict], list[dict]]:
    """Returns (selected questions, excluded questions with the reason). Deterministic: no LLM, no randomness."""
    eligible, excluded = [], []
    for q in rows:
        if q["db_id"] not in rule["databases"]:
            continue
        g = gold_check(dbs[q["db_id"]], q, rule["max_gold_rows"])
        reason = (f"gold SQL fails ({g['error']})" if not g["ok"] else
                  f"gold SQL reads masked personal data {g['masked']}" if g["masked"] else
                  f"gold SQL slower than {rule['max_gold_seconds']} s" if g["elapsed_s"] > rule["max_gold_seconds"] else
                  f"gold result larger than {rule['max_gold_rows']} rows" if g["truncated"] else None)
        (excluded if reason else eligible).append({**q, "gold_rows": g["n_rows"], "gold_seconds": g["elapsed_s"], "exclusion": reason})
    selected = []
    for db_id in rule["databases"]:
        mine = sorted((q for q in eligible if q["db_id"] == db_id), key=lambda q: rank(q["question_id"], rule["seed"]))
        quota = dict(rule["per_database"])
        for difficulty in ("simple", "challenging"):  # shortfall rule: missing simple/challenging questions are replaced by moderate ones
            available = sum(q["difficulty"] == difficulty for q in mine)
            if available < quota[difficulty]:
                quota["moderate"] += quota[difficulty] - available
                quota[difficulty] = available
        for difficulty in DIFFICULTIES:
            selected += [q for q in mine if q["difficulty"] == difficulty][: quota[difficulty]]
    selected.sort(key=lambda q: (rule["databases"].index(q["db_id"]), q["question_id"]))
    return selected, excluded


def pass_k_ids(selected: list[dict], rule: dict) -> list[int]:
    ids = []
    for db_id in rule["databases"]:
        mine = sorted((q for q in selected if q["db_id"] == db_id), key=lambda q: rank(q["question_id"], rule["seed"]))
        ids += [q["question_id"] for q in mine[: rule["pass_k"]["per_database"]]]
    return sorted(ids)


def load_eval_questions(rule: dict | None = None) -> list[dict]:
    """The committed question set, joined with the downloaded BIRD file (whose hash must match the committed one)."""
    rule = rule or load_rule()
    actual = sha256_file(config.questions_path())
    if rule.get("source_sha256") and actual != rule["source_sha256"]:
        raise ValueError(f"{config.questions_path().name} has sha256 {actual[:12]}…, the question set was defined on {rule['source_sha256'][:12]}…")
    by_id = {q["question_id"]: q for q in load_questions()}
    return [by_id[i] for i in rule["question_ids"]]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)
    rule = load_rule()
    dbs = {db_id: ReadOnlyDatabase(db_id) for db_id in rule["databases"]}
    selected, excluded = apply_rule(load_questions(), rule, dbs)
    ids = [q["question_id"] for q in selected]
    for db_id in rule["databases"]:
        counts = {d: sum(q["db_id"] == db_id and q["difficulty"] == d for q in selected) for d in DIFFICULTIES}
        print(f"{db_id:26s} {counts}  excluded: {sum(q['db_id'] == db_id for q in excluded)}")
    print(f"selected {len(ids)} questions; pass^k subset {pass_k_ids(selected, rule)}")
    if args.write:
        rule.update(question_ids=ids, pass_k_question_ids=pass_k_ids(selected, rule), source_sha256=sha256_file(config.questions_path()))
        RULE_PATH.write_text(json.dumps(rule, indent=1) + "\n")
        print(f"wrote {RULE_PATH.name}")
        return 0
    same = ids == rule["question_ids"]
    print("matches the committed ids" if same else "DOES NOT match the committed ids")
    return 0 if same else 1


if __name__ == "__main__":
    raise SystemExit(main())
