"""Command line: `python -m rag_assistant <command>` (or `rag-assistant <command>` once installed).

download   fetch the pinned uv docs into data/corpus/<tag>/
ingest     incremental (re-)indexing of a corpus folder
ask        answer one question (streams to the terminal)
eval       run the labeled evaluation set and write a report
gate       check a report against thresholds (exit code 1 = regression) — for CI
warmup     download the models and tokenizer (used when building the Docker image)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rag_assistant import config as cfg


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rag-assistant", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("download", help="download the pinned documentation pages")
    p.add_argument("--tag", default=cfg.UV_DOCS_TAG)
    p.add_argument("--dest", type=Path, help="default: <data dir>/corpus/<tag>")
    p = sub.add_parser("ingest", help="incrementally index a corpus folder")
    p.add_argument("--corpus-dir", type=Path)
    p.add_argument("--index-dir", type=Path)
    p = sub.add_parser("ask", help="answer one question")
    p.add_argument("question")
    p.add_argument("--no-stream", action="store_true")
    p = sub.add_parser("eval", help="run the evaluation set")
    p.add_argument("--questions", type=Path, default=cfg.PROJECT_DIR / "eval" / "questions.jsonl")
    p.add_argument("--out", type=Path, default=cfg.PROJECT_DIR / "reports")
    p.add_argument("--name", default="current")
    p.add_argument("--retrieval-only", action="store_true", help="no LLM needed: retrieval metrics only")
    p.add_argument("--judge", action="store_true", help="also grade answers with the LLM judge")
    p = sub.add_parser("gate", help="fail (exit 1) if a report misses its thresholds")
    p.add_argument("report", type=Path)
    p.add_argument("--thresholds", type=Path, default=cfg.PROJECT_DIR / "eval" / "gate.json")
    p.add_argument("--retrieval-only", action="store_true")
    sub.add_parser("warmup", help="download models and tokenizer files")
    args = parser.parse_args(argv)
    return {"download": _download, "ingest": _ingest, "ask": _ask, "eval": _eval, "gate": _gate, "warmup": _warmup}[args.command](args)


def _download(args) -> int:
    from rag_assistant.corpus import download_corpus

    settings = cfg.Settings.from_env(corpus_tag=args.tag)
    manifest = download_corpus(args.dest or settings.corpus_dir, tag=args.tag)
    print(f"uv docs @ {manifest['tag']} ({manifest['license']}): {len(manifest['pages'])} pages, {manifest['total_bytes'] / 1e3:.0f} kB, "
          f"{manifest['downloaded']} downloaded now, missing at this tag: {manifest['missing'] or 'none'}")
    return 0


def _ingest(args) -> int:
    from rag_assistant.corpus import load_sources
    from rag_assistant.embeddings import SentenceTransformerEmbedder
    from rag_assistant.index import IndexStore, ingest

    overrides = {k: v for k, v in {"corpus_dir": args.corpus_dir, "index_dir": args.index_dir}.items() if v is not None}
    settings = cfg.Settings.from_env(**overrides)
    embedder = SentenceTransformerEmbedder(settings.embedding_model, device=settings.device, query_prompt=settings.query_prompt)
    report = ingest(load_sources(settings.corpus_dir), IndexStore(settings.index_dir), embedder,
                    chunk_tokens=settings.chunk_tokens, chunk_overlap=settings.chunk_overlap)
    changed = [("added", d) for d in report.added] + [("updated", d) for d in report.updated] + [("removed", d) for d in report.removed]
    for kind, doc_id in changed[:10]:
        print(f"  {kind}: {doc_id}")
    if len(changed) > 10:
        print(f"  … and {len(changed) - 10} more")
    for chunk_id, rules in report.flagged_chunks.items():
        print(f"  ⚠️ flagged {chunk_id}: {', '.join(rules)} (quarantined at query time)")
    print(report.summary())
    return 0


def _ask(args) -> int:
    from rag_assistant.pipeline import RAGPipeline

    pipeline = RAGPipeline.from_settings(cfg.Settings.from_env())
    for event, data in pipeline.ask_events(args.question, stream=not args.no_stream):
        if event == "delta":
            print(data["text"], end="", flush=True)
        elif event == "final":
            print(("\n" if not args.no_stream else "") + f"\n[{data['status']}] " + (data["answer"] if args.no_stream else ""))
            for c in data["citations"]:
                print(f"  - {c['chunk_id']}: “{c['quote']}” ({c['url']})")
        elif event == "error":
            print(f"\nerror: {data['message']}", file=sys.stderr)
            return 2
    return 0


def _eval(args) -> int:
    from rag_assistant.evaluation import evaluate, load_questions, write_report
    from rag_assistant.pipeline import RAGPipeline

    pipeline = RAGPipeline.from_settings(cfg.Settings.from_env())
    report = evaluate(pipeline, load_questions(args.questions), name=args.name, retrieval_only=args.retrieval_only,
                      judge_llm=pipeline.llm if args.judge and not args.retrieval_only else None)
    json_path, _ = write_report(report, args.out)
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in report["summary"].items()}, indent=2))
    print(f"report: {json_path}")
    return 0


def _gate(args) -> int:
    from rag_assistant.evaluation import gate_checks, load_thresholds

    report = json.loads(args.report.read_text())
    checks = gate_checks(report["summary"], load_thresholds(args.thresholds), sections=("retrieval",) if args.retrieval_only else None)
    for c in checks:
        value = "missing" if c["value"] is None else f"{c['value']:.3f}"
        print(f"{'PASS' if c['passed'] else 'FAIL'}  {c['section']:10s} {c['metric']:24s} {value:>8s} {c['op']} {c['threshold']}")
    failed = [c for c in checks if not c["passed"]]
    print(f"gate: {'PASSED' if not failed else f'FAILED ({len(failed)} of {len(checks)} checks)'} for report '{report['name']}'")
    return 1 if failed else 0


def _warmup(args) -> int:
    from rag_assistant.embeddings import CrossEncoderReranker, SentenceTransformerEmbedder
    from rag_assistant.tokens import count_tokens

    settings = cfg.Settings.from_env()
    SentenceTransformerEmbedder(settings.embedding_model, device="cpu", query_prompt=settings.query_prompt).embed_query("warm up")
    CrossEncoderReranker(settings.reranker_model, device="cpu").score("warm up", ["warm up"])
    print(f"models ready: {settings.embedding_model}, {settings.reranker_model}; tokenizer ready ({count_tokens('warm up')} tokens)")
    return 0
