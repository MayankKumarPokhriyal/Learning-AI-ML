"""Harvest the paper corpus from the arXiv API into the local index.

    python -m research_agent.ingest                  # record mode: reuse snapshots, fetch (politely) what is missing
    ARXIV_MODE=replay python -m research_agent.ingest   # rebuild the index offline from snapshots only
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable

from research_agent import config
from research_agent.arxiv import ArxivClient, ArxivError
from research_agent.corpus import PaperIndex

# Well-known papers the evaluation tasks ask about (fetched by id, so they are always in the corpus).
LANDMARK_ID_BATCHES = (
    ("1706.03762", "1810.04805", "2005.14165", "2210.03629", "2302.04761", "2005.11401", "2106.09685", "1412.6980",
     "1512.03385", "2305.18290", "2201.11903", "2403.14720", "2302.12173", "2406.12045", "2312.04511", "2303.11366"),
    ("2307.09288", "2309.06180", "2310.06770", "2503.18813", "2104.09864", "2009.03300"),
)
# A topic search that adds related papers, so search faces realistic distractors. Add queries here to grow the corpus:
# each is one polite, snapshotted request — but arXiv may answer 429 for minutes at a time, so the first run can be slow.
SEED_QUERIES = {
    "rag": 'abs:"retrieval-augmented generation"',
}
PER_QUERY = 50


def harvest(client: ArxivClient, index: PaperIndex, *, id_batches=LANDMARK_ID_BATCHES, seed_queries=SEED_QUERIES,
            per_query: int = PER_QUERY, log: Callable[[str], None] = print) -> dict:
    """Run every request once (snapshots make re-runs free) and upsert the papers. Failures are reported, not hidden."""
    plan = [(f"id_list[{i}]", f"{len(batch)} landmark ids", lambda b=batch: client.get_by_ids(b, max_results=20))
            for i, batch in enumerate(id_batches)]
    plan += [(f"search:{name}", query, lambda q=query: client.search(q, max_results=per_query)) for name, query in seed_queries.items()]
    rows = []
    for request, detail, call in plan:
        hits_before, start = client.stats["snapshot_hits"], time.perf_counter()
        try:
            papers, error = call(), None
        except ArxivError as exc:
            papers, error = [], f"{type(exc).__name__}: {exc}"
        index.upsert(papers, source=f"arxiv-api:{request}")
        origin = "failed" if error else ("snapshot" if client.stats["snapshot_hits"] > hits_before else "live")
        rows.append({"request": request, "query": detail, "papers": len(papers), "origin": origin,
                     "seconds": round(time.perf_counter() - start, 1), "error": error})
        log(f"{request:24s} {len(papers):3d} papers · {origin:8s} · {rows[-1]['seconds']:6.1f} s" + (f" · {error}" if error else ""))
    return {"requests": rows, "papers_in_index": index.count(), "client_stats": dict(client.stats)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", default=config.arxiv_mode(), choices=["replay", "record", "live"])
    parser.add_argument("--per-query", type=int, default=PER_QUERY)
    args = parser.parse_args(argv)
    client = ArxivClient(config.snapshot_dir(), mode=args.mode)
    report = harvest(client, PaperIndex(config.corpus_path()), per_query=args.per_query)
    print(json.dumps({k: v for k, v in report.items() if k != "requests"}, indent=1))
    return 1 if any(r["error"] for r in report["requests"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
