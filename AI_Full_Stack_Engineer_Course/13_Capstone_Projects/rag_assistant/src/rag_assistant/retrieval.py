"""Hybrid retrieval: dense vectors + BM25, fused with reciprocal rank fusion (RRF), reranked by a cross-encoder, then packed
into a token budget."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

import numpy as np

from rag_assistant.chunking import Chunk
from rag_assistant.tokens import count_tokens

PASSAGE_OVERHEAD_TOKENS = 16  # the <passage id="..." source="..."> wrapper around each chunk in the prompt


@dataclass(frozen=True)
class RetrievalConfig:
    mode: str = "hybrid"  # dense | bm25 | hybrid
    k_dense: int = 30
    k_bm25: int = 30
    rrf_k: int = 60
    rerank: bool = True
    rerank_depth: int = 20
    top_k: int = 5
    context_budget: int = 1500
    quarantine_flagged: bool = True

    @classmethod
    def from_settings(cls, settings) -> RetrievalConfig:
        return cls(mode=settings.retrieval_mode, k_dense=settings.k_dense, k_bm25=settings.k_bm25, rrf_k=settings.rrf_k,
                   rerank=settings.rerank, rerank_depth=settings.rerank_depth, top_k=settings.top_k,
                   context_budget=settings.context_budget, quarantine_flagged=settings.quarantine_flagged)

    @property
    def label(self) -> str:
        name = {"dense": "dense", "bm25": "BM25", "hybrid": "hybrid RRF"}[self.mode]
        return f"{name}{' + rerank' if self.rerank else ''} · top-{self.top_k} · {self.context_budget} tokens"

    def to_dict(self) -> dict:
        return asdict(self)


def rrf_fuse(rankings: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    """Reciprocal rank fusion: score(d) = Σ 1 / (k + rank of d in each list), ranks starting at 1. Ties break by id."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


class BM25Index:
    def __init__(self, texts: list[str]):
        import bm25s
        import Stemmer

        self._bm25s, self.stemmer, self.size = bm25s, Stemmer.Stemmer("english"), len(texts)
        self.model = bm25s.BM25(k1=1.5, b=0.75, method="lucene")
        if texts:
            self.model.index(bm25s.tokenize(texts, stopwords="en", stemmer=self.stemmer, show_progress=False), show_progress=False)

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        if self.size == 0:
            return []
        tokens = self._bm25s.tokenize([query], stopwords="en", stemmer=self.stemmer, show_progress=False)
        if not tokens.vocab:  # e.g. a question made only of stopwords
            return []
        ids, scores = self.model.retrieve(tokens, k=min(k, self.size), show_progress=False)
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0], strict=True) if s > 0]


@dataclass
class Candidate:
    row: int
    chunk_id: str
    dense_rank: int | None = None
    dense_score: float | None = None
    bm25_rank: int | None = None
    bm25_score: float | None = None
    rrf_score: float | None = None
    rerank_score: float | None = None

    def to_trace(self) -> dict:
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in asdict(self).items() if v is not None and k != "row"}


@dataclass
class RetrievalResult:
    candidates: list[Candidate]  # final ranking (after quarantine and reranking)
    selected: list[Chunk]  # what goes into the prompt
    context_tokens: int
    timings: dict[str, float] = field(default_factory=dict)  # seconds per stage
    quarantined: list[str] = field(default_factory=list)


def assemble_context(chunks: list[Chunk], ranked_rows: list[int], budget_tokens: int, max_chunks: int) -> tuple[list[Chunk], int]:
    """Walk the ranking best-first; skip near-duplicates (>50% overlap with a chosen chunk) and chunks that don't fit."""
    selected, used = [], 0
    for row in ranked_rows:
        chunk = chunks[row]
        if any(s.doc_id == chunk.doc_id and min(s.end, chunk.end) - max(s.start, chunk.start) > 0.5 * (chunk.end - chunk.start) for s in selected):
            continue
        cost = count_tokens(chunk.indexed_text) + PASSAGE_OVERHEAD_TOKENS
        if used + cost > budget_tokens:
            continue
        selected.append(chunk)
        used += cost
        if len(selected) == max_chunks:
            break
    return selected, used


class Retriever:
    def __init__(self, index, embedder, reranker=None):
        self.index, self.embedder, self.reranker = index, embedder, reranker
        self.bm25 = BM25Index([c.indexed_text for c in index.chunks])

    def retrieve(self, question: str, config: RetrievalConfig, query_vector: np.ndarray | None = None) -> RetrievalResult:
        chunks, timings = self.index.chunks, {}
        clock = time.perf_counter()

        def lap(stage: str) -> None:
            nonlocal clock
            now = time.perf_counter()
            timings[stage] = timings.get(stage, 0.0) + now - clock
            clock = now

        found: dict[int, Candidate] = {}
        dense_rows, bm25_rows = [], []
        if config.mode in ("dense", "hybrid") and chunks:
            if query_vector is None:
                query_vector = self.embedder.embed_query(question)
                lap("embed")
            scores = self.index.embeddings @ query_vector  # cosine similarity: all vectors are unit length
            top = np.argsort(-scores, kind="stable")[:config.k_dense]
            for rank, row in enumerate(top.tolist(), start=1):
                found.setdefault(row, Candidate(row, chunks[row].chunk_id)).__dict__.update(dense_rank=rank, dense_score=float(scores[row]))
                dense_rows.append(row)
            lap("dense")
        if config.mode in ("bm25", "hybrid"):
            for rank, (row, score) in enumerate(self.bm25.search(question, config.k_bm25), start=1):
                found.setdefault(row, Candidate(row, chunks[row].chunk_id)).__dict__.update(bm25_rank=rank, bm25_score=score)
                bm25_rows.append(row)
            lap("bm25")
        if config.mode == "hybrid":
            fused = rrf_fuse([dense_rows, bm25_rows], k=config.rrf_k)
            for row, score in fused:
                found[row].rrf_score = score
            ranking = [found[row] for row, _ in fused]
            lap("fusion")
        else:
            ranking = [found[row] for row in (dense_rows or bm25_rows)]
        quarantined = []
        if config.quarantine_flagged:
            quarantined = [c.chunk_id for c in ranking if chunks[c.row].flags]
            ranking = [c for c in ranking if not chunks[c.row].flags]
        if config.rerank and self.reranker is not None and ranking:
            head = ranking[:config.rerank_depth]
            scores = self.reranker.score(question, [chunks[c.row].indexed_text for c in head])
            for candidate, score in zip(head, scores, strict=True):
                candidate.rerank_score = float(score)
            ranking = [head[i] for i in np.argsort(-scores, kind="stable")] + ranking[config.rerank_depth:]
            lap("rerank")
        selected, used = assemble_context(chunks, [c.row for c in ranking], config.context_budget, config.top_k)
        lap("assemble")
        return RetrievalResult(ranking, selected, used, timings, quarantined)
