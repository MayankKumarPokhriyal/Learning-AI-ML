import pytest

from conftest import FIXTURE_DOCS, INJECTED_DOC, build_index, make_pipeline, make_settings, write_corpus
from rag_assistant.chunking import Chunk
from rag_assistant.retrieval import PASSAGE_OVERHEAD_TOKENS, assemble_context, rrf_fuse
from rag_assistant.tokens import count_tokens


def test_rrf_matches_hand_computed_scores():
    fused = rrf_fuse([[1, 2, 3], [3, 1, 4]], k=60)
    scores = dict(fused)
    assert scores[1] == pytest.approx(1 / 61 + 1 / 62)
    assert scores[3] == pytest.approx(1 / 63 + 1 / 61)
    assert [item for item, _ in fused] == [1, 3, 2, 4]


def test_rrf_uses_ranks_only_and_breaks_ties_by_id():
    assert rrf_fuse([[7], [5]]) == [(5, pytest.approx(1 / 61)), (7, pytest.approx(1 / 61))]


def make_chunk(i: int, text: str, start: int, end: int, doc_id: str = "doc") -> Chunk:
    return Chunk(f"{doc_id}#{i}", doc_id, "Doc", "", "https://example.com", "test", start, end, text, count_tokens(text))


def test_context_packing_respects_budget_and_skips_near_duplicates():
    chunks = [make_chunk(0, "alpha " * 40, 0, 240), make_chunk(1, "alpha " * 40, 10, 250),  # 1 overlaps 0 by >50%
              make_chunk(2, "beta " * 200, 300, 1300), make_chunk(3, "gamma " * 10, 1400, 1460)]
    cost = [count_tokens(c.indexed_text) + PASSAGE_OVERHEAD_TOKENS for c in chunks]
    selected, used = assemble_context(chunks, [0, 1, 2, 3], budget_tokens=cost[0] + cost[3] + 5, max_chunks=5)
    assert [c.chunk_id for c in selected] == ["doc#0", "doc#3"]  # 1 is a duplicate, 2 does not fit, 3 still fits
    assert used == cost[0] + cost[3]


def test_hybrid_retrieval_records_every_stage(pipeline):
    result = pipeline.retrieve("Which environment variable sets UV_CACHE_DIR?")
    top = result.candidates[0]
    assert result.selected[0].doc_id == "concepts/cache"
    assert top.rrf_score is not None and top.rerank_score is not None
    assert {"dense", "bm25", "fusion", "rerank", "assemble"} <= set(result.timings)


@pytest.mark.parametrize("mode", ["dense", "bm25"])
def test_single_retrievers_still_work(pipeline, mode):
    result = pipeline.retrieve("upgrade a tool with uv tool upgrade", pipeline.retrieval_config(mode=mode, rerank=False))
    assert result.selected and result.selected[0].doc_id == "guides/tools"


def test_stopword_only_question_does_not_crash(pipeline):
    result = pipeline.retrieve("the and of", pipeline.retrieval_config(mode="bm25"))
    assert result.selected == []


def test_flagged_chunks_are_quarantined_before_the_prompt(tmp_path):
    corpus = write_corpus(tmp_path / "corpus", {**FIXTURE_DOCS, **INJECTED_DOC})
    settings = make_settings(tmp_path, corpus)
    report = build_index(settings)
    assert list(report.flagged_chunks) == ["guides/cache-tips#0"]
    pipeline = make_pipeline(settings)
    question = "Cache tips: how should users clear the cache?"
    guarded = pipeline.retrieve(question)
    assert "guides/cache-tips#0" in guarded.quarantined
    assert all(c.doc_id != "guides/cache-tips" for c in guarded.selected)
    unguarded = pipeline.retrieve(question, pipeline.retrieval_config(quarantine_flagged=False))
    assert any(c.doc_id == "guides/cache-tips" for c in unguarded.selected)
