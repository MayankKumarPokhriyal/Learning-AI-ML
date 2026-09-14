import json

import numpy as np
import pytest

from conftest import FIXTURE_DOCS, make_settings, write_corpus
from rag_assistant.cache import ResponseCache
from rag_assistant.corpus import load_sources
from rag_assistant.embeddings import HashingEmbedder
from rag_assistant.index import IndexStore, ingest
from rag_assistant.tracing import read_traces


class CountingEmbedder(HashingEmbedder):
    def __init__(self):
        super().__init__()
        self.embedded = 0

    def embed_documents(self, texts):
        self.embedded += len(texts)
        return super().embed_documents(texts)


def test_incremental_ingestion_only_reembeds_changed_documents(tmp_path):
    corpus = write_corpus(tmp_path / "corpus", FIXTURE_DOCS)
    store, embedder = IndexStore(tmp_path / "index"), CountingEmbedder()

    first = ingest(load_sources(corpus), store, embedder, chunk_tokens=120, chunk_overlap=20)
    assert sorted(first.added) == ["concepts/cache", "guides/install", "guides/tools"] and first.chunks_embedded == first.chunks_total
    index_v1 = store.load()

    again = ingest(load_sources(corpus), store, embedder, chunk_tokens=120, chunk_overlap=20)
    assert again.chunks_embedded == 0 and not again.wrote_new_version and again.index_version == first.index_version

    (corpus / "guides/tools.md").write_text(FIXTURE_DOCS["guides/tools.md"] + "\n## Uninstalling tools\n\nRun `uv tool uninstall black`.\n")
    changed = ingest(load_sources(corpus), store, embedder, chunk_tokens=120, chunk_overlap=20)
    index_v2 = store.load()
    tools_chunks = [c for c in index_v2.chunks if c.doc_id == "guides/tools"]
    assert changed.updated == ["guides/tools"] and sorted(changed.unchanged) == ["concepts/cache", "guides/install"]
    assert changed.chunks_embedded == len(tools_chunks)
    assert store.current_version() == changed.index_version != first.index_version
    cache_rows = [i for i, c in enumerate(index_v1.chunks) if c.doc_id == "concepts/cache"]
    assert np.array_equal(index_v1.embeddings[cache_rows], index_v2.embeddings[[index_v2.row_of[index_v1.chunks[i].chunk_id] for i in cache_rows]])

    (corpus / "guides/install.md").unlink()
    removed = ingest(load_sources(corpus), store, embedder, chunk_tokens=120, chunk_overlap=20)
    assert removed.removed == ["guides/install"] and removed.chunks_embedded == 0


def test_new_release_with_unchanged_text_updates_source_metadata_without_reembedding(tmp_path):
    store, embedder = IndexStore(tmp_path / "index"), CountingEmbedder()
    ingest(load_sources(write_corpus(tmp_path / "v1", FIXTURE_DOCS, tag="v1")), store, embedder, chunk_tokens=120, chunk_overlap=20)
    anchors_v1 = [c.url.partition("#")[2] for c in store.load().chunks]
    report = ingest(load_sources(write_corpus(tmp_path / "v2", FIXTURE_DOCS, tag="v2")), store, embedder, chunk_tokens=120, chunk_overlap=20)
    assert report.chunks_embedded == 0 and len(report.unchanged) == 3 and report.wrote_new_version
    chunks = store.load().chunks
    assert all(c.version == "v2" and "/docs/v2/" in c.url for c in chunks)
    assert [c.url.partition("#")[2] for c in chunks] == anchors_v1  # section anchors survive the URL update


def test_changing_chunk_settings_forces_a_full_rebuild(tmp_path):
    corpus = write_corpus(tmp_path / "corpus", FIXTURE_DOCS)
    store = IndexStore(tmp_path / "index")
    ingest(load_sources(corpus), store, HashingEmbedder(), chunk_tokens=120, chunk_overlap=20)
    rebuilt = ingest(load_sources(corpus), store, HashingEmbedder(), chunk_tokens=200, chunk_overlap=20)
    assert "chunk_tokens" in rebuilt.full_rebuild_reason and len(rebuilt.added) == 3


def test_missing_index_raises_a_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="ingest"):
        IndexStore(tmp_path / "nothing").load()


def test_exact_cache_normalizes_questions_and_is_scoped():
    cache = ResponseCache(mode="exact")
    cache.store("How do I clear the cache?", "index-a", "answer")
    hit, _ = cache.lookup("  how do i CLEAR the cache ", "index-a")
    assert hit is not None and hit.kind == "exact" and hit.value == "answer"
    assert cache.lookup("How do I clear the cache?", "index-b") == (None, None)  # another index version never matches
    assert cache.stats["exact_hits"] == 1 and cache.stats["misses"] == 1


def test_semantic_cache_uses_the_threshold_and_embeds_only_on_exact_misses():
    calls = []
    vectors = {"clear the cache": np.array([1.0, 0.0]), "wipe the cache": np.array([0.96, 0.28]), "upgrade a tool": np.array([0.0, 1.0])}

    def embed(question):
        calls.append(question)
        return vectors[question]

    cache = ResponseCache(mode="semantic", threshold=0.95)
    cache.store("clear the cache", "s", "A", vector=vectors["clear the cache"])
    assert cache.lookup("clear the cache", "s", embed=embed)[0].kind == "exact" and calls == []
    hit, _ = cache.lookup("wipe the cache", "s", embed=embed)
    assert hit.kind == "semantic" and hit.similarity == pytest.approx(0.96) and hit.matched_question == "clear the cache"
    assert cache.lookup("upgrade a tool", "s", embed=embed)[0] is None


def test_cache_ttl_and_lru_eviction():
    now = [0.0]
    cache = ResponseCache(mode="exact", max_entries=2, ttl_seconds=10, clock=lambda: now[0])
    cache.store("a", "s", 1)
    cache.store("b", "s", 2)
    cache.lookup("a", "s")  # "a" becomes most recently used
    cache.store("c", "s", 3)  # evicts "b"
    assert cache.lookup("b", "s")[0] is None and cache.lookup("a", "s")[0] is not None
    now[0] = 11.0
    assert cache.lookup("a", "s")[0] is None  # expired


def test_pipeline_cache_hit_and_redacted_jsonl_traces(pipeline, test_settings):
    question = "How do I clear the cache? I'm jane.doe@example.com"
    first = pipeline.ask(question)
    second = pipeline.ask(question.upper())
    assert first["cache"] == "miss" and second["cache"] == "exact" and second["answer"] == first["answer"]
    assert first["status"] == "answered" and first["citations"][0]["chunk_id"].startswith("concepts/cache")
    raw_log = test_settings.trace_path.read_text()
    assert "jane.doe" not in raw_log.lower() and "[REDACTED_EMAIL]" in raw_log
    traces = read_traces(test_settings.trace_path)
    assert [t["cache"] for t in traces] == ["miss", "exact"]
    miss = traces[0]
    assert {"guard_input", "dense", "bm25", "fusion", "rerank", "llm", "validate", "total"} <= set(miss["timings_ms"])
    assert miss["retrieval"][0]["chunk_id"] and "rrf_score" in miss["retrieval"][0] and "rerank_score" in miss["retrieval"][0]
    assert miss["tokens"]["prompt"] > 0 and miss["selected"] and miss["index_version"] == pipeline.index.version
    json.dumps(traces)  # every trace is plain JSON


def test_rejected_input_is_traced(pipeline, test_settings):
    from rag_assistant.guardrails import InputRejected

    with pytest.raises(InputRejected):
        pipeline.ask("x" * (test_settings.max_question_chars + 1))
    assert read_traces(test_settings.trace_path)[-1]["status"] == "rejected"


def test_settings_from_env_reads_typed_values(monkeypatch, tmp_path):
    monkeypatch.setenv("RAG_TOP_K", "3")
    monkeypatch.setenv("RAG_RERANK", "false")
    monkeypatch.setenv("RAG_ALLOWED_ANSWER_DOMAINS", "astral.sh, python.org")
    settings = make_settings(tmp_path, tmp_path).from_env(data_dir=tmp_path)
    assert settings.top_k == 3 and settings.rerank is False and settings.allowed_answer_domains == ("astral.sh", "python.org")
    assert settings.index_dir == tmp_path / "index"
