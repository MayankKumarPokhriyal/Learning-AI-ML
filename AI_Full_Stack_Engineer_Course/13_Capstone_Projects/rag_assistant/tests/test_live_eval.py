"""Live smoke test: the real embedding model, reranker and index, and a real LLM server. Skipped when the index has not been
built or no LLM server is reachable, so the fast suite still runs anywhere (for example in CI without a GPU)."""

import pytest

from rag_assistant import config as cfg
from rag_assistant.config import Settings
from rag_assistant.evaluation import evaluate, load_questions
from rag_assistant.index import IndexStore
from rag_assistant.llm import OpenAICompatibleLLM

pytestmark = pytest.mark.live


@pytest.fixture(scope="module")
def live_pipeline(tmp_path_factory):
    settings = Settings.from_env(cache_mode="off", trace_path=tmp_path_factory.mktemp("traces") / "live.jsonl")
    if IndexStore(settings.index_dir).current_version() is None:
        pytest.skip("no index: run `python -m rag_assistant download` and `python -m rag_assistant ingest` first")
    llm = OpenAICompatibleLLM(settings.llm_base_url, settings.llm_model, settings.llm_api_key, reasoning_effort=settings.reasoning_effort,
                              record_dir=settings.llm_cache_dir)
    if not llm.ping():
        pytest.skip(f"no LLM server reachable at {settings.llm_base_url}")
    from rag_assistant.pipeline import RAGPipeline

    pipeline = RAGPipeline.from_settings(settings, llm=llm)
    yield pipeline
    pipeline.release_models()


def test_live_smoke_evaluation(live_pipeline):
    questions = load_questions(cfg.PROJECT_DIR / "eval" / "questions.jsonl")
    sample = [q for q in questions if q["kind"] == "answerable"][:3] + [q for q in questions if q["kind"] == "unanswerable"][:1]
    report = evaluate(live_pipeline, sample, name="live smoke")
    summary, rows = report["summary"], report["rows"]
    assert summary["recall@5"] >= 2 / 3  # retrieval is deterministic on the pinned corpus
    assert summary["n_invalid_output"] == 0  # structured output must always parse
    assert all(r["status"] in {"answered", "refused", "unsupported", "blocked_output"} for r in rows)
    assert sum(r["rule_correct"] for r in rows) >= 2  # lenient on purpose: generated text varies between runs
