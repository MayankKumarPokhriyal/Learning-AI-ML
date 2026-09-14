"""A small LIVE evaluation against a real OpenAI-compatible LLM server (LLM_BASE_URL). Skipped when none is reachable.

It uses the fixture corpus (two real arXiv records), so it needs no network access besides the LLM server.
LLM output varies between runs, so it asserts a minimum pass count rather than exact answers.
"""

import asyncio
import urllib.request

import pytest

from research_agent.agent import open_agent
from research_agent.config import LLMSettings
from research_agent.evaluate import load_tasks, run_tasks

LIVE_TASK_IDS = ("T01", "T08", "T19")  # answerable from the two fixture papers


def llm_server_reachable() -> bool:
    settings = LLMSettings.from_env()
    request = urllib.request.Request(settings.base_url.rstrip("/") + "/models", headers={"Authorization": f"Bearer {settings.api_key}"})
    try:
        with urllib.request.urlopen(request, timeout=3) as response:
            return response.status == 200
    except OSError:
        return False


pytestmark = [pytest.mark.live, pytest.mark.skipif(not llm_server_reachable(), reason="no OpenAI-compatible LLM server reachable at LLM_BASE_URL")]


def test_live_agent_passes_fixture_tasks(index, tmp_path):
    tasks = [t for t in load_tasks() if t["id"] in LIVE_TASK_IDS]

    async def go():
        env = {"AGENT_CORPUS_PATH": str(index.path), "ARXIV_LIVE_FALLBACK": "0", "AGENT_REPORTS_DIR": str(tmp_path / "reports")}
        async with open_agent(arxiv_mode="replay", extra_env=env) as agent:
            return await run_tasks(agent, tasks, concurrency=2)

    rows = asyncio.run(go())
    assert [r["status"] for r in rows] == ["completed"] * len(tasks)
    assert sum(r["passed"] for r in rows) >= 2, [(r["id"], r["answer"], r["failed_checks"]) for r in rows]
