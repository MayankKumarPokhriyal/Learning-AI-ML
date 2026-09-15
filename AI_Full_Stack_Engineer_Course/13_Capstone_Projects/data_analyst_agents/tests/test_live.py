"""A LIVE smoke test: the real LLM server (LLM_BASE_URL) and the downloaded BIRD databases. Skipped when either is missing.

LLM output varies between runs, so it checks that runs finish with an executable query; the evaluation measures accuracy.
"""

import asyncio
import urllib.request

import pytest

from data_analyst_agents import config
from data_analyst_agents.config import LLMSettings
from data_analyst_agents.llm import OpenAICompatibleLLM
from data_analyst_agents.pipeline import AnalystTeam, TeamSettings
from data_analyst_agents.questions import load_eval_questions
from data_analyst_agents.schema import load_catalogs


def llm_server_reachable() -> bool:
    settings = LLMSettings.from_env()
    request = urllib.request.Request(settings.base_url.rstrip("/") + "/models", headers={"Authorization": f"Bearer {settings.api_key}"})
    try:
        with urllib.request.urlopen(request, timeout=3) as response:
            return response.status == 200
    except OSError:
        return False


pytestmark = [pytest.mark.live, pytest.mark.skipif(not all(config.db_path(d).is_file() for d in config.DATABASES), reason="BIRD data missing: run `make data`"),
              pytest.mark.skipif(not llm_server_reachable(), reason="no OpenAI-compatible LLM server reachable at LLM_BASE_URL")]


def test_live_team_answers_with_an_executable_query():
    catalogs = load_catalogs()
    team = AnalystTeam(OpenAICompatibleLLM(), catalogs, settings=TeamSettings(profile="answer", approval_mode="record"))
    questions = load_eval_questions()[:2]

    async def run_all():  # asyncio.run needs a coroutine; gather() returns a future
        return await asyncio.gather(*(team.run(q["question"], q["evidence"]) for q in questions))

    outcomes = asyncio.run(run_all())
    assert [o.status for o in outcomes] == ["completed"] * len(questions), [(o.status, o.error) for o in outcomes]
    assert all(o.result_obj is not None and o.result_obj.ok for o in outcomes)
