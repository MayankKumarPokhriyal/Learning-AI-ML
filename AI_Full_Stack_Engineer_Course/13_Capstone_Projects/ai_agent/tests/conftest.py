"""Shared fixtures.

Unit and API tests never call a language model: they use `StubLLM`, a scripted DETERMINISTIC TEST DOUBLE, so the loop,
budgets, approvals, and guardrails are tested exactly and offline. Its outputs are not model results.
The paper data is `fixtures/arxiv_two_papers.xml`: a real arXiv API response (id_list query, fetched 2026-09-14),
trimmed to two entries.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from mcp import Client

from research_agent.agent import MCPToolBackend, ResearchAgent
from research_agent.arxiv import parse_feed
from research_agent.corpus import PaperIndex
from research_agent.mcp_server import build_server
from research_agent.store import StateStore
from research_agent.tools import ToolContext

FIXTURES = Path(__file__).parent / "fixtures"
FEED = (FIXTURES / "arxiv_two_papers.xml").read_bytes()
APPROVAL_KEY = "test-signing-key-0123456789abcdef0123456789abcdef"


@pytest.fixture(autouse=True)
def isolated_environment(tmp_path, monkeypatch):
    """Every test gets its own data folder; settings from the developer's shell can't leak in."""
    monkeypatch.setenv("AGENT_DATA_DIR", str(tmp_path / "data"))
    for name in ("AGENT_CORPUS_PATH", "AGENT_REPORTS_DIR", "LLM_CACHE_PATH", "REVIEWER_TOKEN", "APPROVAL_SIGNING_KEY", "ARXIV_MODE"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def fixture_papers():
    return parse_feed(FEED)[0]


@pytest.fixture
def index(tmp_path, fixture_papers) -> PaperIndex:
    corpus = PaperIndex(tmp_path / "papers.sqlite")
    corpus.upsert(fixture_papers, source="test-fixture")
    return corpus


@pytest.fixture
def tool_ctx(index, tmp_path) -> ToolContext:
    return ToolContext(index=index, reports_dir=tmp_path / "reports", arxiv=None, approval_key=APPROVAL_KEY.encode())


@pytest.fixture
def with_agent(tool_ctx, tmp_path):
    """Run `scenario(agent)` against an in-memory MCP server (real MCP protocol, no subprocess)."""

    def run(llm, scenario, **agent_kwargs):
        async def go():
            async with Client(build_server(tool_ctx)) as client:
                agent = ResearchAgent(llm, MCPToolBackend(client), StateStore(tmp_path / "state.sqlite"), traces_dir=tmp_path / "traces",
                                      approval_key=APPROVAL_KEY.encode(), **agent_kwargs)
                return await scenario(agent)

        return asyncio.run(go())

    return run
