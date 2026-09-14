import asyncio

import pytest
from mcp import Client

from conftest import APPROVAL_KEY, FEED
from research_agent import tools
from research_agent.approvals import sign
from research_agent.arxiv import ArxivClient, RateLimiter
from research_agent.corpus import PaperIndex
from research_agent.mcp_server import build_server
from research_agent.tools import ToolContext, ToolInputError

GOOD_REPORT = {"filename": "react-notes.md", "title": "ReAct notes", "content": "ReAct interleaves reasoning and acting (arXiv 2210.03629).",
               "citations": ["2210.03629"]}


@pytest.mark.parametrize("kwargs", [
    {},                                                    # nothing to search for
    {"query": "x" * 201},                                  # too long
    {"category": "evil.XX"},                               # archive not on the allowlist
    {"category": "cs.CL; DROP TABLE papers"},              # not a category
    {"author": "Robert'); DROP TABLE papers;--"},          # not a name
    {"query": "agents", "max_results": 0},
    {"query": "agents", "max_results": 11},
    {"query": "agents", "max_results": True},              # bool is not an int here
    {"query": "agents", "year_from": 1800},
    {"query": "agents", "year_from": 2021, "year_to": 2020},
])
def test_search_papers_rejects_bad_arguments(tool_ctx, kwargs):
    with pytest.raises(ToolInputError):
        tools.search_papers(tool_ctx, **kwargs)


def test_search_papers_returns_compact_results(tool_ctx):
    out = tools.search_papers(tool_ctx, query="reasoning and acting", max_results=2)
    top = out["results"][0]
    assert top["arxiv_id"] == "2210.03629" and "et al. (7 authors)" in top["authors"]
    assert len(top["snippet"]) <= 202 and out["source"] == "local arXiv snapshot"


def test_search_papers_matches_pasted_arxiv_ids_exactly(tool_ctx):
    assert [r["arxiv_id"] for r in tools.search_papers(tool_ctx, query="2210.03629v2")["results"]] == ["2210.03629"]
    assert tools.search_papers(tool_ctx, query="9999.99999")["results"] == []            # unknown id: no random fallback results
    mixed = tools.search_papers(tool_ctx, query="1706.03762 reasoning and acting", max_results=2)["results"]
    assert [r["arxiv_id"] for r in mixed] == ["1706.03762", "2210.03629"]


def test_get_paper_normalizes_and_validates(tool_ctx):
    paper = tools.get_paper(tool_ctx, "arXiv:2210.03629v1")
    assert paper["arxiv_id"] == "2210.03629" and paper["n_authors"] == 7 and paper["url"] == "https://arxiv.org/abs/2210.03629"
    for bad in ["../../etc/passwd", "9999.99999"]:
        with pytest.raises(ToolInputError):
            tools.get_paper(tool_ctx, bad)


def test_get_paper_falls_back_to_the_api_once_and_keeps_the_result(tmp_path):
    calls = []
    client = ArxivClient(tmp_path / "snapshots", opener=lambda url: calls.append(url) or FEED, limiter=RateLimiter(0))
    ctx = ToolContext(index=PaperIndex(tmp_path / "empty.sqlite"), reports_dir=tmp_path / "reports", arxiv=client)
    assert tools.get_paper(ctx, "1706.03762")["source"].startswith("arXiv API")
    assert tools.get_paper(ctx, "1706.03762")["source"] == "local arXiv snapshot"
    assert len(calls) == 1


def test_save_report_runs_only_with_a_valid_approval(tool_ctx):
    with pytest.raises(PermissionError):
        tools.save_report(tool_ctx, **GOOD_REPORT)
    token_for_other_args = sign(APPROVAL_KEY.encode(), "a1", "save_report", {**GOOD_REPORT, "content": "something else"})
    with pytest.raises(PermissionError):
        tools.save_report(tool_ctx, **GOOD_REPORT, approval_id="a1", approval_token=token_for_other_args)
    token = sign(APPROVAL_KEY.encode(), "a1", "save_report", GOOD_REPORT)
    saved = tools.save_report(tool_ctx, **GOOD_REPORT, approval_id="a1", approval_token=token)
    written = (tool_ctx.reports_dir / "react-notes.md").read_text()
    assert saved["saved"] == "react-notes.md" and "https://arxiv.org/abs/2210.03629" in written
    with pytest.raises(ToolInputError, match="already exists"):
        tools.save_report(tool_ctx, **GOOD_REPORT, approval_id="a1", approval_token=token)


@pytest.mark.parametrize("change", [
    {"filename": "../escape.md"}, {"filename": "notes.txt"}, {"filename": "/etc/cron.md"}, {"filename": "sub/dir.md"},
    {"content": "Log in at https://evil.example/login"}, {"citations": ["not-an-id"]}, {"content": "x" * 4001}, {"title": ""},
])
def test_report_arguments_are_validated(change):
    with pytest.raises(ToolInputError):
        tools.validate_report_arguments(**{**GOOD_REPORT, **change})


def _mcp(server, method, *args):
    async def go():
        async with Client(server) as client:
            return await getattr(client, method)(*args)

    return asyncio.run(go())


def test_mcp_server_lists_allowlisted_tools_with_annotations(tool_ctx):
    listed = {t.name: t for t in _mcp(build_server(tool_ctx), "list_tools").tools}
    assert list(listed) == ["search_papers", "get_paper", "save_report"]
    assert listed["search_papers"].annotations.read_only_hint and listed["get_paper"].annotations.read_only_hint
    assert listed["save_report"].annotations.read_only_hint is False
    assert listed["search_papers"].input_schema["properties"]["max_results"]["maximum"] == 10
    assert all(t.description for t in listed.values())
    restricted = _mcp(build_server(tool_ctx, enabled=("get_paper",)), "list_tools").tools
    assert [t.name for t in restricted] == ["get_paper"]


def test_mcp_tool_errors_come_back_as_results(tool_ctx):
    server = build_server(tool_ctx)
    bad_id = _mcp(server, "call_tool", "get_paper", {"arxiv_id": "../etc"})
    assert bad_id.is_error and "not a valid arXiv identifier" in bad_id.content[0].text
    unapproved = _mcp(server, "call_tool", "save_report", GOOD_REPORT)
    assert unapproved.is_error and "approval" in unapproved.content[0].text
    ok = _mcp(server, "call_tool", "get_paper", {"arxiv_id": "2210.03629"})
    assert not ok.is_error and ok.structured_content["title"].startswith("ReAct")
