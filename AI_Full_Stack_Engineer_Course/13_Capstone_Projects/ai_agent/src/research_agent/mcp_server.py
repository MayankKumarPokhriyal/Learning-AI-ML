"""The project's MCP server (stdio): `python -m research_agent.mcp_server`.

Type hints + `Field` descriptions become each tool's JSON Schema; docstrings become descriptions. Only tools on the
allowlist are registered, and every call goes through the validated functions in `tools.py`.
"""

from __future__ import annotations

from typing import Annotated, Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import Field

from research_agent import config, tools
from research_agent.tools import ToolContext, ToolInputError, ToolUnavailableError

READ_ONLY = ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False)
WRITES_FILES = ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=False)


def build_server(ctx: ToolContext | None = None, enabled: tuple[str, ...] = config.TOOL_ALLOWLIST) -> MCPServer:
    ctx = ctx or ToolContext.from_env()
    server = MCPServer("research-agent-tools", instructions="Search and read a local snapshot of arXiv papers; save human-approved reports.",
                       log_level="WARNING")

    def call(fn, **kwargs) -> dict[str, Any]:
        try:
            return fn(ctx, **kwargs)
        except (ToolInputError, ToolUnavailableError, PermissionError) as exc:
            raise ToolError(str(exc)) from exc  # becomes an is_error result the model can read

    if "search_papers" in enabled:
        @server.tool(annotations=READ_ONLY)
        def search_papers(
            query: Annotated[str, Field(description="Words from the title or abstract, e.g. 'reasoning and acting in language models'")] = "",
            author: Annotated[str, Field(description="An author's surname, e.g. 'Vaswani'")] = "",
            category: Annotated[str, Field(description="Optional arXiv category, e.g. 'cs.CL'")] = "",
            year_from: Annotated[int, Field(description="Earliest first-version year; 0 means no limit")] = 0,
            year_to: Annotated[int, Field(description="Latest first-version year; 0 means no limit")] = 0,
            max_results: Annotated[int, Field(ge=1, le=10, description="Number of results, 1-10")] = 5,
        ) -> dict[str, Any]:
            """Search a local snapshot of arXiv papers (BM25 over titles, abstracts, and author names; arXiv ids in the query
            are matched exactly first). Returns arXiv ids, titles, authors, first-version dates, primary categories, and short
            snippets. Use get_paper for a full record."""
            return call(tools.search_papers, query=query, author=author, category=category, year_from=year_from,
                        year_to=year_to, max_results=max_results)

    if "get_paper" in enabled:
        @server.tool(annotations=READ_ONLY)
        def get_paper(arxiv_id: Annotated[str, Field(description="arXiv identifier, e.g. '2210.03629'")]) -> dict[str, Any]:
            """Get one arXiv paper by id: title, all authors, first- and latest-version dates, categories, abstract, DOI."""
            return call(tools.get_paper, arxiv_id=arxiv_id)

    if "save_report" in enabled:
        @server.tool(annotations=WRITES_FILES)
        def save_report(
            filename: Annotated[str, Field(description="Plain file name ending in .md, e.g. 'agent-papers.md'")],
            title: Annotated[str, Field(description="Report title")],
            content: Annotated[str, Field(description="Markdown body, at most 4000 characters")],
            citations: Annotated[list[str], Field(description="arXiv ids the report cites")],
            approval_id: str = "",
            approval_token: str = "",
        ) -> dict[str, Any]:
            """Save a short markdown report to the reports folder. Has a side effect, so it only runs after a human
            approves it (the host adds approval_id and approval_token)."""
            return call(tools.save_report, filename=filename, title=title, content=content, citations=citations,
                        approval_id=approval_id, approval_token=approval_token)

    return server


def main() -> None:
    build_server().run()  # stdio transport


if __name__ == "__main__":
    main()
