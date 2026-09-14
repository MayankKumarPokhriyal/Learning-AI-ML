"""The agent's tools as plain, validated Python functions. `mcp_server.py` exposes them over MCP; tests call them directly.

Every argument is validated here (types, lengths, allowed values, safe file names) because tool arguments are
generated text: treat them exactly like untrusted user input.
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from research_agent import config
from research_agent.approvals import key_from_env, verify
from research_agent.arxiv import ArxivClient, ArxivError, SnapshotMissingError, normalize_arxiv_id
from research_agent.corpus import PaperIndex
from research_agent.guardrails import is_trusted_url


class ToolInputError(ValueError):
    """Bad arguments. The message goes back to the model so it can correct the call."""


class ToolUnavailableError(RuntimeError):
    """A dependency failed (for example the arXiv API is rate-limiting)."""


_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_AUTHOR_RE = re.compile(r"[^\W\d_][\w.' \-]{0,59}")
_CATEGORY_RE = re.compile(r"([a-z][a-z\-]*)(?:\.([A-Za-z][A-Za-z\-]*))?")
_FILENAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_\-]{0,60}\.md")
_URL_RE = re.compile(r"https?://[^\s)<>\]\"']+")
_ID_IN_QUERY = re.compile(r"(?<![\d.])\d{4}\.\d{4,5}(?:v\d+)?(?![\d])")


@dataclass
class ToolContext:
    index: PaperIndex
    reports_dir: Path
    arxiv: ArxivClient | None = None  # live fallback for papers missing from the snapshot (None = offline only)
    approval_key: bytes | None = None  # shared with the host; without it save_report is disabled

    @classmethod
    def from_env(cls) -> ToolContext:
        live = os.environ.get("ARXIV_LIVE_FALLBACK", "1") != "0"
        arxiv = ArxivClient(config.snapshot_dir(), mode=config.arxiv_mode(), max_retries=1, backoff_s=10, timeout_s=90) if live else None
        return cls(index=PaperIndex(config.corpus_path()), reports_dir=config.reports_dir(), arxiv=arxiv, approval_key=key_from_env())


def _clean_text(value: Any, name: str, max_len: int, *, required: bool = False, multiline: bool = False) -> str:
    text = "" if value is None else str(value)
    text = _CONTROL_RE.sub(" ", text) if multiline else re.sub(r"\s+", " ", _CONTROL_RE.sub(" ", text))
    text = text.strip()
    if required and not text:
        raise ToolInputError(f"{name} must not be empty")
    if len(text) > max_len:
        raise ToolInputError(f"{name} is too long ({len(text)} characters, maximum {max_len})")
    return text


def _int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ToolInputError(f"{name} must be an integer")
    return value


def validate_category(category: str) -> str:
    match = _CATEGORY_RE.fullmatch(category)
    if not match or match.group(1) not in config.ALLOWED_ARCHIVES:
        raise ToolInputError(f"unknown arXiv category {category!r}; use one like cs.CL, cs.LG, cs.AI, cs.CR, or stat.ML")
    return category


def _short_authors(authors: tuple[str, ...]) -> str:
    return ", ".join(authors[:3]) + (f" et al. ({len(authors)} authors)" if len(authors) > 3 else "")


def _snippet(text: str, limit: int = 200) -> str:
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + " …"


def search_papers(ctx: ToolContext, query: str = "", author: str = "", category: str = "", year_from: int = 0,
                  year_to: int = 0, max_results: int = 5) -> dict[str, Any]:
    query = _clean_text(query, "query", 200)
    author = _clean_text(author, "author", 60)
    category = _clean_text(category, "category", 24)
    year_from, year_to, max_results = _int(year_from, "year_from"), _int(year_to, "year_to"), _int(max_results, "max_results")
    if not (query or author or category):
        raise ToolInputError("give at least one of: query, author, category")
    if author and not _AUTHOR_RE.fullmatch(author):
        raise ToolInputError("author must be a person's name (letters, spaces, '.', '-', \"'\")")
    if category:
        validate_category(category)
    for name, year in (("year_from", year_from), ("year_to", year_to)):
        if year and not 1991 <= year <= 2100:
            raise ToolInputError(f"{name} must be between 1991 and 2100, or 0 for no limit")
    if year_from and year_to and year_from > year_to:
        raise ToolInputError("year_from must not be after year_to")
    if not 1 <= max_results <= 10:
        raise ToolInputError("max_results must be between 1 and 10")
    # users (and models) often paste ids: exact id matches come first, the remaining words go to full-text search
    exact = [p for p in (ctx.index.get(normalize_arxiv_id(m.group(0))) for m in _ID_IN_QUERY.finditer(query)) if p is not None]
    rest = _ID_IN_QUERY.sub(" ", query).strip()
    papers = ctx.index.search(rest, author=author or None, category=category or None, year_from=year_from or None,
                              year_to=year_to or None, limit=max_results) if (rest or author or category) else []
    seen = {p.arxiv_id for p in exact}
    papers = (list({p.arxiv_id: p for p in exact}.values()) + [p for p in papers if p.arxiv_id not in seen])[:max_results]
    return {"results": [{"arxiv_id": p.arxiv_id, "title": p.title, "authors": _short_authors(p.authors), "published": p.published,
                         "primary_category": p.primary_category, "snippet": _snippet(p.abstract)} for p in papers],
            "n_results": len(papers), "source": "local arXiv snapshot"}


def get_paper(ctx: ToolContext, arxiv_id: str) -> dict[str, Any]:
    try:
        normalized = normalize_arxiv_id(_clean_text(arxiv_id, "arxiv_id", 60, required=True))
    except ValueError as exc:
        raise ToolInputError(str(exc)) from exc
    paper, origin = ctx.index.get(normalized), "local arXiv snapshot"
    if paper is None and ctx.arxiv is not None:
        try:
            fetched = ctx.arxiv.get_by_ids([normalized], max_results=1)
        except SnapshotMissingError as exc:
            raise ToolInputError(f"no paper {normalized} in the local snapshot (offline mode)") from exc
        except ArxivError as exc:
            raise ToolUnavailableError(f"{normalized} is not in the local snapshot and the arXiv API failed "
                                       f"({type(exc).__name__}); try again later") from exc
        if fetched and fetched[0].arxiv_id == normalized:
            ctx.index.upsert(fetched[:1], source="arxiv-api:get_paper")
            paper, origin = fetched[0], "arXiv API (live, now added to the local snapshot)"
    if paper is None:
        raise ToolInputError(f"no arXiv paper with id {normalized}")
    truncated = len(paper.abstract) > config.MAX_ABSTRACT_CHARS
    return {"arxiv_id": paper.arxiv_id, "latest_version": paper.version, "title": paper.title, "authors": list(paper.authors[:25]),
            "n_authors": len(paper.authors), "first_version_date": paper.published, "latest_version_date": paper.updated,
            "primary_category": paper.primary_category, "categories": list(paper.categories),
            "abstract": paper.abstract[: config.MAX_ABSTRACT_CHARS] + (" …" if truncated else ""),
            "doi": paper.doi, "journal_ref": paper.journal_ref, "url": f"https://arxiv.org/abs/{paper.arxiv_id}", "source": origin}


def validate_report_arguments(filename: str, title: str, content: str, citations: list[str]) -> dict[str, Any]:
    """Business rules for save_report, shared by the host (before asking a human) and the server (before writing)."""
    filename = _clean_text(filename, "filename", 64, required=True)
    if not _FILENAME_RE.fullmatch(filename):
        raise ToolInputError("filename must be a plain name like 'agent-papers.md' (letters, digits, '-', '_'; ends in .md; no folders)")
    title = _clean_text(title, "title", 120, required=True)
    content = _clean_text(content, "content", config.MAX_REPORT_CHARS, required=True, multiline=True)
    if not isinstance(citations, list) or len(citations) > 20:
        raise ToolInputError("citations must be a list of at most 20 arXiv ids")
    try:
        ids = list(dict.fromkeys(normalize_arxiv_id(c) for c in citations))
    except ValueError as exc:
        raise ToolInputError(f"citations: {exc}") from exc
    untrusted = [url for url in _URL_RE.findall(f"{title} {content}") if not is_trusted_url(url)]
    if untrusted:
        raise ToolInputError(f"reports may only link to {', '.join(config.TRUSTED_LINK_DOMAINS)}; found {untrusted[0][:60]}")
    return {"filename": filename, "title": title, "content": content, "citations": ids}


def save_report(ctx: ToolContext, filename: str, title: str, content: str, citations: list[str],
                approval_id: str = "", approval_token: str = "") -> dict[str, Any]:
    arguments = {"filename": filename, "title": title, "content": content, "citations": citations}
    if not verify(ctx.approval_key, approval_id, "save_report", arguments, approval_token):
        raise PermissionError("save_report requires a human approval: the host must pass a valid approval_id and approval_token")
    args = validate_report_arguments(**arguments)
    ctx.reports_dir.mkdir(parents=True, exist_ok=True)
    target = (ctx.reports_dir / args["filename"]).resolve()
    if target.parent != ctx.reports_dir.resolve():
        raise ToolInputError("the report must stay inside the reports folder")
    sources = "".join(f"- [arXiv:{i}](https://arxiv.org/abs/{i})\n" for i in args["citations"])
    body = f"# {args['title']}\n\n{args['content']}\n\n## Sources\n{sources}\n_Saved by research-agent after human approval {approval_id}._\n"
    try:
        with target.open("x", encoding="utf-8") as f:  # "x": never overwrite an existing report
            f.write(body)
    except FileExistsError as exc:
        raise ToolInputError(f"a report named {args['filename']} already exists; choose another filename") from exc
    return {"saved": args["filename"], "bytes": len(body.encode()), "sha256": hashlib.sha256(body.encode()).hexdigest()[:16],
            "approval_id": approval_id}
