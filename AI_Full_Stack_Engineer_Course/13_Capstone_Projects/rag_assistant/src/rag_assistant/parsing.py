"""Turn uv's MkDocs Markdown into clean text that keeps its structure (headings, lists, code blocks)."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

PARSER_VERSION = "1"  # bump when cleaning changes: the index re-parses every document

FRONT_MATTER = re.compile(r"\A---\n.*?\n---\n", re.S)
HTML_COMMENT = re.compile(r"<!--.*?-->", re.S)
ADMONITION = re.compile(r'^(?:!!!|\?\?\?\+?)\s*([A-Za-z-]+)(?:\s+"([^"]*)")?\s*$')  # !!! note "Title"
TAB = re.compile(r'^===\+?\s*"([^"]*)"\s*$')  # === "macOS and Linux"
FENCE = re.compile(r"^(`{3,}|~{3,})")
IMAGE = re.compile(r"!\[[^\]]*\]\((?:[^()\s]|\([^)\s]*\))*\)")
LINK_TEXT = r"\[((?:[^\[\]]|\[[^\[\]]*\])+)\]"  # allows one level of brackets, e.g. [`[tool.uv]`](...), and line breaks
INLINE_LINK = re.compile(LINK_TEXT + r"\((?:[^()\s]|\([^)\s]*\))*\)")
REF_LINK = re.compile(LINK_TEXT + r"\[[^\]\n]*\]")
REF_DEFINITION = re.compile(r"^[ ]{0,3}\[[^\]\n]+\]:[ \t]*\n?[ \t]*\S+[^\n]*$", re.M)
HTML_TAG = re.compile(r"</?(?:div|span|p|br|img|a|details|summary|sup|sub|kbd|small|em|strong|table|tr|td|th|thead|tbody)\b[^>]*>", re.I)
ATTR_LIST = re.compile(r"\{:[^}\n]*\}|\{\s*[#.][\w-][^}\n]*\}")
HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")


@dataclass(frozen=True)
class Heading:
    start: int  # character offset of the heading line in the cleaned text
    level: int
    title: str
    anchor: str


@dataclass
class ParsedDocument:
    doc_id: str
    title: str
    url: str
    version: str
    sha256: str  # of the raw file bytes
    text: str
    headings: list[Heading]


def fence_step(body: str, fence: str | None) -> tuple[bool, str | None]:
    """Track fenced code blocks: returns (this line is a fence marker, the fence still open after this line)."""
    m = FENCE.match(body)
    if not m:
        return False, fence
    if fence is None:
        return True, m.group(1)
    if set(body.strip()) == {fence[0]} and len(body.strip()) >= len(fence):
        return True, None
    return False, fence


def _dedent_containers(lines: list[str]) -> list[str]:
    """Admonitions (`!!! note`) and content tabs (`=== "Linux"`) indent their body by 4 spaces. Replace each marker with a
    plain label ("Note:", "Linux:") and remove the extra indentation so code blocks and lists inside are recognised."""
    out, stack, fence = [], [], None
    for line in lines:
        stripped = line.lstrip(" ")
        indent = len(line) - len(stripped)
        if not stripped:
            out.append("")
            continue
        if fence is None:
            while stack and indent < stack[-1]:
                stack.pop()
        content = line[min(indent, stack[-1] if stack else 0):]
        body = content.lstrip(" ")
        if fence is None:
            if m := ADMONITION.match(body):
                label = m.group(1).capitalize()
                out.append(f"{label}: {m.group(2)}" if m.group(2) else f"{label}:")
                stack.append(indent + 4)
                continue
            if m := TAB.match(body):
                out.append(f"{m.group(1)}:")
                stack.append(indent + 4)
                continue
        _, fence = fence_step(body, fence)
        out.append(content)
    return out


def _clean_prose(text: str) -> str:
    text = REF_DEFINITION.sub("", text)
    text = IMAGE.sub("", text)
    text = INLINE_LINK.sub(r"\1", text)  # keep the link text, drop the URL (sources are tracked per document)
    text = REF_LINK.sub(r"\1", text)
    text = HTML_TAG.sub("", text)
    return ATTR_LIST.sub("", text)


def clean_markdown(raw: str) -> str:
    text = raw.replace("\r\n", "\n").replace("\t", "    ")
    text = FRONT_MATTER.sub("", text)
    text = HTML_COMMENT.sub("", text)
    out, prose, fence = [], [], None
    for line in _dedent_containers(text.split("\n")):
        is_marker, fence_after = fence_step(line.lstrip(" "), fence)
        if is_marker or fence is not None:  # code stays exactly as written
            if prose:
                out.append(_clean_prose("\n".join(prose)))
                prose = []
            out.append(line)
        else:
            prose.append(line)  # prose is cleaned in runs, so links that wrap across lines are handled
        fence = fence_after
    if prose:
        out.append(_clean_prose("\n".join(prose)))
    text = "\n".join(line.rstrip() for line in "\n".join(out).split("\n"))
    return re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"


def slugify(title: str) -> str:
    """Heading anchor in the style of Python-Markdown's table of contents."""
    return re.sub(r"\s+", "-", re.sub(r"[^\w\s-]", "", title.lower()).strip())


def find_headings(text: str) -> list[Heading]:
    headings, fence, offset = [], None, 0
    for line in text.split("\n"):
        is_marker, fence_after = fence_step(line.lstrip(" "), fence)
        if fence is None and not is_marker and (m := HEADING.match(line)):
            title = m.group(2).strip()
            headings.append(Heading(offset, len(m.group(1)), title, slugify(title)))
        fence = fence_after
        offset += len(line) + 1
    return headings


def parse_document(raw: bytes, *, doc_id: str, url: str, version: str) -> ParsedDocument:
    text = clean_markdown(raw.decode("utf-8"))
    headings = find_headings(text)
    title = next((h.title for h in headings if h.level == 1), doc_id)
    return ParsedDocument(doc_id, title, url, version, hashlib.sha256(raw).hexdigest(), text, headings)
