"""Structure-aware chunking: one chunk per heading section; long sections are split at paragraph (then line, then token)
boundaries into windows of at most `max_tokens`, where consecutive windows repeat up to `overlap` tokens."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from rag_assistant.parsing import ParsedDocument, fence_step
from rag_assistant.tokens import count_tokens, encoding


@dataclass
class Chunk:
    chunk_id: str  # "<doc_id>#<n>", e.g. "concepts/cache#3" — what the LLM cites
    doc_id: str
    title: str  # document title
    section: str  # heading path below the title, e.g. "Cache directory"
    url: str  # source URL with the section anchor
    version: str  # release tag of the source
    start: int  # character offsets in the parsed document text
    end: int
    text: str
    n_tokens: int
    flags: list[str] = field(default_factory=list)  # guardrail findings, e.g. ["override_instructions"]

    @property
    def header(self) -> str:
        return f"{self.title} > {self.section}" if self.section else self.title

    @property
    def indexed_text(self) -> str:
        """What gets embedded and BM25-indexed: a context header plus the text."""
        return f"{self.header}\n{self.text}"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> Chunk:
        return cls(**data)


def section_spans(doc: ParsedDocument) -> list[tuple[int, int, str, str]]:
    """(start, end, heading path, anchor) for the text before the first heading and for every heading section."""
    marks, path = [(0, "", "")], {}
    for h in doc.headings:
        path = {level: title for level, title in path.items() if level < h.level}
        path[h.level] = h.title
        section = " > ".join(title for level, title in sorted(path.items()) if level > 1)
        marks.append((h.start, section, h.anchor if h.level > 1 else ""))
    ends = [start for start, _, _ in marks[1:]] + [len(doc.text)]
    return [(start, end, section, anchor) for (start, section, anchor), end in zip(marks, ends, strict=True) if doc.text[start:end].strip()]


def block_spans(text: str, start: int, end: int) -> list[tuple[int, int]]:
    """Paragraph blocks separated by blank lines; a fenced code block stays one block even if it contains blank lines."""
    blocks, block_start, block_end, fence, pos = [], None, None, None, start
    for line in text[start:end].split("\n"):
        line_end = pos + len(line)
        _, fence_after = fence_step(line.lstrip(" "), fence)
        if not line.strip() and fence is None:
            if block_start is not None:
                blocks.append((block_start, block_end))
                block_start = None
        else:
            if block_start is None:
                block_start = pos
            block_end = line_end
        fence = fence_after
        pos = line_end + 1
    if block_start is not None:
        blocks.append((block_start, block_end))
    return blocks


def _line_spans(text: str, start: int, end: int) -> list[tuple[int, int]]:
    spans, pos = [], start
    for line in text[start:end].split("\n"):
        if line.strip():
            spans.append((pos, pos + len(line)))
        pos += len(line) + 1
    return spans


def token_windows(text: str, start: int, end: int, max_tokens: int, overlap: int) -> list[tuple[int, int]]:
    """Last resort for a single huge line: sliding windows over LLM tokens, returned as character spans."""
    piece = text[start:end]
    tokens = encoding().encode(piece, disallowed_special=())
    _, offsets = encoding().decode_with_offsets(tokens)
    offsets = list(offsets) + [len(piece)]
    spans = []
    for s in range(0, len(tokens), max(1, max_tokens - overlap)):
        e = min(s + max_tokens, len(tokens))
        spans.append((start + offsets[s], start + offsets[e]))
        if e == len(tokens):
            break
    return spans


def _fit_blocks(text: str, blocks: list[tuple[int, int]], max_tokens: int, overlap: int) -> list[tuple[int, int]]:
    """Make every block fit: split an oversized block into lines, and an oversized line into token windows."""
    fitted = []
    for s, e in blocks:
        if count_tokens(text[s:e]) <= max_tokens:
            fitted.append((s, e))
            continue
        lines = _line_spans(text, s, e)
        if len(lines) > 1:
            fitted.extend(_fit_blocks(text, lines, max_tokens, overlap))
        else:
            fitted.extend(token_windows(text, s, e, max_tokens, overlap))
    return fitted


def window_spans(text: str, blocks: list[tuple[int, int]], max_tokens: int, overlap: int) -> list[tuple[int, int]]:
    """Greedily pack consecutive blocks into windows of <= max_tokens. The next window starts early enough to repeat
    the previous window's last blocks, as long as the repeated part is <= overlap tokens (so facts on a boundary survive)."""
    spans, i, n = [], 0, len(blocks)
    while i < n:
        j = i
        while j + 1 < n and count_tokens(text[blocks[i][0]:blocks[j + 1][1]]) <= max_tokens:
            j += 1
        spans.append((blocks[i][0], blocks[j][1]))
        if j == n - 1:
            break
        nxt = j + 1
        for k in range(j, i, -1):
            repeated_fits = count_tokens(text[blocks[k][0]:blocks[j][1]]) <= overlap
            leaves_room = count_tokens(text[blocks[k][0]:blocks[j + 1][1]]) <= max_tokens
            if not (repeated_fits and leaves_room):
                break
            nxt = k
        i = nxt
    return spans


def chunk_document(doc: ParsedDocument, max_tokens: int = 300, overlap: int = 50, min_tokens: int = 50) -> list[Chunk]:
    if not 0 <= overlap < max_tokens:
        raise ValueError("overlap must be >= 0 and smaller than max_tokens")
    units: list[list] = []
    for start, end, section, anchor in section_spans(doc):
        if count_tokens(doc.text[start:end]) <= max_tokens:
            units.append([start, end, section, anchor])
        else:
            blocks = _fit_blocks(doc.text, block_spans(doc.text, start, end), max_tokens, overlap)
            units.extend([s, e, section, anchor] for s, e in window_spans(doc.text, blocks, max_tokens, overlap))
    merged: list[list] = []
    for unit in units:  # glue a tiny unit (e.g. a heading with one sentence) to its neighbour when both fit
        if merged:
            prev = merged[-1]
            tiny = count_tokens(doc.text[prev[0]:prev[1]]) < min_tokens or count_tokens(doc.text[unit[0]:unit[1]]) < min_tokens
            if prev[1] <= unit[0] and tiny and count_tokens(doc.text[prev[0]:unit[1]]) <= max_tokens:
                prev[1] = unit[1]
                continue
        merged.append(unit)
    chunks = []
    for start, end, section, anchor in merged:
        while start < end and doc.text[start].isspace():
            start += 1
        while end > start and doc.text[end - 1].isspace():
            end -= 1
        if start == end:
            continue
        text = doc.text[start:end]
        chunks.append(Chunk(chunk_id=f"{doc.doc_id}#{len(chunks)}", doc_id=doc.doc_id, title=doc.title, section=section,
                            url=f"{doc.url}#{anchor}" if anchor else doc.url, version=doc.version, start=start, end=end,
                            text=text, n_tokens=count_tokens(text)))
    return chunks
