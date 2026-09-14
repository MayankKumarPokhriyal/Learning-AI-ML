from conftest import FIXTURE_DOCS
from rag_assistant.chunking import chunk_document
from rag_assistant.parsing import clean_markdown, parse_document
from rag_assistant.tokens import count_tokens


def parse(text: str, doc_id: str = "doc"):
    return parse_document(text.encode(), doc_id=doc_id, url=f"https://example.com/{doc_id}", version="test")


def long_section(n_paragraphs: int = 12) -> str:
    paragraphs = [f"Paragraph {i}: " + " ".join(f"alpha{i}x{j}" for j in range(6)) + "." for i in range(n_paragraphs)]
    return "# Doc\n\n## Long section\n\n" + "\n\n".join(paragraphs) + "\n"


def test_mkdocs_front_matter_tabs_and_links_are_cleaned():
    text = clean_markdown(FIXTURE_DOCS["guides/install.md"])
    assert "title: Installing" not in text
    assert "macOS and Linux:" in text and '=== "' not in text
    assert "\n$ curl -LsSf https://astral.sh/uv/install.sh | sh\n" in text  # code block dedented and kept verbatim
    assert "See the installer reference for details." in text and "../reference" not in text


def test_admonition_becomes_a_label_with_dedented_body():
    assert "Note:\n\nCache entries for one package can be removed with `uv cache clean ruff`." in clean_markdown(FIXTURE_DOCS["concepts/cache.md"])


def test_links_with_brackets_or_line_breaks_keep_only_their_text():
    raw = "A dedicated [`[tool.uv.pip]`](../settings.md#pip) section and a [wrapped\nlink](https://example.com/x).\n"
    assert clean_markdown(raw) == "A dedicated `[tool.uv.pip]` section and a wrapped\nlink.\n"


def test_comment_lines_inside_code_blocks_are_not_headings():
    doc = parse("# Title\n\n```console\n# not a heading\n$ uv sync\n```\n\n## Real heading\n\ntext\n")
    assert [h.title for h in doc.headings] == ["Title", "Real heading"]


def test_chunks_respect_the_token_limit_and_lose_no_text():
    doc = parse(long_section())
    chunks = chunk_document(doc, max_tokens=120, overlap=30)
    assert len(chunks) > 1
    assert all(c.n_tokens <= 120 for c in chunks)
    assert all(any(f"Paragraph {i}:" in c.text for c in chunks) for i in range(12))
    assert all(doc.text[c.start:c.end] == c.text for c in chunks)  # offsets point at the real text


def test_overlap_is_honoured_and_bounded():
    doc = parse(long_section())
    with_overlap = chunk_document(doc, max_tokens=120, overlap=40)
    repeated = [doc.text[b.start:a.end] for a, b in zip(with_overlap, with_overlap[1:], strict=False) if b.start < a.end]
    assert repeated, "consecutive windows should repeat text when overlap > 0"
    assert all(count_tokens(text) <= 40 for text in repeated)
    no_overlap = chunk_document(doc, max_tokens=120, overlap=0)
    assert all(a.end <= b.start for a, b in zip(no_overlap, no_overlap[1:], strict=False))


def test_single_huge_line_is_split_into_token_windows():
    doc = parse("# Doc\n\n" + " ".join(f"token{i}" for i in range(800)) + "\n")
    chunks = chunk_document(doc, max_tokens=100, overlap=10)
    assert len(chunks) > 5 and all(c.n_tokens <= 100 for c in chunks)


def test_chunk_metadata_section_path_and_anchor():
    doc = parse(FIXTURE_DOCS["concepts/cache.md"], doc_id="concepts/cache")
    chunks = chunk_document(doc, max_tokens=300, overlap=50, min_tokens=0)
    assert chunks[0].chunk_id == "concepts/cache#0"
    cache_dir = next(c for c in chunks if "UV_CACHE_DIR" in c.text)
    assert cache_dir.section == "Cache directory"
    assert cache_dir.url.endswith("#cache-directory")
    assert cache_dir.indexed_text.startswith("Caching > Cache directory\n")


def test_tiny_sections_are_merged_with_a_neighbour():
    doc = parse(FIXTURE_DOCS["concepts/cache.md"], doc_id="concepts/cache")
    assert len(chunk_document(doc, max_tokens=300, overlap=50, min_tokens=50)) == 1
