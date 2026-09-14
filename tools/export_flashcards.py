#!/usr/bin/env python3
"""Export every interview question in the course as flashcards you can import into Anki.

    python tools/export_flashcards.py                      # writes flashcards/ai_course_interview_flashcards.csv
    python tools/export_flashcards.py --include-quiz       # also adds the 🧪 Quick Quiz questions
    python tools/export_flashcards.py --out my_cards.csv

Anki: File → Import → choose the CSV. The header lines set the separator, HTML, and the tags column,
so each card is tagged with its module and notebook (e.g. 03_Classical_Machine_Learning::04_Logistic_Regression).
"""
import argparse
import csv
import html
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
COURSE = REPO / "AI_Full_Stack_Engineer_Course"

QA_RE = re.compile(
    r"\*\*Q(?P<num>\d+)\.\s*(?P<q>.+?)\*\*\s*"          # **Q3. question**
    r"<details>\s*<summary>.*?</summary>(?P<a>.*?)</details>",
    re.S,
)
QUIZ_RE = re.compile(
    r"^\*\*(?P<num>\d+)\.\*\*\s*(?P<q>.+?)\s*"            # **1.** question
    r"<details>\s*<summary>.*?</summary>(?P<a>.*?)</details>",
    re.S | re.M,
)


def md_to_html(text: str) -> str:
    """Tiny, safe markdown → HTML for flashcards (bold, italics, inline code, code blocks, line breaks)."""
    text = text.strip()
    blocks = []

    def keep_code(m):
        blocks.append("<pre><code>" + html.escape(m.group(2).strip("\n")) + "</code></pre>")
        return f"\x00{len(blocks) - 1}\x00"

    text = re.sub(r"```(\w*)\n(.*?)```", keep_code, text, flags=re.S)
    text = html.escape(text)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<i>\1</i>", text)
    text = text.replace("\n", "<br>")
    return re.sub(r"\x00(\d+)\x00", lambda m: blocks[int(m.group(1))], text)


def section(md_cells, start_heading, stop_prefix="## "):
    """Concatenate markdown from a `## start_heading` until the next `## ` heading."""
    out, inside = [], False
    for s in md_cells:
        for line in s.splitlines(keepends=True):
            if line.startswith("## "):
                inside = line[3:].strip().startswith(start_heading)
                if not inside:
                    continue
            if inside:
                out.append(line)
    return "".join(out)


def cards_for(notebook: Path, include_quiz: bool):
    cells = json.loads(notebook.read_text())["cells"]
    md = ["".join(c["source"]) if isinstance(c["source"], list) else c["source"] for c in cells if c["cell_type"] == "markdown"]
    tag = f"{notebook.parent.name}::{notebook.stem}"
    title = re.sub(r"^#\s*", "", md[0].splitlines()[0]) if md else notebook.stem
    kind = notebook.parent.name
    # Interview-prep notebooks keep questions throughout; topic/capstone notebooks keep them in "🎤 Interview Q&A"
    qa_text = "\n".join(md) if kind.startswith("15_") else section(md, "🎤 Interview Q&A")
    for m in QA_RE.finditer(qa_text):
        front = f"<small>{html.escape(title)}</small><br><br>" + md_to_html(m.group("q"))
        yield front, md_to_html(m.group("a")), f"{tag} interview"
    if include_quiz:
        for m in QUIZ_RE.finditer(section(md, "🧪 Quick Quiz")):
            front = f"<small>{html.escape(title)} · quiz</small><br><br>" + md_to_html(m.group("q"))
            yield front, md_to_html(m.group("a")), f"{tag} quiz"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(REPO / "flashcards" / "ai_course_interview_flashcards.csv"))
    ap.add_argument("--include-quiz", action="store_true")
    args = ap.parse_args()

    notebooks = sorted(COURSE.glob("*/*.ipynb"))
    rows, per_module = [], {}
    for nb in notebooks:
        cards = list(cards_for(nb, args.include_quiz))
        rows.extend(cards)
        per_module[nb.parent.name] = per_module.get(nb.parent.name, 0) + len(cards)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        f.write("#separator:Comma\n#html:true\n#tags column:3\n")
        csv.writer(f).writerows(rows)

    for module, n in per_module.items():
        print(f"{module:34s} {n:4d} cards")
    print(f"{len(rows)} cards from {len(notebooks)} notebooks → {out.relative_to(REPO) if out.is_relative_to(REPO) else out}")


if __name__ == "__main__":
    main()
