"""The local paper corpus: a SQLite table plus an FTS5 full-text index (BM25 ranking) over harvested arXiv papers."""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Iterable
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

from research_agent.arxiv import Paper

SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
    arxiv_id TEXT PRIMARY KEY,
    version INTEGER NOT NULL,
    title TEXT NOT NULL,
    authors TEXT NOT NULL,          -- '; '-separated
    abstract TEXT NOT NULL,
    published TEXT NOT NULL,
    updated TEXT NOT NULL,
    primary_category TEXT NOT NULL,
    categories TEXT NOT NULL,       -- space-separated
    doi TEXT,
    journal_ref TEXT,
    source TEXT NOT NULL,           -- the request (or controlled test) a record came from
    ingested_at TEXT NOT NULL
);
CREATE VIRTUAL TABLE IF NOT EXISTS papers_fts USING fts5(
    arxiv_id UNINDEXED, title, abstract, authors, tokenize = 'porter unicode61'
);
"""
STOPWORDS = frozenset(
    "a about an and are as at be by did do does find for from how in into is it its of on or paper papers that the "
    "this to was what when which who whose with arxiv".split()
)


def fts_query(text: str, max_terms: int = 12) -> str:
    """Free text -> an FTS5 OR-query of quoted terms. Quoting every term means user text can never inject FTS syntax."""
    terms = [t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 1 and t not in STOPWORDS]
    return " OR ".join(f'"{t}"' for t in list(dict.fromkeys(terms))[:max_terms])


def _row_to_paper(row: sqlite3.Row) -> Paper:
    return Paper(arxiv_id=row["arxiv_id"], version=row["version"], title=row["title"], authors=tuple(row["authors"].split("; ")),
                 abstract=row["abstract"], published=row["published"], updated=row["updated"],
                 primary_category=row["primary_category"], categories=tuple(row["categories"].split()),
                 doi=row["doi"], journal_ref=row["journal_ref"])


class PaperIndex:
    """One SQLite file. A new connection per call keeps it safe to use from several threads and processes."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as con, con:
            con.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.path, timeout=30)
        con.row_factory = sqlite3.Row
        return con

    def upsert(self, papers: Iterable[Paper], source: str) -> int:
        rows = list(papers)
        now = datetime.now(UTC).isoformat(timespec="seconds")
        with closing(self._connect()) as con, con:
            for p in rows:
                con.execute("INSERT OR REPLACE INTO papers VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                            (p.arxiv_id, p.version, p.title, "; ".join(p.authors), p.abstract, p.published, p.updated,
                             p.primary_category, " ".join(p.categories), p.doi, p.journal_ref, source, now))
                con.execute("DELETE FROM papers_fts WHERE arxiv_id = ?", (p.arxiv_id,))
                con.execute("INSERT INTO papers_fts (arxiv_id, title, abstract, authors) VALUES (?, ?, ?, ?)",
                            (p.arxiv_id, p.title, p.abstract, " ".join(p.authors)))
        return len(rows)

    def get(self, arxiv_id: str) -> Paper | None:
        with closing(self._connect()) as con:
            row = con.execute("SELECT * FROM papers WHERE arxiv_id = ?", (arxiv_id,)).fetchone()
        return _row_to_paper(row) if row else None

    def search(self, query: str = "", *, author: str | None = None, category: str | None = None,
               year_from: int | None = None, year_to: int | None = None, limit: int = 5) -> list[Paper]:
        """BM25-ranked full-text search (title weighted 10×, authors 3×, abstract 1×) with optional filters."""
        match = fts_query(query) if query else ""
        if match:
            sql = ("SELECT p.* FROM papers_fts JOIN papers AS p ON p.arxiv_id = papers_fts.arxiv_id "
                   "WHERE papers_fts MATCH ?")
            params: list = [match]
            order = "ORDER BY bm25(papers_fts, 0.0, 10.0, 1.0, 3.0), p.published DESC"
        else:
            sql, params, order = "SELECT p.* FROM papers AS p WHERE 1 = 1", [], "ORDER BY p.published DESC"
        if author:
            sql += " AND p.authors LIKE ?"
            params.append(f"%{author}%")
        if category:
            sql += " AND (' ' || p.categories || ' ') LIKE ?"
            params.append(f"% {category} %")
        if year_from is not None:
            sql += " AND CAST(substr(p.published, 1, 4) AS INTEGER) >= ?"
            params.append(int(year_from))
        if year_to is not None:
            sql += " AND CAST(substr(p.published, 1, 4) AS INTEGER) <= ?"
            params.append(int(year_to))
        with closing(self._connect()) as con:
            rows = con.execute(f"{sql} {order} LIMIT ?", [*params, int(limit)]).fetchall()
        return [_row_to_paper(row) for row in rows]

    def count(self) -> int:
        with closing(self._connect()) as con:
            return con.execute("SELECT count(*) FROM papers").fetchone()[0]

    def stats(self) -> dict:
        with closing(self._connect()) as con:
            years = con.execute("SELECT min(substr(published, 1, 4)), max(substr(published, 1, 4)) FROM papers").fetchone()
            top = con.execute("SELECT primary_category, count(*) AS n FROM papers GROUP BY 1 ORDER BY n DESC LIMIT 5").fetchall()
            sources = con.execute("SELECT source, count(*) AS n FROM papers GROUP BY 1 ORDER BY n DESC").fetchall()
        return {"papers": self.count(), "first_year": years[0], "last_year": years[1],
                "top_primary_categories": {r[0]: r[1] for r in top}, "sources": {r[0]: r[1] for r in sources}}

    def copy_to(self, path: str | Path) -> PaperIndex:
        """A full copy (SQLite backup API) — used to build controlled test variants without touching the real corpus."""
        target = Path(path)
        target.unlink(missing_ok=True)
        with closing(self._connect()) as source, closing(sqlite3.connect(target)) as destination:
            source.backup(destination)
        return PaperIndex(target)
