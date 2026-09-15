"""Schema catalog: tables, columns, BIRD column descriptions, sample values — and BM25 retrieval over columns.

The SQL writer only sees the tables the planner picked (context isolation), rendered with the descriptions that make
cryptic columns usable (financial.district.A11 is "average salary"). Sample values are data, so they are truncated and
dropped when they look like instructions.
"""

from __future__ import annotations

import csv
import io
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

from data_analyst_agents import config
from data_analyst_agents.db import SQL_KEYWORDS, ReadOnlyDatabase
from data_analyst_agents.safety import injection_flags

STOPWORDS = frozenset("""a an the of in on for to and or is are was were be been by with what which who whom whose how many much list
give show me all each per as at from that this those these than there their its it do does did please find tell name names""".split())


def tokenize(text: str) -> list[str]:
    """camelCase and snake_case aware word tokens, lower-cased, stop words removed, a crude plural strip."""
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text or "")
    tokens = []
    for token in re.findall(r"[a-z0-9]+", text.lower()):
        if token in STOPWORDS:
            continue
        if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
            token = token[:-1]
        tokens.append(token)
    return tokens


class BM25:
    """Okapi BM25 with the Lucene idf, log(1 + (N - df + 0.5) / (df + 0.5)), which is never negative."""

    def __init__(self, documents: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.n = len(documents)
        self.lengths = [len(d) for d in documents]
        self.avgdl = (sum(self.lengths) / self.n) if self.n else 1.0
        self.tf = [Counter(d) for d in documents]
        self.df = Counter(term for d in documents for term in set(d))

    def idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        return math.log(1 + (self.n - df + 0.5) / (df + 0.5))

    def scores(self, query: list[str]) -> list[float]:
        terms = list(dict.fromkeys(query))
        out = []
        for tf, length in zip(self.tf, self.lengths, strict=True):
            score = 0.0
            for term in terms:
                f = tf.get(term, 0)
                if f:
                    score += self.idf(term) * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * length / self.avgdl))
            out.append(score)
        return out


def _decode(raw: bytes) -> str:
    for encoding in ("utf-8-sig", "cp1252", "latin-1"):  # BIRD ships mostly UTF-8 (some with a BOM) and a few Windows-1252 files
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise AssertionError("latin-1 decodes any byte sequence")


def _clean(text: str | None) -> str:
    text = (text or "").replace("commonsense evidence:", " ").replace("�", " ")
    return re.sub(r"\s+", " ", text).strip()


def read_descriptions(folder: Path | None) -> dict[str, dict[str, dict[str, str]]]:
    """{table: {column: {label, description, values}}} from BIRD's database_description/*.csv files."""
    out: dict[str, dict[str, dict[str, str]]] = {}
    if folder is None or not folder.is_dir():
        return out
    for path in sorted(folder.glob("*.csv")):
        columns = {}
        for row in csv.DictReader(io.StringIO(_decode(path.read_bytes()))):
            row = {(k or "").strip(): (v if isinstance(v, str) else "") for k, v in row.items()}
            name = (row.get("original_column_name") or "").strip()
            if name:
                columns[name.lower()] = {"label": _clean(row.get("column_name")), "description": _clean(row.get("column_description")),
                                         "values": _clean(row.get("value_description"))}
        out[path.stem.lower()] = columns
    return out


@dataclass
class Column:
    table: str
    name: str
    type: str
    pk: bool
    pii: bool
    label: str = ""
    description: str = ""
    values: str = ""
    samples: list[str] = field(default_factory=list)
    references: str | None = None

    def meaning(self) -> str:
        parts, seen = [], {self.name.lower()}
        for text in (self.label, self.description, self.values):
            if text and text.lower() not in seen:
                parts.append(text)
                seen.add(text.lower())
        return "; ".join(parts)

    def tokens(self) -> list[str]:
        return tokenize(" ".join([self.table, self.name, self.label, self.description, self.values]))


@dataclass
class Table:
    name: str
    rows: int
    columns: list[Column]


class SchemaCatalog:
    def __init__(self, db: ReadOnlyDatabase, description_dir: Path | None = None, *, samples: int = 3):
        self.db, self.db_id = db, db.db_id
        descriptions = read_descriptions(description_dir if description_dir is not None else config.description_dir(db.db_id))
        self.tables: dict[str, Table] = {}
        self.links: dict[str, set[str]] = defaultdict(set)
        self.dropped_samples = 0
        for table in db.tables:
            refs = {}
            for column, ref_table, ref_column in db.foreign_keys(table):
                refs[column.lower()] = f"{ref_table}.{ref_column}" if ref_column else ref_table
                self.links[table].add(ref_table)
                self.links[ref_table].add(table)
            described = descriptions.get(table.lower(), {})
            columns = []
            for info in db.columns(table):
                d = described.get(info["name"].lower(), {})
                values = [] if info["pii"] or samples == 0 else db.sample_values(table, info["name"], samples)
                safe = [v for v in values if not injection_flags(v)]  # sample values are untrusted data too
                self.dropped_samples += len(values) - len(safe)
                columns.append(Column(table, info["name"], info["type"], info["pk"], info["pii"], d.get("label", ""), d.get("description", ""),
                                      d.get("values", ""), safe, refs.get(info["name"].lower())))
            self.tables[table] = Table(table, db.row_count(table), columns)
        self._columns = [c for t in self.tables.values() for c in t.columns]
        self._bm25 = BM25([c.tokens() for c in self._columns])

    def resolve(self, name: str) -> str | None:
        wanted = name.strip().strip('"`[]').lower()
        return next((t for t in self.tables if t.lower() == wanted), None)

    def retrieve(self, question: str, evidence: str = "", k: int = 4) -> list[tuple[str, float]]:
        """Rank tables by the sum of their three best-matching columns' BM25 scores."""
        scores = self._bm25.scores(tokenize(f"{question} {evidence}"))
        per_table: dict[str, list[float]] = defaultdict(list)
        for column, score in zip(self._columns, scores, strict=True):
            per_table[column.table].append(score)
        ranked = sorted(((t, round(sum(sorted(v, reverse=True)[:3]), 3)) for t, v in per_table.items()), key=lambda x: (-x[1], x[0]))
        return ranked[:k]

    def relevance(self, question: str, evidence: str = "") -> float:
        return round(sum(score for _, score in self.retrieve(question, evidence, k=3)), 3)

    def connect_tables(self, tables: list[str]) -> list[str]:
        """Add one bridge table for every chosen pair that has no direct foreign key but shares a neighbour."""
        chosen = [t for t in dict.fromkeys(self.resolve(x) for x in tables) if t]
        result = list(chosen)
        for a, b in combinations(chosen, 2):
            if b not in self.links[a]:
                bridges = sorted(self.links[a] & self.links[b])
                if bridges and bridges[0] not in result:
                    result.append(bridges[0])
        return result

    def render(self, tables: list[str] | None = None, *, max_chars: int = 9000, samples: bool = True) -> str:
        lines = [f"DATABASE {self.db_id}"]
        for name in tables or list(self.tables):
            table = self.tables[self.resolve(name) or name]
            note = f' — a reserved word: write "{table.name}"' if table.name.lower() in SQL_KEYWORDS else ""
            lines.append(f"TABLE {table.name} ({table.rows:,} rows){note}")
            for c in table.columns:
                parts = [f"  - {c.name} {c.type}".rstrip()]
                if c.pk:
                    parts.append("PRIMARY KEY")
                if c.references:
                    parts.append(f"→ {c.references}")
                if c.meaning():
                    parts.append(f"— {c.meaning()[:240]}")
                if c.pii:
                    parts.append("[MASKED personal data: always reads as NULL]")
                elif samples and c.samples:
                    numeric = c.type.upper() in {"INTEGER", "REAL", "NUMERIC", "FLOAT", "DOUBLE", "INT"}
                    parts.append("e.g. " + ", ".join(s if numeric else repr(s) for s in c.samples))
                lines.append(" ".join(parts))
        text = "\n".join(lines)
        return text if len(text) <= max_chars else text[:max_chars] + "\n… (schema truncated)"

    def overview(self, question: str = "", evidence: str = "", k: int = 4) -> str:
        """Compact map for the planner: every table with column names (labels for cryptic ones); best matches first."""
        order = [t for t, _ in self.retrieve(question, evidence, k=len(self.tables))] if question else list(self.tables)
        lines = [f"DATABASE {self.db_id} (keyword relevance {self.relevance(question, evidence) if question else 0})"]
        for name in order:
            table = self.tables[name]
            cols = [c.name + (f' "{c.label}"' if c.label and c.label.lower().replace(" ", "") != c.name.lower().replace("_", "") and len(c.name) <= 4 else "")
                    for c in table.columns]
            lines.append(f"  {name}({', '.join(cols)})")
        return "\n".join(lines)


def load_catalogs(db_ids=config.DATABASES, limits=None) -> dict[str, SchemaCatalog]:
    return {db_id: SchemaCatalog(ReadOnlyDatabase(db_id, limits=limits)) for db_id in db_ids}
