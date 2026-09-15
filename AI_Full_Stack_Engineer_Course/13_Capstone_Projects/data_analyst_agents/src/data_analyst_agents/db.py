"""Read-only database access, by construction rather than by prompt.

Four independent layers: the file is opened with `mode=ro`, `PRAGMA query_only` is on, the SQL must start with
SELECT/WITH, and an SQLite **authorizer** callback approves every table/column read and denies everything else
(writes, ATTACH, PRAGMA, temp objects, dangerous functions, tables outside the allowlist). PII columns are answered
with SQLITE_IGNORE, which makes SQLite read them as NULL. A progress handler stops queries past their time limit.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
import urllib.parse
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from data_analyst_agents import config
from data_analyst_agents.config import QueryLimits

_ACTIONS = ["ALTER_TABLE", "ANALYZE", "ATTACH", "CREATE_INDEX", "CREATE_TABLE", "CREATE_TEMP_INDEX", "CREATE_TEMP_TABLE",
            "CREATE_TEMP_TRIGGER", "CREATE_TEMP_VIEW", "CREATE_TRIGGER", "CREATE_VIEW", "CREATE_VTABLE", "DELETE", "DETACH",
            "DROP_INDEX", "DROP_TABLE", "DROP_TEMP_INDEX", "DROP_TEMP_TABLE", "DROP_TEMP_TRIGGER", "DROP_TEMP_VIEW",
            "DROP_TRIGGER", "DROP_VIEW", "DROP_VTABLE", "FUNCTION", "INSERT", "PRAGMA", "READ", "RECURSIVE", "REINDEX",
            "SAVEPOINT", "SELECT", "TRANSACTION", "UPDATE"]
ACTION_NAMES = {getattr(sqlite3, f"SQLITE_{name}"): name for name in _ACTIONS if hasattr(sqlite3, f"SQLITE_{name}")}
READ_ONLY_START = re.compile(r"^\s*(?:(?:--[^\n]*\n|/\*.*?\*/)\s*)*(select|with)\b", re.I | re.S)
CTE_NAME = re.compile(r"(?:\bwith\s+(?:recursive\s+)?|,\s*)[`\"\[]?([A-Za-z_]\w*)[`\"\]]?\s*(?:\([^)]*\))?\s+as\s*(?:not\s+)?(?:materialized\s*)?\(", re.I)
SQL_KEYWORDS = frozenset({"order", "group", "select", "from", "where", "table", "index", "join", "limit", "by", "values", "transaction"})
_FROM_JOIN = re.compile(r"\b(?:from|join)\s+[`\"\[]?(\w+)[`\"\]]?(?:\s+(?:as\s+)?(?!(?:on|where|inner|left|right|full|cross|join|group|order|"
                        r"limit|natural|using|union|except|intersect|having|window)\b)(\w+))?", re.I)


def quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def clean_sql(sql: str | None) -> str:
    text = (sql or "").strip()
    fenced = re.search(r"```(?:sql|sqlite)?\s*(.*?)```", text, re.S | re.I)
    if fenced:
        text = fenced.group(1)
    return text.strip().rstrip(";").strip()


@dataclass
class QueryResult:
    sql: str
    columns: list[str] = field(default_factory=list)
    rows: list[tuple] = field(default_factory=list)
    truncated: bool = False
    elapsed_s: float = 0.0
    error: str | None = None
    error_kind: str | None = None  # rejected | not_authorized | timeout | sqlite_error
    masked_columns: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.error is None

    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps([self.columns, self.rows], default=str).encode()).hexdigest()

    def to_dict(self, preview_rows: int = 20) -> dict:
        return {"sql": self.sql, "columns": self.columns, "n_rows": len(self.rows), "truncated": self.truncated, "elapsed_s": self.elapsed_s,
                "error": self.error, "error_kind": self.error_kind, "masked_columns": self.masked_columns,
                "preview": [list(r) for r in self.rows[:preview_rows]]}


class ReadOnlyDatabase:
    """One SQLite database behind the four read-only layers. Thread-safe: every query uses its own connection."""

    def __init__(self, db_id: str, path: str | Path | None = None, *, pii_columns: dict[str, Iterable[str]] | None = None,
                 allowed_tables: Iterable[str] | None = None, limits: QueryLimits | None = None):
        self.db_id = db_id
        self.path = Path(path) if path is not None else config.db_path(db_id)
        if not self.path.is_file():
            raise FileNotFoundError(f"database {db_id!r} not found ({self.path.name}); run `make data` first")
        self.limits = limits or QueryLimits()
        pii = config.PII_COLUMNS.get(db_id, {}) if pii_columns is None else pii_columns
        self.pii = {table.lower(): {c.lower() for c in cols} for table, cols in pii.items()}
        con = self._connect()
        try:  # catalog reads happen on a connection without the authorizer
            names = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type IN ('table', 'view') ORDER BY name")]
        finally:
            con.close()
        self._real_names = {n.lower() for n in names} | {"sqlite_master", "sqlite_schema", "sqlite_temp_master", "sqlite_sequence"}
        visible = [n for n in names if n.lower() not in config.HIDDEN_TABLES and not n.lower().startswith("sqlite_")]
        if allowed_tables is not None:
            wanted = {t.lower() for t in allowed_tables}
            visible = [n for n in visible if n.lower() in wanted]
        self.tables = visible
        self._allowed = {t.lower() for t in visible}
        self._row_counts: dict[str, int] = {}

    # ---------- connections and catalog ----------
    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(f"file:{urllib.parse.quote(str(self.path))}?mode=ro", uri=True, check_same_thread=False)
        con.execute("PRAGMA query_only = ON")
        return con

    def _catalog(self, sql: str) -> list[tuple]:
        con = self._connect()
        try:
            return con.execute(sql).fetchall()
        finally:
            con.close()

    def _check_table(self, table: str) -> str:
        if table.lower() not in self._allowed:
            raise KeyError(f"table {table!r} is not in the allowlist of {self.db_id}")
        return next(t for t in self.tables if t.lower() == table.lower())

    def columns(self, table: str) -> list[dict]:
        table = self._check_table(table)
        return [{"name": r[1], "type": r[2] or "", "pk": bool(r[5]), "pii": self.is_pii(table, r[1])}
                for r in self._catalog(f"PRAGMA table_info({quote_ident(table)})")]

    def foreign_keys(self, table: str) -> list[tuple[str, str, str | None]]:
        """(column, referenced table, referenced column or None = its primary key), only towards allowlisted tables."""
        table = self._check_table(table)
        rows = self._catalog(f"PRAGMA foreign_key_list({quote_ident(table)})")
        return sorted({(r[3], r[2], r[4]) for r in rows if r[2].lower() in self._allowed})

    def row_count(self, table: str) -> int:
        table = self._check_table(table)
        if table not in self._row_counts:
            self._row_counts[table] = self._catalog(f"SELECT COUNT(*) FROM {quote_ident(table)}")[0][0]
        return self._row_counts[table]

    def is_pii(self, table: str, column: str) -> bool:
        return column.lower() in self.pii.get(table.lower(), set())

    # ---------- the authorizer: the real read-only / allowlist / masking control ----------
    def _authorizer(self, audit: dict, sql: str = ""):
        # SQLite reports reads of a (recursive) CTE under the CTE's name. Such names are readable, unless they shadow a
        # real table or view of the file: then the read is treated as a read of that object and must pass the allowlist.
        cte_names = {name.lower() for name in CTE_NAME.findall(sql)} - self._real_names

        def check(action: int, arg1: str | None, arg2: str | None, _db_name: str | None, _source: str | None) -> int:
            if action == sqlite3.SQLITE_READ:
                table = (arg1 or "").lower()
                if table in cte_names:
                    return sqlite3.SQLITE_OK
                if table not in self._allowed:
                    audit["denied"].append(f"read of table {arg1}")
                    return sqlite3.SQLITE_DENY
                if (arg2 or "").lower() in self.pii.get(table, set()):
                    audit["masked"].add(f"{arg1}.{arg2}")
                    return sqlite3.SQLITE_IGNORE  # the column reads as NULL
                return sqlite3.SQLITE_OK
            if action in (sqlite3.SQLITE_SELECT, sqlite3.SQLITE_RECURSIVE):
                return sqlite3.SQLITE_OK
            if action == sqlite3.SQLITE_FUNCTION:
                if (arg2 or "").lower() in config.DENIED_SQL_FUNCTIONS:
                    audit["denied"].append(f"function {arg2}()")
                    return sqlite3.SQLITE_DENY
                return sqlite3.SQLITE_OK
            audit["denied"].append(ACTION_NAMES.get(action, f"action {action}") + (f" {arg1}" if arg1 else ""))
            return sqlite3.SQLITE_DENY

        return check

    def execute(self, sql: str, *, max_rows: int | None = None, timeout_s: float | None = None) -> QueryResult:
        max_rows = max_rows or self.limits.max_rows
        timeout_s = timeout_s or self.limits.timeout_s
        result = QueryResult(sql=clean_sql(sql))
        if not READ_ONLY_START.match(result.sql):
            result.error, result.error_kind = "only one read-only SELECT (or WITH … SELECT) statement is allowed", "rejected"
            return result
        audit: dict = {"denied": [], "masked": set()}
        con = self._connect()
        start = time.perf_counter()
        deadline = start + timeout_s
        con.set_authorizer(self._authorizer(audit, result.sql))
        con.set_progress_handler(lambda: int(time.perf_counter() > deadline), 1000)  # non-zero return aborts the query
        try:
            cursor = con.execute(result.sql)
            result.columns = [d[0] for d in cursor.description or []]
            rows = cursor.fetchmany(max_rows + 1)
            result.truncated = len(rows) > max_rows
            result.rows = [tuple(r) for r in rows[:max_rows]]
        except sqlite3.Error as err:
            message = str(err)
            if "interrupted" in message:
                result.error_kind, result.error = "timeout", f"the query ran longer than the {timeout_s:g} s limit and was stopped"
            elif "not authorized" in message or "prohibited" in message:  # SQLite words denied column reads as "access to t.c is prohibited"
                denied = "; ".join(dict.fromkeys(audit["denied"])) or message
                result.error_kind, result.error = "not_authorized", f"not authorized ({denied}). Allowed: read-only SELECT on tables {self.tables}"
            elif "one statement at a time" in message:
                result.error_kind, result.error = "rejected", "only one statement is allowed"
            else:
                result.error_kind, result.error = "sqlite_error", message
        finally:
            result.elapsed_s = round(time.perf_counter() - start, 4)
            result.masked_columns = sorted(audit["masked"])
            con.close()
        return result

    # ---------- helpers for agents and the verifier ----------
    def explain(self, sql: str) -> tuple[list[str], str | None]:
        text = clean_sql(sql)
        if not READ_ONLY_START.match(text):
            return [], "not a SELECT"
        con = self._connect()
        con.set_authorizer(self._authorizer({"denied": [], "masked": set()}, text))
        try:
            return [row[3] for row in con.execute("EXPLAIN QUERY PLAN " + text).fetchall()], None
        except sqlite3.Error as err:
            return [], str(err)
        finally:
            con.close()

    def estimate_cost(self, sql: str) -> dict:
        """Heuristic cost from EXPLAIN QUERY PLAN: rows of every table the plan reads with a full SCAN (no index)."""
        plan, error = self.explain(sql)
        names: dict[str, str] = {}
        for table, alias in _FROM_JOIN.findall(clean_sql(sql)):
            if table.lower() in self._allowed:
                real = self._check_table(table)
                names[table.lower()] = real
                if alias:
                    names[alias.lower()] = real
        scans = []
        for line in plan:
            match = re.match(r"SCAN (\w+)", line)
            if match and match.group(1).lower() in names:
                table = names[match.group(1).lower()]
                scans.append((table, self.row_count(table)))
        scanned = sum(n for _, n in scans)
        return {"plan": plan, "error": error, "full_scans": scans, "scanned_rows": scanned, "expensive": scanned > self.limits.expensive_scan_rows}

    def count_rows(self, sql: str) -> int | None:
        result = self.execute(f"SELECT COUNT(*) FROM ({clean_sql(sql)})", max_rows=1)
        return result.rows[0][0] if result.ok and result.rows else None

    def sample_values(self, table: str, column: str, k: int = 3, max_chars: int = 40) -> list[str]:
        if self.is_pii(table, column):
            return []
        col, tab = quote_ident(column), quote_ident(self._check_table(table))
        result = self.execute(f"SELECT DISTINCT {col} FROM {tab} WHERE {col} IS NOT NULL LIMIT {int(k)}", max_rows=k, timeout_s=2)
        return [str(value)[:max_chars] for (value,) in result.rows] if result.ok else []


def open_databases(db_ids: Iterable[str] = config.DATABASES, limits: QueryLimits | None = None) -> dict[str, ReadOnlyDatabase]:
    return {db_id: ReadOnlyDatabase(db_id, limits=limits) for db_id in db_ids}
