"""A polite arXiv API client: rate limiting, retries with backoff, snapshot record/replay, and safe Atom parsing.

arXiv asks API users for at most one request every three seconds. Every response is saved under
`data/arxiv_snapshots/`, keyed by its exact URL, so the same requests can be replayed offline and reproducibly.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from defusedxml import ElementTree  # stdlib XML parsers are vulnerable to entity-expansion attacks on untrusted input

from research_agent import config

ATOM = "{http://www.w3.org/2005/Atom}"
ARXIV = "{http://arxiv.org/schemas/atom}"
OPENSEARCH = "{http://a9.com/-/spec/opensearch/1.1/}"
ARXIV_ID_RE = re.compile(r"(?:\d{4}\.\d{4,5}|[a-z][a-z\-]*(?:\.[A-Z]{2})?/\d{7})")
MODES = ("replay", "record", "live")
RETRYABLE_HTTP = {429, 500, 502, 503, 504}


class ArxivError(RuntimeError):
    """The arXiv API could not answer (network failure, HTTP error, or an error feed)."""


class RateLimitedError(ArxivError):
    """arXiv answered 'Rate exceeded.' (HTTP 429) even after retries."""


class SnapshotMissingError(ArxivError):
    """Replay mode was asked for a request that was never recorded."""


def normalize_arxiv_id(raw: str) -> str:
    """'arXiv:2210.03629v3' -> '2210.03629'. Raises ValueError for anything that is not an arXiv identifier."""
    text = str(raw).strip()
    text = re.sub(r"^(?:arxiv:|https?://(?:export\.)?arxiv\.org/(?:abs|pdf)/)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"v\d+$", "", text.removesuffix(".pdf"))
    if not ARXIV_ID_RE.fullmatch(text):
        raise ValueError(f"not a valid arXiv identifier: {str(raw)[:40]!r} (expected something like 2210.03629)")
    return text


@dataclass(frozen=True)
class Paper:
    arxiv_id: str  # without version suffix
    version: int
    title: str
    authors: tuple[str, ...]
    abstract: str
    published: str  # YYYY-MM-DD of version 1
    updated: str  # YYYY-MM-DD of the latest version
    primary_category: str
    categories: tuple[str, ...]
    doi: str | None = None
    journal_ref: str | None = None

    @property
    def year(self) -> int:
        return int(self.published[:4])

    def to_dict(self) -> dict:
        record = asdict(self)
        record["authors"], record["categories"] = list(self.authors), list(self.categories)
        return record


def _clean(element, tag: str) -> str:
    child = element.find(tag)
    return " ".join(child.text.split()) if child is not None and child.text else ""


def parse_feed(body: bytes) -> tuple[list[Paper], int]:
    """Atom feed -> (papers, total results). Raises RateLimitedError / ArxivError for non-feed or error responses."""
    stripped = body.lstrip()
    if not stripped.startswith(b"<"):
        message = stripped[:80].decode("utf-8", "replace").strip()
        if "rate exceeded" in message.lower():
            raise RateLimitedError(f"arXiv rate limit: {message!r}")
        raise ArxivError(f"unexpected non-XML response from arXiv: {message!r}")
    root = ElementTree.fromstring(body)
    total = int(root.findtext(f"{OPENSEARCH}totalResults", default="0") or 0)
    papers = []
    for entry in root.findall(f"{ATOM}entry"):
        entry_id = entry.findtext(f"{ATOM}id", default="")
        if "/api/errors" in entry_id:
            raise ArxivError(f"arXiv API error: {_clean(entry, f'{ATOM}summary')}")
        match = re.search(r"arxiv\.org/abs/(.+?)(?:v(\d+))?$", entry_id)
        title = _clean(entry, f"{ATOM}title")
        if not match or not title:
            continue
        primary = entry.find(f"{ARXIV}primary_category")
        papers.append(Paper(
            arxiv_id=match.group(1),
            version=int(match.group(2) or 1),
            title=title,
            authors=tuple(_clean(author, f"{ATOM}name") for author in entry.findall(f"{ATOM}author")),
            abstract=_clean(entry, f"{ATOM}summary"),
            published=(entry.findtext(f"{ATOM}published") or "")[:10],
            updated=(entry.findtext(f"{ATOM}updated") or "")[:10],
            primary_category=primary.get("term", "") if primary is not None else "",
            categories=tuple(c.get("term", "") for c in entry.findall(f"{ATOM}category")),
            doi=_clean(entry, f"{ARXIV}doi") or None,
            journal_ref=_clean(entry, f"{ARXIV}journal_ref") or None,
        ))
    return papers, total


class RateLimiter:
    """Thread-safe minimum spacing between requests (the clock and sleep are injectable for tests)."""

    def __init__(self, min_interval_s: float, clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep):
        self.min_interval_s, self._clock, self._sleep = min_interval_s, clock, sleep
        self._next_allowed = float("-inf")
        self._lock = threading.Lock()

    def wait(self) -> float:
        """Block until a request may be sent; returns the seconds waited."""
        with self._lock:
            now = self._clock()
            delay = max(0.0, self._next_allowed - now)
            if delay:
                self._sleep(delay)
            self._next_allowed = max(now, self._next_allowed) + self.min_interval_s
            return delay


class SnapshotStore:
    """Raw API responses on disk keyed by the exact request URL, with a JSON sidecar (url, time, size, sha256)."""

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)

    @staticmethod
    def key(url: str) -> str:
        return hashlib.sha256(url.encode()).hexdigest()[:32]

    def get(self, url: str) -> bytes | None:
        path = self.directory / f"{self.key(url)}.xml"
        return path.read_bytes() if path.is_file() else None

    def put(self, url: str, body: bytes, fetched_at: str | None = None) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        key = self.key(url)
        (self.directory / f"{key}.xml").write_bytes(body)
        meta = {"url": url, "fetched_at": fetched_at or datetime.now(UTC).isoformat(timespec="seconds"),
                "bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()}
        (self.directory / f"{key}.json").write_text(json.dumps(meta, indent=1))

    def manifest(self) -> list[dict]:
        return [json.loads(p.read_text()) for p in sorted(self.directory.glob("*.json"))]


class ArxivClient:
    """arXiv API access with a shared rate limiter, retries with exponential backoff, and snapshots."""

    def __init__(self, snapshot_dir: str | Path, *, mode: str = "record", base_url: str = config.ARXIV_API_URL,
                 min_interval_s: float = config.ARXIV_MIN_INTERVAL_S, timeout_s: float = 120.0, max_retries: int = 4,
                 backoff_s: float = 15.0, user_agent: str = config.USER_AGENT, opener: Callable[[str], bytes] | None = None,
                 limiter: RateLimiter | None = None, sleep: Callable[[float], None] = time.sleep):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        self.mode, self.base_url, self.timeout_s = mode, base_url, timeout_s
        self.max_retries, self.backoff_s, self.user_agent = max_retries, backoff_s, user_agent
        self.store = SnapshotStore(snapshot_dir)
        self.limiter = limiter or RateLimiter(min_interval_s)
        self._open = opener or self._urlopen
        self._sleep = sleep
        self.stats = {"live_requests": 0, "snapshot_hits": 0, "retries": 0, "rate_limit_wait_s": 0.0}

    def build_url(self, *, search_query: str | None = None, id_list: Sequence[str] | None = None, start: int = 0,
                  max_results: int = 10, sort_by: str | None = None, sort_order: str | None = None) -> str:
        params: dict[str, str | int] = {}
        if search_query:
            params["search_query"] = search_query
        if id_list:
            params["id_list"] = ",".join(normalize_arxiv_id(i) for i in id_list)
        if start:
            params["start"] = int(start)
        params["max_results"] = int(max_results)
        if sort_by:
            params["sortBy"] = sort_by
        if sort_order:
            params["sortOrder"] = sort_order
        return f"{self.base_url}?{urllib.parse.urlencode(params, safe=',:', quote_via=urllib.parse.quote)}"

    def _urlopen(self, url: str) -> bytes:
        request = urllib.request.Request(url, headers={"User-Agent": self.user_agent})
        with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
            return response.read()

    def fetch(self, url: str) -> tuple[bytes, str]:
        """Response body and where it came from: 'snapshot' or 'live'."""
        if self.mode != "live":
            cached = self.store.get(url)
            if cached is not None:
                self.stats["snapshot_hits"] += 1
                return cached, "snapshot"
            if self.mode == "replay":
                raise SnapshotMissingError(f"no snapshot for {url} (replay mode) — run once with ARXIV_MODE=record")
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            self.stats["rate_limit_wait_s"] += self.limiter.wait()
            self.stats["live_requests"] += 1
            try:
                body = self._open(url)
                parse_feed(body)  # a 200 response can still be 'Rate exceeded.' or an error feed
            except urllib.error.HTTPError as exc:
                if exc.code not in RETRYABLE_HTTP:
                    raise ArxivError(f"arXiv HTTP {exc.code} for {url}") from exc
                last_error = exc
            except (RateLimitedError, OSError) as exc:  # OSError covers URLError, timeouts, connection resets
                last_error = exc
            else:
                self.store.put(url, body)
                return body, "live"
            if attempt < self.max_retries:
                self.stats["retries"] += 1
                self._sleep(min(self.backoff_s * 2**attempt, 300.0))
        if isinstance(last_error, RateLimitedError) or getattr(last_error, "code", None) == 429:
            raise RateLimitedError(f"arXiv kept rate-limiting after {self.max_retries + 1} attempts") from last_error
        raise ArxivError(f"arXiv request failed after {self.max_retries + 1} attempts: {type(last_error).__name__}") from last_error

    def search(self, search_query: str, *, max_results: int = 10, start: int = 0, sort_by: str = "relevance",
               sort_order: str = "descending") -> list[Paper]:
        url = self.build_url(search_query=search_query, start=start, max_results=max_results, sort_by=sort_by, sort_order=sort_order)
        return parse_feed(self.fetch(url)[0])[0]

    def get_by_ids(self, ids: Sequence[str], max_results: int = 20) -> list[Paper]:
        return parse_feed(self.fetch(self.build_url(id_list=ids, max_results=max_results))[0])[0]
