"""Response cache: exact (normalized question) or semantic (query-embedding similarity), scoped to one index version and
retrieval configuration so a re-index or configuration change can never serve a stale answer."""

from __future__ import annotations

import re
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


def normalize_question(question: str) -> str:
    return re.sub(r"\s+", " ", question.casefold()).strip().rstrip("?!. ")


@dataclass
class CacheHit:
    value: object
    kind: str  # exact | semantic
    similarity: float
    matched_question: str


class ResponseCache:
    def __init__(self, mode: str = "semantic", threshold: float = 0.95, max_entries: int = 1000, ttl_seconds: float = 3600.0,
                 clock: Callable[[], float] = time.monotonic):
        if mode not in {"off", "exact", "semantic"}:
            raise ValueError(f"unknown cache mode {mode!r}")
        self.mode, self.threshold, self.max_entries, self.ttl, self.clock = mode, threshold, max_entries, ttl_seconds, clock
        self._entries: OrderedDict[tuple[str, str], tuple[float, str, np.ndarray | None, object]] = OrderedDict()
        self._lock = threading.Lock()
        self.stats = {"lookups": 0, "exact_hits": 0, "semantic_hits": 0, "misses": 0, "stores": 0}

    def _alive(self, created: float) -> bool:
        return self.clock() - created <= self.ttl

    def lookup(self, question: str, scope: str, embed: Callable[[str], np.ndarray] | None = None) -> tuple[CacheHit | None, np.ndarray | None]:
        """Exact match first. On a miss in semantic mode, embed the question (only then) and look for the most similar cached
        question in the same scope above the threshold. Returns (hit or None, the query vector if one was computed)."""
        if self.mode == "off":
            return None, None
        key = (scope, normalize_question(question))
        with self._lock:
            self.stats["lookups"] += 1
            entry = self._entries.get(key)
            if entry is not None and self._alive(entry[0]):
                self._entries.move_to_end(key)
                self.stats["exact_hits"] += 1
                return CacheHit(entry[3], "exact", 1.0, entry[1]), entry[2]
        vector = embed(question) if (self.mode == "semantic" and embed is not None) else None
        with self._lock:
            if vector is not None:
                best_key, best_similarity = None, -1.0
                for other_key, (created, _, other_vector, _) in self._entries.items():
                    if other_key[0] == scope and other_vector is not None and self._alive(created):
                        similarity = float(np.dot(vector, other_vector))
                        if similarity > best_similarity:
                            best_key, best_similarity = other_key, similarity
                if best_key is not None and best_similarity >= self.threshold:
                    self._entries.move_to_end(best_key)
                    _, matched, _, value = self._entries[best_key]
                    self.stats["semantic_hits"] += 1
                    return CacheHit(value, "semantic", best_similarity, matched), vector
            self.stats["misses"] += 1
            return None, vector

    def store(self, question: str, scope: str, value: object, vector: np.ndarray | None = None) -> None:
        if self.mode == "off":
            return
        key = (scope, normalize_question(question))
        with self._lock:
            self._entries[key] = (self.clock(), question, vector, value)
            self._entries.move_to_end(key)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)  # evict the least recently used entry
            self.stats["stores"] += 1

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    @property
    def hit_rate(self) -> float:
        lookups = self.stats["lookups"]
        return (self.stats["exact_hits"] + self.stats["semantic_hits"]) / lookups if lookups else 0.0

    def __len__(self) -> int:
        return len(self._entries)
