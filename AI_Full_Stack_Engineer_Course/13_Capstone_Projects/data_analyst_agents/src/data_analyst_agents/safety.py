"""Deterministic safety helpers: export intent, injection heuristics, quarantine of untrusted data cells, output checks.

Database cells are **untrusted text**: anyone who can write a row (a club member typing an expense description, a
customer filling in a form) can hide instructions in it. Heuristics only raise flags; the controls that bound the
damage are structural: the LLMs see text cells as handles (Quarantine), exports need a human, links are stripped, and
every number in a finding must exist in the data.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable

EXPORT_INTENT = re.compile(
    r"\b(export|download|csv|excel|xlsx|spreadsheet|send (?:me )?(?:the )?(?:file|data|results?|table)|save (?:the )?(?:results?|data|table) (?:to|as))\b",
    re.I)

INJECTION_PATTERNS = {
    "override_instructions": re.compile(r"\b(ignore|disregard|forget|override)\b.{0,40}\b(instructions?|rules|prompts?|above|previous|guidelines)\b", re.I | re.S),
    "addressed_to_ai": re.compile(r"\b(note|message|instructions?)\s+(to|for)\s+(the\s+)?(ai|assistant|analyst|model|llm|agent|report writer)\b"
                                  r"|\b(system|assistant|developer)\s*:", re.I),
    "action_request": re.compile(r"\b(export|approve|delete|drop\s+table|grant|transfer|wire)\b.{0,40}\b(data|table|file|members?|now|immediately|funds)\b", re.I | re.S),
    "link": re.compile(r"https?://|www\.", re.I),
    "urgency_or_credentials": re.compile(r"\b(verify|re-?verify|confirm)\b.{0,30}\b(account|password|login|identity)\b|\bcorrupt(ed)?\b", re.I | re.S),
}
URL_RE = re.compile(r"(?:https?://|www\.)[^\s)>\]`'\"]+", re.I)
NUMBER_RE = re.compile(r"(?<![\w.⟦])-?\d[\d,]*(?:\.\d+)?%?")
FAKE_HANDLE = re.compile(r"⟦(?!v\d+⟧)([^⟦⟧]{1,80})⟧")


def export_requested(user_text: str) -> bool:
    """Export intent comes ONLY from the user's own words, never from a model or from data."""
    return bool(EXPORT_INTENT.search(user_text or ""))


def injection_flags(text: str) -> list[str]:
    return [name for name, pattern in INJECTION_PATTERNS.items() if pattern.search(text or "")]


def defang(text: str, max_chars: int = 80) -> str:
    """Render an untrusted value as inert text: one line, no backticks, links made non-clickable, truncated."""
    value = re.sub(r"\s+", " ", str(text)).replace("`", "'").strip()
    value = re.sub(r"https?://", lambda m: m.group(0).replace("tt", "xx").replace("://", "[:]//"), value, flags=re.I)
    value = value.replace("www.", "www[.]")
    return value if len(value) <= max_chars else value[: max_chars - 1] + "…"


def strip_links(text: str) -> tuple[str, list[str]]:
    found = URL_RE.findall(text or "")
    return URL_RE.sub("[link removed]", text or ""), found


class Quarantine:
    """Dual-LLM / CaMeL-style handling of data cells.

    Numbers and short, flag-free labels pass through. Every other text cell is replaced by a handle such as ⟦v3⟧ before
    any LLM sees it; after generation, code substitutes the real values as inert code spans. A model can mention a value
    by handle but can never read (and therefore never obey) the text inside it. With enabled=False, cells pass verbatim.
    """

    SAFE_TEXT = re.compile(r"^[\w .,:;'&()/%+#-]{0,32}$")
    HANDLE = re.compile(r"⟦v\d+⟧")

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._handle_of: dict[str, str] = {}
        self.values: dict[str, str] = {}

    def cell(self, value):
        if value is None or isinstance(value, bool | int):
            return value
        if isinstance(value, float):
            return round(value, 6)
        text = str(value)
        if not self.enabled or (self.SAFE_TEXT.match(text) and not injection_flags(text)):
            return text
        if text not in self._handle_of:
            handle = f"⟦v{len(self._handle_of) + 1}⟧"
            self._handle_of[text], self.values[handle] = handle, text
        return self._handle_of[text]

    def rows(self, columns: list[str], rows: list[tuple], limit: int = 20) -> list[dict]:
        return [{c: self.cell(v) for c, v in zip(columns, row, strict=False)} for row in rows[:limit]]

    def substitute(self, text: str) -> str:
        filled = self.HANDLE.sub(lambda m: f"`{defang(self.values[m.group(0)])}`" if m.group(0) in self.values else m.group(0), text or "")
        return FAKE_HANDLE.sub(r"\1", filled)  # the model sometimes puts handle brackets around plain words: they are not handles


def numbers_in(text: str) -> list[float]:
    out = []
    for token in NUMBER_RE.findall(text or ""):
        try:
            out.append(float(token.rstrip("%").replace(",", "")))
        except ValueError:
            continue
    return out


def unsupported_numbers(text: str, allowed: Iterable[float]) -> list[str]:
    """Numbers in `text` that don't match any allowed number, allowing rounding to the digits written, ratio ↔ percent,
    and — for tokens written with % — a share computed from two allowed numbers (a / b × 100, within one point)."""
    candidates = [float(a) for a in allowed if isinstance(a, int | float) and not isinstance(a, bool) and math.isfinite(float(a))]
    few = candidates[:60]
    shares = [100 * a / b for a in few for b in few if b and a != b and 0 < a / b < 10]
    bad = []
    for token in NUMBER_RE.findall(text or ""):
        raw = token.rstrip("%").replace(",", "")
        try:
            value = float(raw)
        except ValueError:
            continue
        decimals = len(raw.split(".")[1]) if "." in raw else 0
        tolerance = 0.5 * 10 ** (-decimals) + 1e-9
        ok = any(abs(value - a) <= tolerance or abs(value - a * 100) <= tolerance or abs(value - round(a)) <= tolerance for a in candidates)
        if not ok and token.endswith("%"):
            ok = any(abs(value - share) <= 1.0 for share in shares)
        if not ok and len(raw.lstrip("-")) == 4 and 1900 <= value <= 2100:
            ok = True  # a year mentioned in passing
        if not ok:
            bad.append(token)
    return bad
