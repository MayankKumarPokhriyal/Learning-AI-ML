"""Guardrails: input limits, prompt-injection detection for ingested text, URL checks on answers, and redaction for logs."""

from __future__ import annotations

import re
from urllib.parse import urlsplit

from rag_assistant.tokens import count_tokens

# ── 1. Input limits ───────────────────────────────────────────────────────────────────────────────────────────
CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


class InputRejected(ValueError):
    def __init__(self, reason: str, status_code: int):
        super().__init__(reason)
        self.reason, self.status_code = reason, status_code


def check_question(question: str, max_chars: int, max_tokens: int) -> str:
    """Strip control characters and enforce size limits before any model sees the text (cost and abuse control)."""
    cleaned = CONTROL_CHARS.sub("", question).strip()
    if not cleaned:
        raise InputRejected("the question is empty", 422)
    if len(cleaned) > max_chars:
        raise InputRejected(f"the question has {len(cleaned)} characters; the limit is {max_chars}", 413)
    n_tokens = count_tokens(cleaned)
    if n_tokens > max_tokens:
        raise InputRejected(f"the question has {n_tokens} tokens; the limit is {max_tokens}", 413)
    return cleaned


# ── 2. Prompt-injection detection (run on every chunk at ingestion) ─────────────────────────────────────────────
INJECTION_RULES: dict[str, re.Pattern] = {
    "override_instructions": re.compile(
        r"\b(?:ignore|disregard|forget|override)\b[^.\n]{0,40}\b(?:previous|prior|above|earlier|preceding|all|any|your|system)\b"
        r"[^.\n]{0,30}\b(?:instructions?|prompts?|rules|directions|guidelines)\b", re.I),
    "addressed_to_ai": re.compile(
        r"\b(?:note|message|instructions?|attention)\s+(?:for|to)\s+(?:the\s+)?(?:ai|llms?|language models?|(?:ai\s+)?assistants?|chat ?bots?)\b"
        r"|\b(?:ai|llm)\s+(?:assistants?|agents?|models?)\s*(?:must|should|:)", re.I),
    "role_manipulation": re.compile(r"\byou are now\b|\bnew system prompt\b|\bdeveloper mode\b|\bjailbreak\b", re.I),
    "prompt_exfiltration": re.compile(r"\b(?:reveal|print|repeat|leak)\b[^.\n]{0,30}\b(?:system prompt|hidden instructions)\b", re.I),
    "chat_markup": re.compile(r"<\|(?:im_start|im_end|start|end|system|channel|message)\|>|<\s*/?\s*(?:system|assistant)\s*>|\[/?INST\]", re.I),
}


def detect_injection(text: str) -> list[str]:
    """Names of the rules that match. A tripwire, not a wall: paraphrased attacks can evade patterns."""
    return [name for name, pattern in INJECTION_RULES.items() if pattern.search(text)]


# ── 3. Output check: links in answers ──────────────────────────────────────────────────────────────────────────
URL_RE = re.compile(r"https?://[^\s<>\"'`)\]}]+", re.I)


def host_allowed(url: str, allowed_domains: tuple[str, ...]) -> bool:
    host = (urlsplit(url).hostname or "").lower()
    return any(host == d or host.endswith("." + d) for d in allowed_domains)


def disallowed_urls(text: str, allowed_domains: tuple[str, ...]) -> list[str]:
    return [m.group().rstrip(".,;:") for m in URL_RE.finditer(text) if not host_allowed(m.group(), allowed_domains)]


class UrlHoldback:
    """Streaming version of the URL check: hold back text that may be the start of a URL until the URL is complete,
    so a disallowed link is never sent to the client, even for a moment."""

    PREFIXES = ("https://", "http://")

    def __init__(self, allowed_domains: tuple[str, ...]):
        self.allowed, self.pending, self.blocked = allowed_domains, "", False

    def feed(self, text: str) -> str:
        if self.blocked:
            return ""
        self.pending += text
        cut = len(self.pending)
        for m in URL_RE.finditer(self.pending):
            if m.end() == len(self.pending):  # possibly unfinished: wait for more text
                cut = min(cut, m.start())
            elif not host_allowed(m.group(), self.allowed):
                self.blocked, self.pending = True, ""
                return ""
        for prefix in self.PREFIXES:  # a partial "htt" at the very end could become a URL
            for k in range(len(prefix) - 1, 0, -1):
                if self.pending.lower().endswith(prefix[:k]):
                    cut = min(cut, len(self.pending) - k)
                    break
        out, self.pending = self.pending[:cut], self.pending[cut:]
        return out

    def flush(self) -> str:
        if self.blocked:
            return ""
        out, self.pending = self.pending, ""
        if disallowed_urls(out, self.allowed):
            self.blocked = True
            return ""
        return out


# ── 4. Redaction for logs and traces ───────────────────────────────────────────────────────────────────────────
CARD_CANDIDATE = re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)")
REDACTION_RULES: list[tuple[str, re.Pattern, str]] = [
    ("private_key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.S), "[REDACTED_PRIVATE_KEY]"),
    ("url_credentials", re.compile(r"\b([a-z][a-z0-9+.-]*://)[^/\s:@]+:[^/\s@]+@", re.I), r"\1[REDACTED_CREDENTIALS]@"),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}"), "[REDACTED_JWT]"),
    ("aws_access_key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"), "[REDACTED_AWS_KEY]"),
    ("github_token", re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{40,})\b"), "[REDACTED_GITHUB_TOKEN]"),
    ("api_key", re.compile(r"\b(?:sk|pk|rk)-(?:proj-|live-|test-)?[A-Za-z0-9_-]{20,}\b"), "[REDACTED_API_KEY]"),
    ("pypi_token", re.compile(r"\bpypi-[A-Za-z0-9_-]{50,}\b"), "[REDACTED_PYPI_TOKEN]"),
    ("slack_token", re.compile(r"\bxox[abpors]-[A-Za-z0-9-]{10,}\b"), "[REDACTED_SLACK_TOKEN]"),
    ("bearer_token", re.compile(r"\b(Bearer)\s+[A-Za-z0-9._~+/-]{16,}=*", re.I), r"\1 [REDACTED_TOKEN]"),
    ("secret_assignment", re.compile(r"\b(password|passwd|secret|token|api[_-]?key)(\s*[:=]\s*)(\"[^\"]+\"|'[^']+'|[^\s,;\"']+)", re.I),
     r"\1\2[REDACTED_SECRET]"),
    ("email", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), "[REDACTED_EMAIL]"),
    ("phone", re.compile(r"(?<![\w.])(?:\+\d{1,3}[\s.-]?)?\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}(?!\w|\.\d)"), "[REDACTED_PHONE]"),
    ("ip_address", re.compile(r"(?<![\d.])(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)(?![\d.])"), "[REDACTED_IP]"),
]


def _luhn_ok(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2:
            d = d * 2 - 9 if d * 2 > 9 else d * 2
        total += d
    return total % 10 == 0


def redact(text: str) -> tuple[str, dict[str, int]]:
    """Replace secrets and personal data with placeholders. Returns the redacted text and a count per rule."""
    counts: dict[str, int] = {}

    def card(m: re.Match) -> str:
        digits = re.sub(r"\D", "", m.group())
        if 13 <= len(digits) <= 19 and _luhn_ok(digits):
            counts["credit_card"] = counts.get("credit_card", 0) + 1
            return "[REDACTED_CARD]"
        return m.group()

    for name, pattern, replacement in REDACTION_RULES:
        if name == "email":  # cards before phone/email rules so long digit runs aren't half-redacted
            text = CARD_CANDIDATE.sub(card, text)
        text, n = pattern.subn(replacement, text)
        if n:
            counts[name] = counts.get(name, 0) + n
    return text, counts
