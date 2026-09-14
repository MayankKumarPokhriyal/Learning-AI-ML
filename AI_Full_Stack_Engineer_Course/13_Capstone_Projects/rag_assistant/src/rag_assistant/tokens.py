"""Token counting with the LLM's own tokenizer (gpt-oss uses o200k_harmony), so budgets match what the server sees."""

from __future__ import annotations

from functools import lru_cache

import tiktoken

ENCODING_NAME = "o200k_harmony"


@lru_cache(maxsize=1)
def encoding() -> tiktoken.Encoding:
    return tiktoken.get_encoding(ENCODING_NAME)


def count_tokens(text: str) -> int:
    return len(encoding().encode(text, disallowed_special=()))
