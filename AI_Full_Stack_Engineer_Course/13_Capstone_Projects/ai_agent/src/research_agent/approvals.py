"""Signed approvals: the MCP server executes a side-effect tool only with a token proving a human approved these exact arguments.

The host (API) and the tool server share APPROVAL_SIGNING_KEY. The token is an HMAC over (approval id, tool, arguments),
so a changed argument, a different tool, or a model that invents a token is rejected — defense in depth behind the
host's approval queue (a confused or compromised host still can't write files without a valid approval).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from typing import Any

HIDDEN_ARGUMENTS = ("approval_id", "approval_token")  # added by the host after approval; never shown to the model


def canonical(approval_id: str, tool: str, arguments: dict[str, Any]) -> bytes:
    payload = {"approval_id": approval_id, "tool": tool, "arguments": {k: v for k, v in arguments.items() if k not in HIDDEN_ARGUMENTS}}
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def sign(key: bytes, approval_id: str, tool: str, arguments: dict[str, Any]) -> str:
    return hmac.new(key, canonical(approval_id, tool, arguments), hashlib.sha256).hexdigest()


def verify(key: bytes | None, approval_id: str, tool: str, arguments: dict[str, Any], token: str) -> bool:
    if not key or not approval_id or not token:
        return False
    return hmac.compare_digest(sign(key, approval_id, tool, arguments), token)


def key_from_env() -> bytes | None:
    value = os.environ.get("APPROVAL_SIGNING_KEY")
    return value.encode() if value else None
