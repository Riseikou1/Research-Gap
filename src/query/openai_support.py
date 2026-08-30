"""Shared defensive helpers for optional OpenAI query components."""

from __future__ import annotations

from typing import Any


def find_refusal(response: Any) -> str | None:
    for output in getattr(response, "output", []) or []:
        if getattr(output, "type", None) != "message":
            continue
        for content in getattr(output, "content", []) or []:
            if getattr(content, "type", None) != "refusal":
                continue
            refusal = getattr(content, "refusal", None)
            if isinstance(refusal, str) and refusal.strip():
                return refusal.strip()
    return None


def incomplete_reason(response: Any) -> str | None:
    details = getattr(response, "incomplete_details", None)
    reason = getattr(details, "reason", None) if details is not None else None
    return str(reason) if reason else None


def format_provider_error(operation: str, exc: Exception) -> str:
    """Format a provider error without logging request data or credentials."""

    details = [f"OpenAI {operation} request failed: {exc}"]
    status_code = getattr(exc, "status_code", None)
    request_id = getattr(exc, "request_id", None)
    if status_code is not None:
        details.append(f"status={status_code}")
    if request_id:
        details.append(f"request_id={request_id}")
    return " | ".join(details)
