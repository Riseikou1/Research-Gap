"""Structured, redacted application logging and correlation context."""

from __future__ import annotations

import contextvars
import hashlib
import json
import logging
import re
from datetime import datetime, timezone

request_id_context: contextvars.ContextVar[str | None] = contextvars.ContextVar("request_id", default=None)
analysis_id_context: contextvars.ContextVar[str | None] = contextvars.ContextVar("analysis_id", default=None)

_SECRET = re.compile(
    r"(?i)(sk-[a-z0-9_-]+|bearer\s+\S+|"
    r"(?:api[-_ ]?key|authorization|password|access[-_ ]?token|refresh[-_ ]?token|"
    r"service[-_ ]?role[-_ ]?key|webhook[-_ ]?secret)\s*[:=]\s*\S+)"
)
_URL = re.compile(r"https?://\S+")
_EMAIL = re.compile(r"\b[^\s@]+@[^\s@]+\b")


def safe_category(error: BaseException) -> str:
    return type(error).__name__[:80]


def safe_user_id(value: str | None) -> str | None:
    if not value:
        return None
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        message = record.getMessage()
        message = _EMAIL.sub("[email]", _URL.sub("[url]", _SECRET.sub("[redacted]", message)))[:1000]
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname.lower(),
            "logger": record.name,
            "message": message,
            "request_id": request_id_context.get(),
            "analysis_id": analysis_id_context.get(),
        }
        for name in ("safe_user_id", "stage", "duration_ms", "outcome", "retry_count", "cache_status", "provider", "usage"):
            value = getattr(record, name, None)
            if value is not None:
                payload[name] = value
        return json.dumps(payload, ensure_ascii=True, separators=(",", ":"))


def configure_structured_logging() -> None:
    root = logging.getLogger()
    if getattr(root, "_research_gap_structured", False):
        return
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root.handlers[:] = [handler]
    root.setLevel(logging.INFO)
    root._research_gap_structured = True  # type: ignore[attr-defined]
