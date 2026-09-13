from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from email.message import Message
from pathlib import Path
from urllib.error import HTTPError

from src.operations.logging import JsonFormatter
from src.retrieval.openalex import _retry_delay

ROOT = Path(__file__).parents[2]


def test_retry_after_supports_seconds_and_http_dates() -> None:
    numeric = Message()
    numeric["Retry-After"] = "7"
    error = HTTPError("https://provider.invalid", 429, "rate", numeric, None)
    assert _retry_delay(error, 0.5) == 7

    dated = Message()
    dated["Retry-After"] = "Sun, 13 Sep 2026 12:00:10 GMT"
    error = HTTPError("https://provider.invalid", 503, "busy", dated, None)
    assert _retry_delay(
        error, 0.5, now=datetime(2026, 9, 13, 12, 0, tzinfo=timezone.utc),
    ) == 10


def test_structured_logging_redacts_secrets_urls_and_emails() -> None:
    record = logging.LogRecord(
        "test", logging.ERROR, __file__, 1,
        "failed sk-secretvalue https://private.invalid/path owner@example.test", (), None,
    )
    payload = json.loads(JsonFormatter().format(record))
    assert "secretvalue" not in payload["message"]
    assert "private.invalid" not in payload["message"]
    assert "owner@example.test" not in payload["message"]


def test_container_is_non_root_and_excludes_local_state_and_secrets() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text()
    dockerignore = (ROOT / ".dockerignore").read_text().splitlines()
    assert "USER researchgap" in dockerfile
    assert "src.persistence.migrate" not in dockerfile
    assert "HEALTHCHECK" in dockerfile and "/health/live" in dockerfile
    assert ".env" in dockerignore and "data" in dockerignore and "tests" in dockerignore


def test_runtime_lock_is_exactly_pinned() -> None:
    lines = [line for line in (ROOT / "requirements.lock").read_text().splitlines() if line]
    assert lines
    assert all("==" in line for line in lines)
