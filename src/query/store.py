"""Small persistent cache for provider-backed research planning."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
import unicodedata
from pathlib import Path
from threading import RLock
from typing import Any


# Bump when planning prompts, transport schemas, or their interpretation
# changes. Old rows remain harmless misses.
PLANNING_CACHE_VERSION = 5


def normalize_idea_for_cache(value: str) -> str:
    """Normalize only representation-level whitespace and Unicode form."""

    return " ".join(
        unicodedata.normalize("NFKC", value).split()
    ).strip()


def planning_cache_key(
    *,
    kind: str,
    input_value: Any,
    provider: str,
    model: str | None,
    configuration: dict[str, Any] | None = None,
) -> str:
    """Build a collision-resistant key from all compatibility inputs."""

    payload = {
        "version": PLANNING_CACHE_VERSION,
        "kind": kind,
        "input": input_value,
        "provider": provider,
        "model": model,
        "configuration": configuration or {},
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class PlanningStore:
    """Versioned SQLite storage for decomposition/query-generation payloads."""

    def __init__(self, path: str | Path | None) -> None:
        self.path = Path(path) if path is not None else None
        self._lock = RLock()
        self._connection: sqlite3.Connection | None = None

        if self.path is None:
            return

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(
            self.path,
            timeout=30.0,
            check_same_thread=False,
        )
        self._connection.execute("PRAGMA busy_timeout = 30000")
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS planning_cache (
                cache_kind TEXT NOT NULL,
                cache_key TEXT NOT NULL,
                schema_version INTEGER NOT NULL,
                payload TEXT NOT NULL,
                updated_at REAL NOT NULL,
                PRIMARY KEY (cache_kind, cache_key, schema_version)
            )
            """
        )
        self._connection.commit()

    def get(self, *, kind: str, key: str) -> Any | None:
        if self._connection is None:
            return None

        try:
            with self._lock:
                row = self._connection.execute(
                    """
                    SELECT payload
                    FROM planning_cache
                    WHERE cache_kind = ?
                      AND cache_key = ?
                      AND schema_version = ?
                    """,
                    (kind, key, PLANNING_CACHE_VERSION),
                ).fetchone()
        except sqlite3.DatabaseError:
            return None

        if row is None:
            return None

        try:
            return json.loads(row[0])
        except (TypeError, ValueError, json.JSONDecodeError):
            return None

    def put(self, *, kind: str, key: str, payload: Any) -> None:
        if self._connection is None:
            return

        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        with self._lock:
            self._connection.execute(
                """
                INSERT OR REPLACE INTO planning_cache
                    (cache_kind, cache_key, schema_version, payload, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    kind,
                    key,
                    PLANNING_CACHE_VERSION,
                    encoded,
                    time.time(),
                ),
            )
            self._connection.commit()

    def close(self) -> None:
        if self._connection is None:
            return
        with self._lock:
            self._connection.close()
            self._connection = None


__all__ = [
    "PLANNING_CACHE_VERSION",
    "PlanningStore",
    "normalize_idea_for_cache",
    "planning_cache_key",
]
