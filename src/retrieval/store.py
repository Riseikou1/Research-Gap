"""Short-lived persistent cache for provider retrieval results."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from threading import RLock
from typing import Callable

from src.models.paper import Paper
from src.persistence.cache import PersistentCache


class RetrievalStore:
    """Store raw provider result lists with an explicit freshness window."""

    def __init__(
        self,
        path: str | Path | None,
        *,
        ttl_seconds: float,
        clock: Callable[[], float] | None = None,
        database_url: str | None = None,
    ) -> None:
        if ttl_seconds <= 0:
            raise ValueError("retrieval cache TTL must be positive")

        self.path = Path(path) if path is not None else None
        self.ttl_seconds = ttl_seconds
        self._clock = clock or time.time
        self._lock = RLock()
        self._connection: sqlite3.Connection | None = None
        self._durable = (
            PersistentCache(path, database_url=database_url)
            if path is not None and database_url
            else None
        )

        if self.path is None or self._durable is not None:
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
            CREATE TABLE IF NOT EXISTS retrieval_cache (
                cache_key TEXT PRIMARY KEY,
                stored_at REAL NOT NULL,
                payload TEXT NOT NULL
            )
            """
        )
        self._connection.commit()

    def get(self, cache_key: str) -> list[Paper] | None:
        if self._connection is None and self._durable is None:
            return None

        now = self._clock()
        if self._durable is not None:
            try:
                row = self._durable.get("retrieval", cache_key)
                if row is None:
                    return None
                stored_at, payload = row
                if now - stored_at >= self.ttl_seconds:
                    self._durable.delete("retrieval", cache_key)
                    return None
            except (TypeError, ValueError):
                return None
            return self._decode(payload)
        try:
            with self._lock:
                row = self._connection.execute(
                    "SELECT stored_at, payload FROM retrieval_cache WHERE cache_key = ?",
                    (cache_key,),
                ).fetchone()

                if row is None:
                    return None

                stored_at, payload = row
                if now - float(stored_at) >= self.ttl_seconds:
                    self._connection.execute(
                        "DELETE FROM retrieval_cache WHERE cache_key = ?",
                        (cache_key,),
                    )
                    self._connection.commit()
                    return None
        except (sqlite3.DatabaseError, TypeError, ValueError):
            return None

        return self._decode(payload)

    @staticmethod
    def _decode(payload: str) -> list[Paper] | None:
        try:
            raw_items = json.loads(payload)
            if not isinstance(raw_items, list):
                return None
            # JSON validation intentionally parses serialized dates/enums while
            # retaining the model's strict validation for the decoded shape.
            return [
                Paper.model_validate_json(json.dumps(item, ensure_ascii=False))
                for item in raw_items
            ]
        except (TypeError, ValueError, json.JSONDecodeError):
            return None

    def put(self, cache_key: str, papers: list[Paper]) -> None:
        if self._connection is None and self._durable is None:
            return

        encoded_items = []
        for paper in papers:
            item = paper.model_dump(mode="json")
            for computed in ("matched_queries", "retrieval_modes", "retrieved_by"):
                item.pop(computed, None)
            encoded_items.append(item)

        payload = json.dumps(encoded_items, ensure_ascii=False, separators=(",", ":"))
        if self._durable is not None:
            self._durable.put("retrieval", cache_key, payload, stored_at=self._clock())
            return

        with self._lock:
            self._connection.execute(
                """
                INSERT OR REPLACE INTO retrieval_cache
                    (cache_key, stored_at, payload)
                VALUES (?, ?, ?)
                """,
                (
                    cache_key,
                    self._clock(),
                    payload,
                ),
            )
            self._connection.commit()

    def close(self) -> None:
        if self._connection is None:
            return
        with self._lock:
            self._connection.close()
            self._connection = None


__all__ = ["RetrievalStore"]
