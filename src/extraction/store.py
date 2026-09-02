"""Small persistent store for structured paper evidence."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from threading import RLock

from src.models.paper import Paper

from .evidence import PaperEvidence


class EvidenceStore:
    """Read and write evidence using a versioned SQLite cache key."""

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
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS evidence_cache (
                paper_id TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                model TEXT NOT NULL,
                schema_version INTEGER NOT NULL,
                payload TEXT NOT NULL,
                PRIMARY KEY (paper_id, content_hash, model, schema_version)
            )
            """
        )
        self._connection.commit()

    @staticmethod
    def content_hash(paper: Paper) -> str:
        """Hash the evidence-bearing title and abstract, not only the ID."""

        content = "\0".join((paper.title.strip(), paper.abstract or ""))
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def get(
        self,
        *,
        paper_id: str,
        content_hash: str,
        model: str,
        schema_version: int,
    ) -> PaperEvidence | None:
        if self._connection is None:
            return None

        with self._lock:
            row = self._connection.execute(
                """
                SELECT payload
                FROM evidence_cache
                WHERE paper_id = ?
                  AND content_hash = ?
                  AND model = ?
                  AND schema_version = ?
                """,
                (paper_id, content_hash, model, schema_version),
            ).fetchone()

        if row is None:
            return None

        try:
            return PaperEvidence.model_validate(json.loads(row[0]))
        except (TypeError, ValueError, json.JSONDecodeError):
            # A corrupt cache entry is a miss. The next successful extraction
            # will replace it without making cache state part of correctness.
            return None

    def put(
        self,
        evidence: PaperEvidence,
        *,
        content_hash: str,
        model: str,
        schema_version: int,
    ) -> None:
        if self._connection is None:
            return

        payload = json.dumps(
            evidence.model_dump(mode="json"),
            ensure_ascii=False,
            separators=(",", ":"),
        )

        with self._lock:
            self._connection.execute(
                """
                INSERT OR REPLACE INTO evidence_cache
                    (paper_id, content_hash, model, schema_version, payload)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    evidence.paper_id,
                    content_hash,
                    model,
                    schema_version,
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


__all__ = ["EvidenceStore"]
