"""Small provider-cache adapter shared by SQLite and PostgreSQL deployments."""

from __future__ import annotations

from pathlib import Path

from .database import Database


class PersistentCache:
    """Store opaque versioned payloads in existing durable infrastructure."""

    def __init__(self, path: str | Path, *, database_url: str | None = None) -> None:
        self.database = Database(path, url=database_url)
        with self.database.connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS provider_cache (
                    namespace TEXT NOT NULL,
                    cache_key TEXT NOT NULL,
                    stored_at DOUBLE PRECISION NOT NULL,
                    payload TEXT NOT NULL,
                    PRIMARY KEY (namespace, cache_key)
                )
                """
            )
            connection.commit()

    def get(self, namespace: str, cache_key: str) -> tuple[float, str] | None:
        with self.database.connect() as connection:
            row = connection.execute(
                "SELECT stored_at, payload FROM provider_cache WHERE namespace=? AND cache_key=?",
                (namespace, cache_key),
            ).fetchone()
        if row is None:
            return None
        return float(row["stored_at"]), str(row["payload"])

    def put(self, namespace: str, cache_key: str, payload: str, *, stored_at: float) -> None:
        with self.database.connect() as connection:
            connection.execute(
                """
                INSERT INTO provider_cache(namespace, cache_key, stored_at, payload)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(namespace, cache_key) DO UPDATE SET
                    stored_at=excluded.stored_at,
                    payload=excluded.payload
                """,
                (namespace, cache_key, stored_at, payload),
            )
            connection.commit()

    def delete(self, namespace: str, cache_key: str) -> None:
        with self.database.connect() as connection:
            connection.execute(
                "DELETE FROM provider_cache WHERE namespace=? AND cache_key=?",
                (namespace, cache_key),
            )
            connection.commit()


__all__ = ["PersistentCache"]
