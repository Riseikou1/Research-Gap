"""Database connections and ordered schema migrations."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Iterator

import psycopg
from psycopg.rows import dict_row


class MigrationError(RuntimeError):
    """Raised when the durable analysis schema cannot be migrated."""


class Database:
    def __init__(
        self,
        path: str | Path,
        *,
        url: str | None = None,
    ) -> None:
        self.path = Path(path)
        self.url = url.strip() if url else None
        self._migration_lock = Lock()

    @property
    def is_postgres(self) -> bool:
        return self.url is not None

    @contextmanager
    def connect(self) -> Iterator[Any]:
        if self.is_postgres:
            connection = psycopg.connect(
                self.url,
                row_factory=dict_row,
            )
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)

            connection = sqlite3.connect(
                self.path,
                timeout=30.0,
            )

            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA busy_timeout = 30000")
            connection.execute("PRAGMA foreign_keys = ON")

        try:
            yield connection
        finally:
            connection.close()

    def migrate(self) -> list[str]:
        """Apply bundled SQL migrations exactly once, in filename order."""

        migration_dir = Path(__file__).with_name("migrations")
        files = sorted(
            migration_dir.glob("[0-9][0-9][0-9][0-9]_*.sql")
        )

        if not files:
            raise MigrationError("no database migrations were found")

        applied_now: list[str] = []

        with self._migration_lock, self.connect() as connection:
            if not self.is_postgres:
                connection.execute("PRAGMA journal_mode = WAL")
                connection.execute("PRAGMA synchronous = NORMAL")

            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version TEXT PRIMARY KEY,
                    applied_at TEXT NOT NULL
                )
                """
            )

            applied = {
                row["version"]
                for row in connection.execute(
                    "SELECT version FROM schema_migrations"
                ).fetchall()
            }

            for path in files:
                version = path.stem

                if version in applied:
                    continue

                try:
                    sql = path.read_text(encoding="utf-8")

                    if self.is_postgres:
                        connection.execute(sql)

                        connection.execute(
                            """
                            INSERT INTO schema_migrations (
                                version,
                                applied_at
                            )
                            VALUES (%s, %s)
                            """,
                            (
                                version,
                                datetime.now(timezone.utc).isoformat(),
                            ),
                        )

                    else:
                        connection.executescript(sql)

                        connection.execute(
                            """
                            INSERT INTO schema_migrations (
                                version,
                                applied_at
                            )
                            VALUES (
                                ?,
                                strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                            )
                            """,
                            (version,),
                        )

                    connection.commit()

                except (OSError, sqlite3.DatabaseError, psycopg.Error) as exc:
                    connection.rollback()
                    raise MigrationError(
                        f"failed to apply migration {version}: {exc}"
                    ) from exc

                applied_now.append(version)

        return applied_now

    def check(self) -> None:
        with self.connect() as connection:
            connection.execute(
                "SELECT 1 FROM analyses LIMIT 1"
            ).fetchone()
