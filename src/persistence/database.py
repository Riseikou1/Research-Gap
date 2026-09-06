"""SQLite connections and ordered schema migrations."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from threading import Lock
from typing import Iterator


class MigrationError(RuntimeError):
    """Raised when the durable analysis schema cannot be migrated."""


class Database:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._migration_lock = Lock()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 30000")
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
        finally:
            connection.close()

    def migrate(self) -> list[str]:
        """Apply each bundled SQL migration exactly once, in filename order."""
        migration_dir = Path(__file__).with_name("migrations")
        files = sorted(migration_dir.glob("[0-9][0-9][0-9][0-9]_*.sql"))
        if not files:
            raise MigrationError("no database migrations were found")

        applied_now: list[str] = []
        with self._migration_lock, self.connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA synchronous = NORMAL")
            connection.execute(
                "CREATE TABLE IF NOT EXISTS schema_migrations ("
                "version TEXT PRIMARY KEY, applied_at TEXT NOT NULL)"
            )
            applied = {
                row["version"]
                for row in connection.execute("SELECT version FROM schema_migrations")
            }
            for path in files:
                version = path.stem
                if version in applied:
                    continue
                try:
                    connection.executescript(path.read_text(encoding="utf-8"))
                    connection.execute(
                        "INSERT INTO schema_migrations (version, applied_at) "
                        "VALUES (?, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))",
                        (version,),
                    )
                    connection.commit()
                except (OSError, sqlite3.DatabaseError) as exc:
                    connection.rollback()
                    raise MigrationError(f"failed to apply migration {version}: {exc}") from exc
                applied_now.append(version)
        return applied_now

    def check(self) -> None:
        with self.connect() as connection:
            connection.execute("SELECT 1 FROM analyses LIMIT 1").fetchone()
