"""Database connections, SQL adaptation, transactions, and schema migrations."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Iterator, Sequence

import psycopg
from psycopg.rows import dict_row


DATABASE_ERRORS = (OSError, sqlite3.DatabaseError, psycopg.Error)


class MigrationError(RuntimeError):
    """Raised when the durable analysis schema cannot be migrated."""


def _postgres_sql(sql: str) -> str:
    """Convert qmark parameters outside SQL strings/comments to psycopg parameters."""

    converted: list[str] = []
    index = 0
    state = "sql"
    while index < len(sql):
        char = sql[index]
        following = sql[index + 1] if index + 1 < len(sql) else ""

        if state == "sql":
            if char == "'":
                state = "single_quote"
            elif char == '"':
                state = "double_quote"
            elif char == "-" and following == "-":
                converted.extend((char, following))
                index += 2
                state = "line_comment"
                continue
            elif char == "/" and following == "*":
                converted.extend((char, following))
                index += 2
                state = "block_comment"
                continue
            elif char == "?":
                converted.append("%s")
                index += 1
                continue
        elif state == "single_quote":
            if char == "'" and following == "'":
                converted.extend((char, following))
                index += 2
                continue
            if char == "'":
                state = "sql"
        elif state == "double_quote":
            if char == '"' and following == '"':
                converted.extend((char, following))
                index += 2
                continue
            if char == '"':
                state = "sql"
        elif state == "line_comment":
            if char in "\r\n":
                state = "sql"
        elif state == "block_comment" and char == "*" and following == "/":
            converted.extend((char, following))
            index += 2
            state = "sql"
            continue

        converted.append(char)
        index += 1

    return "".join(converted)


def _sql_statements(script: str) -> Iterator[str]:
    """Yield complete SQL statements without SQLite's implicit-commit executescript."""

    statement: list[str] = []
    for line in script.splitlines(keepends=True):
        statement.append(line)
        candidate = "".join(statement)
        if sqlite3.complete_statement(candidate):
            if candidate.strip():
                yield candidate
            statement = []
    remainder = "".join(statement)
    if remainder.strip():
        raise MigrationError("migration ended with an incomplete SQL statement")


class DatabaseConnection:
    """Expose one mapping-row, qmark-parameter API for both supported drivers."""

    def __init__(self, connection: Any, *, postgres: bool) -> None:
        self._connection = connection
        self.is_postgres = postgres

    def execute(self, sql: str, parameters: Sequence[object] | None = None) -> Any:
        if parameters is None:
            return self._connection.execute(sql)
        query = _postgres_sql(sql) if self.is_postgres else sql
        return self._connection.execute(query, parameters)

    def commit(self) -> None:
        self._connection.commit()

    def rollback(self) -> None:
        self._connection.rollback()

    def close(self) -> None:
        self._connection.close()


class Database:
    def __init__(
        self,
        path: str | Path,
        *,
        url: str | None = None,
    ) -> None:
        self.path = Path(path)
        self.url = url.strip() if url and url.strip() else None
        self._migration_lock = Lock()

    @property
    def is_postgres(self) -> bool:
        return self.url is not None

    @contextmanager
    def connect(self) -> Iterator[DatabaseConnection]:
        if self.is_postgres:
            native_connection = psycopg.connect(
                self.url,
                row_factory=dict_row,
            )
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            native_connection = sqlite3.connect(self.path, timeout=30.0)
            native_connection.row_factory = sqlite3.Row
            native_connection.execute("PRAGMA busy_timeout = 30000")
            native_connection.execute("PRAGMA foreign_keys = ON")

        connection = DatabaseConnection(native_connection, postgres=self.is_postgres)
        try:
            yield connection
        finally:
            connection.close()

    @contextmanager
    def transaction(
        self,
        connection: DatabaseConnection,
        *,
        lock_keys: Sequence[str] = (),
    ) -> Iterator[None]:
        """Run an atomic write, serializing named resources on PostgreSQL."""

        self.begin(connection, lock_keys=lock_keys)
        try:
            yield
            connection.commit()
        except BaseException:
            connection.rollback()
            raise

    def begin(
        self,
        connection: DatabaseConnection,
        *,
        lock_keys: Sequence[str] = (),
    ) -> None:
        """Begin a write transaction and acquire its logical PostgreSQL locks."""

        connection.execute("BEGIN" if self.is_postgres else "BEGIN IMMEDIATE")
        for key in sorted(set(lock_keys)):
            self.lock(connection, key)

    def lock(self, connection: DatabaseConnection, key: str) -> None:
        """Acquire a transaction-scoped PostgreSQL advisory lock for a logical key."""

        if self.is_postgres:
            connection.execute(
                "SELECT pg_advisory_xact_lock(hashtextextended(?, 0))",
                (key,),
            )

    def migrate(self) -> list[str]:
        """Apply bundled SQL migrations exactly once, in filename order."""

        migration_dir = Path(__file__).with_name("migrations")
        files = sorted(migration_dir.glob("[0-9][0-9][0-9][0-9]_*.sql"))
        if not files:
            raise MigrationError("no database migrations were found")

        applied_now: list[str] = []
        current_version = "unknown"
        with self._migration_lock, self.connect() as connection:
            if not self.is_postgres:
                connection.execute("PRAGMA journal_mode = WAL")
                connection.execute("PRAGMA synchronous = NORMAL")

            try:
                with self.transaction(connection, lock_keys=("schema-migrations",)):
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
                        current_version = path.stem
                        if current_version in applied:
                            continue
                        sql = path.read_text(encoding="utf-8")
                        for statement in _sql_statements(sql):
                            connection.execute(statement)
                        connection.execute(
                            """
                            INSERT INTO schema_migrations (version, applied_at)
                            VALUES (?, ?)
                            """,
                            (current_version, datetime.now(timezone.utc).isoformat()),
                        )
                        applied_now.append(current_version)
            except DATABASE_ERRORS as exc:
                raise MigrationError(
                    f"failed to apply migration {current_version}: {exc}"
                ) from exc

        return applied_now

    def check(self) -> None:
        with self.connect() as connection:
            connection.execute("SELECT 1 FROM analyses LIMIT 1").fetchone()
