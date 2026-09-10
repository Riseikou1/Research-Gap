"""Driver-independent checks for the SQLite/PostgreSQL persistence boundary."""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.admin import bootstrap
from src.api.app import create_app
from src.config import Settings
from src.persistence.database import Database, DatabaseConnection, _postgres_sql
from src.persistence import migrate


class FakeNativeConnection:
    def __init__(self) -> None:
        self.executions: list[tuple[str, object | None]] = []
        self.committed = False
        self.rolled_back = False
        self.closed = False

    def execute(self, sql: str, parameters=None):
        self.executions.append((sql, parameters))
        return self

    def commit(self) -> None:
        self.committed = True

    def rollback(self) -> None:
        self.rolled_back = True

    def close(self) -> None:
        self.closed = True


def clean_settings(path: Path) -> Settings:
    with patch.dict(os.environ, {}, clear=True):
        settings = Settings.from_env()
    return replace(settings, analysis_database_path=path)


def test_sqlite_is_the_fallback_and_application_selects_postgres(tmp_path: Path):
    sqlite_settings = clean_settings(tmp_path / "fallback.sqlite")
    sqlite_app = create_app(settings=sqlite_settings)
    assert sqlite_app.state.components.database.is_postgres is False

    postgres_settings = replace(
        sqlite_settings,
        database_url="postgresql://example.invalid/research_gap",
    )
    postgres_app = create_app(settings=postgres_settings)
    assert postgres_app.state.components.database.is_postgres is True
    assert postgres_app.state.components.database.path == tmp_path / "fallback.sqlite"


def test_qmark_conversion_ignores_quoted_text_and_comments():
    sql = "SELECT '?', \"?\", value FROM records WHERE first=? AND second=? -- ?\n/* ? */"
    assert _postgres_sql(sql) == (
        "SELECT '?', \"?\", value FROM records "
        "WHERE first=%s AND second=%s -- ?\n/* ? */"
    )


def test_postgres_connection_adapts_parameters_and_uses_mapping_rows(tmp_path: Path):
    native = FakeNativeConnection()
    with patch("src.persistence.database.psycopg.connect", return_value=native) as connect:
        database = Database(
            tmp_path / "unused.sqlite",
            url="postgresql://example.invalid/research_gap",
        )
        with database.connect() as connection:
            connection.execute("SELECT * FROM records WHERE record_id=?", ("one",))

    assert native.executions == [
        ("SELECT * FROM records WHERE record_id=%s", ("one",)),
    ]
    assert native.closed is True
    assert connect.call_args.kwargs["row_factory"] is not None


def test_postgres_transactions_use_advisory_locks_not_sqlite_write_lock():
    native = FakeNativeConnection()
    connection = DatabaseConnection(native, postgres=True)
    database = Database("unused.sqlite", url="postgresql://example.invalid/research_gap")

    with database.transaction(connection, lock_keys=("credit:user",)):
        connection.execute("UPDATE accounts SET status=?", ("active",))

    statements = [sql for sql, _parameters in native.executions]
    assert statements[0] == "BEGIN"
    assert statements[1] == "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))"
    assert statements[2] == "UPDATE accounts SET status=%s"
    assert all("BEGIN IMMEDIATE" not in sql for sql in statements)
    assert native.committed is True


def test_migration_cli_uses_configured_url_without_printing_it(tmp_path: Path, capsys):
    settings = replace(
        clean_settings(tmp_path / "fallback.sqlite"),
        database_url="postgresql://example.invalid/private_database",
    )
    database = MagicMock()
    database.migrate.return_value = []
    with patch.object(migrate.Settings, "from_env", return_value=settings), patch.object(
        migrate, "Database", return_value=database
    ) as database_type, patch("sys.argv", ["migrate"]):
        assert migrate.main() == 0

    database_type.assert_called_once_with(
        settings.analysis_database_path,
        url=settings.database_url,
    )
    assert settings.database_url not in capsys.readouterr().out


def test_migration_cli_database_override_explicitly_selects_sqlite(tmp_path: Path):
    settings = replace(
        clean_settings(tmp_path / "configured.sqlite"),
        database_url="postgresql://example.invalid/private_database",
    )
    override = tmp_path / "override.sqlite"
    database = MagicMock()
    database.migrate.return_value = []
    with patch.object(migrate.Settings, "from_env", return_value=settings), patch.object(
        migrate, "Database", return_value=database
    ) as database_type, patch("sys.argv", ["migrate", "--database", str(override)]):
        assert migrate.main() == 0

    database_type.assert_called_once_with(override)


def test_admin_bootstrap_uses_configured_database_url(tmp_path: Path, capsys):
    settings = replace(
        clean_settings(tmp_path / "fallback.sqlite"),
        database_url="postgresql://example.invalid/private_database",
    )
    database = MagicMock()
    security = MagicMock()
    security.admin_account.return_value = {
        "display_name": "existing-owner",
        "email": "owner@example.test",
    }
    with patch.object(bootstrap.Settings, "from_env", return_value=settings), patch.object(
        bootstrap, "Database", return_value=database
    ) as database_type, patch.object(
        bootstrap, "SecurityRepository", return_value=security
    ), patch("sys.argv", ["bootstrap", "--email", "owner@example.test"]):
        assert bootstrap.main() == 0

    database_type.assert_called_once_with(
        settings.analysis_database_path,
        url=settings.database_url,
    )
    database.migrate.assert_called_once_with()
    assert settings.database_url not in capsys.readouterr().out
