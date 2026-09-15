"""Command-line entry point for applying analysis-history migrations."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.config import database_path, database_url

from .database import Database


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply Research GAP database migrations")
    parser.add_argument("--database", type=Path, help="Override RESEARCH_GAP_DATABASE_PATH")
    args = parser.parse_args()
    database = (
        Database(args.database)
        if args.database is not None
        else Database(database_path(), url=database_url())
    )
    applied = database.migrate()
    if applied:
        print("Applied migrations: " + ", ".join(applied))
    else:
        print("Database schema is up to date.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
