"""Command-line entry point for applying analysis-history migrations."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.config import Settings

from .database import Database


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply Research GAP database migrations")
    parser.add_argument("--database", type=Path, help="Override RESEARCH_GAP_DATABASE_PATH")
    args = parser.parse_args()
    settings = Settings.from_env()
    database = (
        Database(args.database)
        if args.database is not None
        else Database(settings.analysis_database_path, url=settings.database_url)
    )
    applied = database.migrate()
    if applied:
        print("Applied migrations: " + ", ".join(applied))
    else:
        print("Database schema is up to date.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
