"""Small repository for durable analysis records."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone

from .database import Database
from .models import AnalysisHistoryItem, AnalysisRecord, NewAnalysis, parse_timestamp


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


class AnalysisRepository:
    def __init__(self, database: Database) -> None:
        self.database = database

    def create(self, analysis: NewAnalysis) -> AnalysisRecord:
        created_at = utc_now()
        with self.database.connect() as connection:
            connection.execute(
                """
                INSERT INTO analyses (
                    analysis_id, research_idea, status, decomposer, query_generator,
                    paper_limit, created_at, configuration_json
                ) VALUES (?, ?, 'pending', ?, ?, ?, ?, ?)
                """,
                (
                    analysis.analysis_id,
                    analysis.research_idea,
                    analysis.decomposer,
                    analysis.query_generator,
                    analysis.paper_limit,
                    created_at.isoformat(),
                    _json(analysis.configuration),
                ),
            )
            connection.commit()
        record = self.get(analysis.analysis_id)
        if record is None:
            raise RuntimeError("analysis was not persisted")
        return record

    def get(self, analysis_id: str) -> AnalysisRecord | None:
        with self.database.connect() as connection:
            row = connection.execute(
                "SELECT * FROM analyses WHERE analysis_id = ?", (analysis_id,)
            ).fetchone()
        return self._record(row) if row else None

    def list_recent(self, limit: int = 20) -> list[AnalysisHistoryItem]:
        if not 1 <= limit <= 100:
            raise ValueError("analysis history limit must be between 1 and 100")
        with self.database.connect() as connection:
            rows = connection.execute(
                """
                SELECT analysis_id, research_idea, status, created_at, completed_at
                FROM analyses ORDER BY created_at DESC, analysis_id DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [
            AnalysisHistoryItem(
                analysis_id=row["analysis_id"],
                research_idea=row["research_idea"],
                status=row["status"],
                created_at=parse_timestamp(row["created_at"]),
                completed_at=parse_timestamp(row["completed_at"]),
            )
            for row in rows
        ]

    def list_pending(self) -> list[AnalysisRecord]:
        with self.database.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM analyses WHERE status = 'pending' ORDER BY created_at"
            ).fetchall()
        return [self._record(row) for row in rows]

    def mark_running(self, analysis_id: str) -> bool:
        with self.database.connect() as connection:
            cursor = connection.execute(
                """
                UPDATE analyses SET status = 'running', started_at = ?, completed_at = NULL
                WHERE analysis_id = ? AND status = 'pending'
                """,
                (utc_now().isoformat(), analysis_id),
            )
            connection.commit()
            return cursor.rowcount == 1

    def mark_completed(self, analysis_id: str, result: dict[str, object]) -> bool:
        with self.database.connect() as connection:
            cursor = connection.execute(
                """
                UPDATE analyses
                SET status = 'completed', completed_at = ?, result_json = ?, error_message = NULL
                WHERE analysis_id = ? AND status = 'running'
                """,
                (utc_now().isoformat(), _json(result), analysis_id),
            )
            connection.commit()
            return cursor.rowcount == 1

    def mark_failed(self, analysis_id: str, error_message: str) -> bool:
        message = " ".join(error_message.split())[:1000] or "Analysis failed."
        now = utc_now().isoformat()
        with self.database.connect() as connection:
            cursor = connection.execute(
                """
                UPDATE analyses
                SET status = 'failed', started_at = COALESCE(started_at, ?), completed_at = ?,
                    result_json = NULL, error_message = ?
                WHERE analysis_id = ? AND status IN ('pending', 'running')
                """,
                (now, now, message, analysis_id),
            )
            connection.commit()
            return cursor.rowcount == 1

    def fail_interrupted(self) -> int:
        now = utc_now().isoformat()
        with self.database.connect() as connection:
            cursor = connection.execute(
                """
                UPDATE analyses SET status = 'failed', completed_at = ?, result_json = NULL,
                    error_message = 'Analysis interrupted by application shutdown.'
                WHERE status = 'running'
                """,
                (now,),
            )
            connection.commit()
            return cursor.rowcount

    def delete(self, analysis_id: str) -> bool:
        with self.database.connect() as connection:
            cursor = connection.execute(
                "DELETE FROM analyses WHERE analysis_id = ?", (analysis_id,)
            )
            connection.commit()
            return cursor.rowcount == 1

    @staticmethod
    def _record(row: sqlite3.Row) -> AnalysisRecord:
        return AnalysisRecord(
            analysis_id=row["analysis_id"],
            research_idea=row["research_idea"],
            status=row["status"],
            decomposer=row["decomposer"],
            query_generator=row["query_generator"],
            paper_limit=row["paper_limit"],
            created_at=parse_timestamp(row["created_at"]),
            started_at=parse_timestamp(row["started_at"]),
            completed_at=parse_timestamp(row["completed_at"]),
            configuration=json.loads(row["configuration_json"]),
            result=json.loads(row["result_json"]) if row["result_json"] else None,
            error_message=row["error_message"],
        )
