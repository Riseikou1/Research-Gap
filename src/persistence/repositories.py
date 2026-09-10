"""Small repository for durable analysis records."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping

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
                    paper_limit, created_at, configuration_json, owner_kind, owner_id,
                    mode, stage, progress_json, reservation_id
                ) VALUES (?, ?, 'pending', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    analysis.analysis_id,
                    analysis.research_idea,
                    analysis.decomposer,
                    analysis.query_generator,
                    analysis.paper_limit,
                    created_at.isoformat(),
                    _json(analysis.configuration),
                    analysis.owner_kind,
                    analysis.owner_id,
                    analysis.mode,
                    analysis.stage,
                    _json(analysis.progress),
                    analysis.reservation_id,
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

    def get_for_owner(self, analysis_id: str, owner_kind: str, owner_id: str) -> AnalysisRecord | None:
        with self.database.connect() as connection:
            row = connection.execute(
                "SELECT * FROM analyses WHERE analysis_id = ? AND owner_kind = ? AND owner_id = ?",
                (analysis_id, owner_kind, owner_id),
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
                mode=row["mode"] if "mode" in row.keys() else "full",
                stage=row["stage"] if "stage" in row.keys() else "preparing",
            )
            for row in rows
        ]

    def list_for_owner(self, owner_kind: str, owner_id: str, limit: int = 20) -> list[AnalysisHistoryItem]:
        if not 1 <= limit <= 100:
            raise ValueError("analysis history limit must be between 1 and 100")
        with self.database.connect() as connection:
            rows = connection.execute(
                """SELECT analysis_id, research_idea, status, created_at, completed_at, mode, stage
                   FROM analyses WHERE owner_kind = ? AND owner_id = ?
                   ORDER BY created_at DESC, analysis_id DESC LIMIT ?""",
                (owner_kind, owner_id, limit),
            ).fetchall()
        return [AnalysisHistoryItem(
            analysis_id=row["analysis_id"], research_idea=row["research_idea"],
            status=row["status"], created_at=parse_timestamp(row["created_at"]),
            completed_at=parse_timestamp(row["completed_at"]), mode=row["mode"], stage=row["stage"],
        ) for row in rows]

    def count_active_for_owner(self, owner_kind: str, owner_id: str) -> int:
        with self.database.connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS count FROM analyses WHERE owner_kind=? AND owner_id=? "
                "AND status IN ('pending','running')", (owner_kind, owner_id),
            ).fetchone()
        return int(row["count"])

    def update_stage(self, analysis_id: str, stage: str, progress: dict[str, object] | None = None) -> bool:
        with self.database.connect() as connection:
            cursor = connection.execute(
                "UPDATE analyses SET stage=?, progress_json=? WHERE analysis_id=? "
                "AND status IN ('pending','running')",
                (stage[:64], _json(progress or {}), analysis_id),
            )
            connection.commit()
            return cursor.rowcount == 1

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
                UPDATE analyses SET status = 'running', started_at = ?, completed_at = NULL,
                    stage = 'preparing'
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
                SET status = 'completed', completed_at = ?, result_json = ?, error_message = NULL,
                    stage = 'completed'
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
                SET status = 'failed', started_at = COALESCE(started_at, ?), completed_at = ?, stage = 'failed',
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

    def delete_for_owner(self, analysis_id: str, owner_kind: str, owner_id: str) -> bool:
        with self.database.connect() as connection:
            cursor = connection.execute(
                "DELETE FROM analyses WHERE analysis_id=? AND owner_kind=? AND owner_id=?",
                (analysis_id, owner_kind, owner_id),
            )
            connection.commit()
            return cursor.rowcount == 1

    def claim_guest(self, guest_id: str, user_id: str) -> int:
        """Atomically move only this still-valid guest's analyses to its signed-in account."""
        now = utc_now().isoformat()
        with self.database.connect() as connection:
            with self.database.transaction(connection, lock_keys=(f"guest:{guest_id}",)):
                guest = connection.execute(
                    "SELECT claimed_by_user_id, expires_at FROM guest_sessions WHERE guest_id=?",
                    (guest_id,),
                ).fetchone()
                if not guest or guest["claimed_by_user_id"] or guest["expires_at"] <= now:
                    return 0
                cursor = connection.execute(
                    "UPDATE analyses SET owner_kind='user', owner_id=? "
                    "WHERE owner_kind='guest' AND owner_id=?",
                    (user_id, guest_id),
                )
                connection.execute(
                    "UPDATE guest_sessions SET claimed_by_user_id=?, last_seen_at=? WHERE guest_id=?",
                    (user_id, now, guest_id),
                )
                return cursor.rowcount

    def cleanup_expired_guests(self) -> int:
        now = utc_now().isoformat()
        rate_cutoff = (utc_now() - timedelta(days=31)).isoformat()
        with self.database.connect() as connection:
            with self.database.transaction(connection, lock_keys=("guest-cleanup",)):
                cursor = connection.execute(
                    "DELETE FROM analyses WHERE owner_kind='guest' AND owner_id IN "
                    "(SELECT guest_id FROM guest_sessions WHERE expires_at < ?)",
                    (now,),
                )
                connection.execute("DELETE FROM guest_sessions WHERE expires_at < ?", (now,))
                connection.execute(
                    "DELETE FROM rate_events WHERE created_at < ?",
                    (rate_cutoff,),
                )
                return cursor.rowcount

    @staticmethod
    def _record(row: Mapping[str, Any]) -> AnalysisRecord:
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
            owner_kind=row["owner_kind"] if "owner_kind" in row.keys() else None,
            owner_id=row["owner_id"] if "owner_id" in row.keys() else None,
            mode=row["mode"] if "mode" in row.keys() else "full",
            stage=row["stage"] if "stage" in row.keys() else "preparing",
            progress=json.loads(row["progress_json"]) if "progress_json" in row.keys() else {},
            reservation_id=row["reservation_id"] if "reservation_id" in row.keys() else None,
            result=json.loads(row["result_json"]) if row["result_json"] else None,
            error_message=row["error_message"],
        )
