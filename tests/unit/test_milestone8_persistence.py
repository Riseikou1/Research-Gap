"""Offline tests for durable analysis records and schema migrations."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.persistence.database import Database
from src.persistence.models import NewAnalysis
from src.persistence.repositories import AnalysisRepository


def new_analysis(analysis_id: str = "analysis-1") -> NewAnalysis:
    return NewAnalysis(
        analysis_id=analysis_id,
        research_idea="federated learning for traffic control",
        decomposer="deterministic",
        query_generator="deterministic",
        paper_limit=20,
        configuration={"pipeline_version": "m8-v1", "models": {"embedding": None}},
    )


class PersistenceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.database = Database(Path(self.directory.name) / "analyses.sqlite3")
        self.repository = AnalysisRepository(self.database)
        self.database.migrate()

    def tearDown(self) -> None:
        self.directory.cleanup()

    def test_migrations_are_ordered_and_idempotent(self) -> None:
        self.assertEqual(self.database.migrate(), [])
        with self.database.connect() as connection:
            versions = connection.execute(
                "SELECT version FROM schema_migrations ORDER BY version"
            ).fetchall()
        self.assertEqual(
            [row["version"] for row in versions],
            ["0001_create_analyses", "0002_web_accounts_billing"],
        )

    def test_create_and_load_round_trip(self) -> None:
        created = self.repository.create(new_analysis())
        loaded = self.repository.get(created.analysis_id)
        self.assertEqual(loaded, created)
        self.assertEqual(loaded.status, "pending")
        self.assertEqual(loaded.configuration["pipeline_version"], "m8-v1")

    def test_running_and_completed_result_round_trip(self) -> None:
        self.repository.create(new_analysis())
        self.assertTrue(self.repository.mark_running("analysis-1"))
        result = {
            "queries": [{"text": "traffic control"}],
            "papers": [{"id": "W1"}],
            "evidence": [],
            "landscape": {"total_papers": 1},
            "gaps": [],
            "idea_assessment": {"label": "uncertain"},
        }
        self.assertTrue(self.repository.mark_completed("analysis-1", result))
        loaded = self.repository.get("analysis-1")
        self.assertEqual(loaded.status, "completed")
        self.assertIsNotNone(loaded.started_at)
        self.assertIsNotNone(loaded.completed_at)
        self.assertEqual(loaded.result, result)
        self.assertIsNone(loaded.error_message)

    def test_failed_result_is_safe_and_persistent(self) -> None:
        self.repository.create(new_analysis())
        self.assertTrue(self.repository.mark_failed("analysis-1", " provider   unavailable "))
        loaded = self.repository.get("analysis-1")
        self.assertEqual(loaded.status, "failed")
        self.assertEqual(loaded.error_message, "provider unavailable")
        self.assertIsNone(loaded.result)
        self.assertIsNotNone(loaded.completed_at)

    def test_history_is_bounded_and_records_delete(self) -> None:
        for index in range(3):
            self.repository.create(new_analysis(f"analysis-{index}"))
        self.assertEqual(len(self.repository.list_recent(2)), 2)
        self.assertTrue(self.repository.delete("analysis-1"))
        self.assertIsNone(self.repository.get("analysis-1"))
        self.assertFalse(self.repository.delete("missing"))
        with self.assertRaisesRegex(ValueError, "between 1 and 100"):
            self.repository.list_recent(101)

    def test_only_pending_jobs_can_claim_execution(self) -> None:
        self.repository.create(new_analysis())
        self.assertTrue(self.repository.mark_running("analysis-1"))
        self.assertFalse(self.repository.mark_running("analysis-1"))
        self.assertTrue(self.repository.mark_completed("analysis-1", {"papers": []}))
        self.assertFalse(self.repository.mark_failed("analysis-1", "late failure"))


if __name__ == "__main__":
    unittest.main()
