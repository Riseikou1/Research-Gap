"""Offline tests for bounded background analysis execution."""

from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path
from threading import Event, Lock

from src.application.jobs import AnalysisJobRunner, safe_error_message
from src.persistence.database import Database
from src.persistence.models import NewAnalysis
from src.persistence.repositories import AnalysisRepository


class JobRunnerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        database = Database(Path(self.directory.name) / "jobs.sqlite3")
        database.migrate()
        self.repository = AnalysisRepository(database)
        self.runners: list[AnalysisJobRunner] = []

    def tearDown(self) -> None:
        for runner in self.runners:
            runner.shutdown()
        self.directory.cleanup()

    def create(self, analysis_id: str) -> None:
        self.repository.create(NewAnalysis(
            analysis_id=analysis_id,
            research_idea=f"idea {analysis_id}",
            decomposer="deterministic",
            query_generator="deterministic",
            paper_limit=10,
            configuration={"pipeline_version": "m8-v1"},
        ))

    def runner(self, executor, max_workers: int = 2) -> AnalysisJobRunner:
        runner = AnalysisJobRunner(self.repository, executor, max_workers=max_workers)
        self.runners.append(runner)
        return runner

    def wait_for_terminal(self, analysis_id: str, timeout: float = 2.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            record = self.repository.get(analysis_id)
            if record is not None and record.terminal:
                return record
            time.sleep(0.01)
        self.fail(f"analysis {analysis_id} did not finish")

    def test_pending_to_running_to_completed(self) -> None:
        self.create("complete")
        statuses = []

        def execute(record):
            statuses.append(self.repository.get(record.analysis_id).status)
            return {"papers": [{"id": "W1"}]}

        self.assertTrue(self.runner(execute).submit("complete"))
        completed = self.wait_for_terminal("complete")
        self.assertEqual(statuses, ["running"])
        self.assertEqual(completed.status, "completed")
        self.assertEqual(completed.result["papers"][0]["id"], "W1")

    def test_exception_becomes_failed_record(self) -> None:
        self.create("failed")

        def fail(_record):
            raise RuntimeError("provider unavailable")

        self.runner(fail).submit("failed")
        record = self.wait_for_terminal("failed")
        self.assertEqual(record.status, "failed")
        self.assertEqual(record.error_message, "RuntimeError: provider unavailable")

    def test_duplicate_execution_is_prevented(self) -> None:
        self.create("duplicate")
        release = Event()
        calls = []

        def execute(record):
            calls.append(record.analysis_id)
            release.wait(1)
            return {}

        runner = self.runner(execute, max_workers=1)
        self.assertTrue(runner.submit("duplicate"))
        self.assertFalse(runner.submit("duplicate"))
        release.set()
        self.wait_for_terminal("duplicate")
        self.assertEqual(calls, ["duplicate"])

    def test_concurrency_is_bounded(self) -> None:
        for index in range(5):
            self.create(f"job-{index}")
        release = Event()
        lock = Lock()
        active = peak = 0
        two_started = Event()

        def execute(_record):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
                if active == 2:
                    two_started.set()
            release.wait(1)
            with lock:
                active -= 1
            return {}

        runner = self.runner(execute, max_workers=2)
        for index in range(5):
            runner.submit(f"job-{index}")
        self.assertTrue(two_started.wait(1))
        time.sleep(0.05)
        self.assertEqual(peak, 2)
        release.set()
        for index in range(5):
            self.wait_for_terminal(f"job-{index}")

    def test_recovery_fails_interrupted_and_resumes_pending(self) -> None:
        self.create("interrupted")
        self.create("pending")
        self.repository.mark_running("interrupted")
        runner = self.runner(lambda record: {"idea": record.research_idea})
        runner.recover()
        self.assertEqual(self.repository.get("interrupted").status, "failed")
        self.assertEqual(self.wait_for_terminal("pending").status, "completed")

    def test_queued_job_can_be_cancelled_without_deadlock(self) -> None:
        self.create("running")
        self.create("queued")
        release = Event()
        calls = []

        def execute(record):
            calls.append(record.analysis_id)
            release.wait(1)
            return {}

        runner = self.runner(execute, max_workers=1)
        runner.submit("running")
        runner.submit("queued")
        self.assertTrue(runner.cancel("queued"))
        release.set()
        self.wait_for_terminal("running")
        time.sleep(0.05)
        self.assertEqual(calls, ["running"])
        self.assertEqual(self.repository.get("queued").status, "pending")

    def test_persisted_error_text_redacts_credentials(self) -> None:
        message = safe_error_message(RuntimeError("api_key=sk-secretvalue123 request failed"))
        self.assertNotIn("sk-secretvalue123", message)
        self.assertIn("[redacted]", message)


if __name__ == "__main__":
    unittest.main()
