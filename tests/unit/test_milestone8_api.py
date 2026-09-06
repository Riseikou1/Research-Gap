"""Offline HTTP tests for the Milestone 8 local service."""

from __future__ import annotations

import os
import tempfile
import time
import unittest
from dataclasses import replace
from pathlib import Path
from threading import Event
from unittest.mock import patch

try:
    from fastapi.testclient import TestClient
except ModuleNotFoundError as exc:  # Local contributors may not have installed new requirements yet.
    raise unittest.SkipTest(f"FastAPI test dependencies unavailable: {exc}")

from src.api.app import create_app
from src.config import Settings


def result_for(record):
    return {
        "idea": {"original_text": record.research_idea},
        "queries": [{"text": record.research_idea, "strategy": "original"}],
        "candidate_count": 1,
        "papers": [{"id": "W1", "title": "Example"}],
        "evidence": [],
        "landscape": {"total_papers": 1},
        "gaps": [],
        "idea_assessment": {"label": "uncertain"},
    }


class ApiTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings.from_env()
        settings = replace(
            settings,
            openai_api_key="sk-local-test-secret",
            openalex=replace(settings.openalex, api_key="openalex-test-secret"),
            analysis_database_path=Path(self.directory.name) / "api.sqlite3",
            max_analysis_workers=2,
        )
        self.client_context = TestClient(create_app(settings=settings, analysis_executor=result_for))
        self.client = self.client_context.__enter__()

    def tearDown(self) -> None:
        self.client_context.__exit__(None, None, None)
        self.directory.cleanup()

    def wait_for_terminal(self, analysis_id: str) -> dict[str, object]:
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            response = self.client.get(f"/analyses/{analysis_id}")
            if response.json()["status"] in {"completed", "failed"}:
                return response.json()
            time.sleep(0.01)
        self.fail("analysis did not finish")

    def test_create_returns_immediately_and_result_can_be_retrieved(self) -> None:
        response = self.client.post("/analyses", json={"research_idea": "graph learning for drug discovery"})
        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.json()["status"], "pending")
        analysis_id = response.json()["analysis_id"]
        record = self.wait_for_terminal(analysis_id)
        self.assertEqual(record["status"], "completed")
        self.assertEqual(record["result"]["papers"][0]["id"], "W1")
        self.assertEqual(record["configuration"]["pipeline_version"], "m8-v1")
        serialized = str(record["configuration"])
        self.assertNotIn("sk-local-test-secret", serialized)
        self.assertNotIn("openalex-test-secret", serialized)

    def test_history_is_lightweight_and_delete_removes_record(self) -> None:
        response = self.client.post("/analyses", json={"research_idea": "example research idea"})
        analysis_id = response.json()["analysis_id"]
        self.wait_for_terminal(analysis_id)
        history = self.client.get("/analyses?limit=1")
        self.assertEqual(history.status_code, 200)
        self.assertEqual(len(history.json()), 1)
        self.assertNotIn("result", history.json()[0])
        deleted = self.client.delete(f"/analyses/{analysis_id}")
        self.assertEqual(deleted.status_code, 204)
        self.assertEqual(self.client.get(f"/analyses/{analysis_id}").status_code, 404)

    def test_unknown_analysis_and_invalid_requests(self) -> None:
        self.assertEqual(self.client.get("/analyses/missing").status_code, 404)
        self.assertEqual(self.client.delete("/analyses/missing").status_code, 404)
        self.assertEqual(self.client.post("/analyses", json={"research_idea": ""}).status_code, 422)
        self.assertEqual(
            self.client.post("/analyses", json={"research_idea": "valid idea", "unexpected": True}).status_code,
            422,
        )

    def test_health_checks_database(self) -> None:
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})


class FailedApiJobTest(unittest.TestCase):
    def test_worker_exception_does_not_crash_api(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            settings = replace(
                Settings.from_env(),
                analysis_database_path=Path(directory) / "failed.sqlite3",
            )

            def fail(_record):
                raise RuntimeError("safe provider failure")

            with TestClient(create_app(settings=settings, analysis_executor=fail)) as client:
                created = client.post("/analyses", json={"research_idea": "an idea"})
                analysis_id = created.json()["analysis_id"]
                deadline = time.monotonic() + 2
                while time.monotonic() < deadline:
                    record = client.get(f"/analyses/{analysis_id}").json()
                    if record["status"] == "failed":
                        break
                    time.sleep(0.01)
                self.assertEqual(record["status"], "failed")
                self.assertEqual(record["error_message"], "RuntimeError: safe provider failure")
                self.assertEqual(client.get("/health").status_code, 200)


class ActiveApiJobTest(unittest.TestCase):
    def test_post_does_not_wait_and_running_job_is_not_deleted(self) -> None:
        started, release = Event(), Event()
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            settings = replace(
                Settings.from_env(),
                analysis_database_path=Path(directory) / "active.sqlite3",
                max_analysis_workers=1,
            )

            def block(record):
                started.set()
                release.wait(2)
                return result_for(record)

            with TestClient(create_app(settings=settings, analysis_executor=block)) as client:
                try:
                    created = client.post("/analyses", json={"research_idea": "long analysis"})
                    self.assertEqual(created.status_code, 201)
                    analysis_id = created.json()["analysis_id"]
                    self.assertTrue(started.wait(1))
                    self.assertEqual(client.delete(f"/analyses/{analysis_id}").status_code, 409)
                finally:
                    release.set()
                deadline = time.monotonic() + 2
                while time.monotonic() < deadline:
                    if client.get(f"/analyses/{analysis_id}").json()["status"] == "completed":
                        break
                    time.sleep(0.01)
                self.assertEqual(client.delete(f"/analyses/{analysis_id}").status_code, 204)


if __name__ == "__main__":
    unittest.main()
