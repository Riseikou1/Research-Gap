"""Bounded local execution for long-running analyses."""

from __future__ import annotations

import logging
import re
from concurrent.futures import Future, ThreadPoolExecutor
from threading import Lock
from typing import Callable

from src.persistence.models import AnalysisRecord
from src.persistence.repositories import AnalysisRepository

LOGGER = logging.getLogger(__name__)
AnalysisExecutor = Callable[[AnalysisRecord], dict[str, object]]


def safe_error_message(error: Exception) -> str:
    """Return useful failure text without persisting credentials or a traceback."""
    message = " ".join(str(error).split())
    message = re.sub(r"\bsk-[A-Za-z0-9_-]{8,}\b", "[redacted]", message)
    message = re.sub(
        r"(?i)\b(api[-_ ]?key|authorization|bearer)\s*[:=]\s*[^\s,;]+",
        r"\1=[redacted]",
        message,
    )
    if not message:
        message = "Analysis failed without an error message."
    return f"{type(error).__name__}: {message}"[:1000]


class AnalysisJobRunner:
    """Submit each analysis once and execute at most ``max_workers`` concurrently."""

    def __init__(self, repository: AnalysisRepository, executor: AnalysisExecutor, *, max_workers: int) -> None:
        if max_workers < 1:
            raise ValueError("max_workers must be positive")
        self.repository = repository
        self.analysis_executor = executor
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="research-gap-analysis")
        self._futures: dict[str, Future[None]] = {}
        self._lock = Lock()
        self._closed = False
        self.on_success: Callable[[str], object] | None = None
        self.on_failure: Callable[[str], object] | None = None

    def submit(self, analysis_id: str) -> bool:
        with self._lock:
            if self._closed:
                raise RuntimeError("analysis job runner is shut down")
            current = self._futures.get(analysis_id)
            if current is not None and not current.done():
                return False
            future = self._pool.submit(self._execute, analysis_id)
            self._futures[analysis_id] = future
        # Register outside the lock: add_done_callback runs immediately when
        # very fast work has already completed.
        future.add_done_callback(lambda completed, job_id=analysis_id: self._forget(job_id, completed))
        return True

    def recover(self) -> None:
        """Fail work interrupted by a crash and resume jobs that never started."""
        self.repository.fail_interrupted()
        for record in self.repository.list_pending():
            self.submit(record.analysis_id)

    def cancel(self, analysis_id: str) -> bool:
        """Cancel queued work; running pipeline calls cannot be interrupted safely."""
        with self._lock:
            future = self._futures.get(analysis_id)
            if future is None or future.done():
                return True
        # Future.cancel() can run callbacks synchronously, so do not hold the
        # registry lock while calling it.
        return future.cancel()

    def shutdown(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._pool.shutdown(wait=True, cancel_futures=False)

    def _execute(self, analysis_id: str) -> None:
        try:
            if not self.repository.mark_running(analysis_id):
                return
            record = self.repository.get(analysis_id)
            if record is None:
                return
            result = self.analysis_executor(record)
            if not isinstance(result, dict):
                raise TypeError("analysis executor must return a dictionary")
            if self.on_success:
                self.on_success(analysis_id)
            self.repository.mark_completed(analysis_id, result)
        except Exception as exc:
            message = safe_error_message(exc)
            LOGGER.exception("analysis failed id=%s error=%s", analysis_id, message)
            try:
                if self.on_failure:
                    self.on_failure(analysis_id)
                self.repository.mark_failed(analysis_id, message)
            except Exception as persistence_error:
                LOGGER.error(
                    "could not persist analysis failure id=%s error_type=%s",
                    analysis_id,
                    type(persistence_error).__name__,
                )

    def _forget(self, analysis_id: str, future: Future[None]) -> None:
        with self._lock:
            if self._futures.get(analysis_id) is future:
                self._futures.pop(analysis_id, None)
