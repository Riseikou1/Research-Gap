from __future__ import annotations

from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor

import pytest

from src.operations.repository import OperationsRepository, ProviderBudgetExceeded
from src.operations.usage import ProviderUsage, usage_from_response
from src.persistence.database import Database
from src.persistence.models import NewAnalysis
from src.persistence.repositories import AnalysisRepository
from src.persistence.security import SecurityRepository


def test_budget_reservation_is_persistent_conservative_and_releasable(tmp_path) -> None:
    database = Database(tmp_path / "operations.sqlite3")
    database.migrate()
    operations = OperationsRepository(database, daily_budget_usd=0.50, reservation_usd=0.25)

    assert operations.reserve_budget("a1")
    assert operations.reserve_budget("a1")  # idempotent
    assert operations.reserve_budget("a2")
    with pytest.raises(ProviderBudgetExceeded):
        operations.reserve_budget("a3")
    assert operations.release_budget("a1")
    assert operations.reserve_budget("a3")


def test_concurrent_budget_reservations_cannot_oversubscribe(tmp_path) -> None:
    database = Database(tmp_path / "concurrent-budget.sqlite3")
    database.migrate()
    operations = OperationsRepository(database, daily_budget_usd=0.25, reservation_usd=0.25)

    def attempt(identifier: str) -> bool:
        try:
            operations.reserve_budget(identifier)
            return True
        except ProviderBudgetExceeded:
            return False

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(attempt, ["concurrent-1", "concurrent-2"]))
    assert sorted(outcomes) == [False, True]


def test_usage_records_unknown_as_unavailable_and_prices_known_tokens(tmp_path) -> None:
    database = Database(tmp_path / "usage.sqlite3")
    database.migrate()
    AnalysisRepository(database).create(NewAnalysis(
        analysis_id="analysis-1", research_idea="bounded citation graph",
        decomposer="deterministic", query_generator="deterministic", paper_limit=5,
        configuration={}, owner_kind="local", owner_id="local", mode="full",
    ))
    operations = OperationsRepository(
        database, input_per_million_usd=1.0, output_per_million_usd=2.0,
        reservation_usd=0.25,
    )
    unknown = ProviderUsage(
        provider="openai", model="fake", stage="test", timestamp=datetime.now(timezone.utc),
    )
    known = ProviderUsage(
        provider="openai", model="fake", stage="test", input_tokens=1000,
        output_tokens=500, total_tokens=1500, timestamp=datetime.now(timezone.utc),
    )
    assert operations.record_usage("analysis-1", [known]) == pytest.approx(0.002)
    assert operations.record_usage("analysis-1", [unknown]) == 0.25
    diagnostics = operations.diagnostics()["provider_usage_today"]
    assert diagnostics["request_count"] == 2
    assert diagnostics["unavailable_count"] == 1


def test_usage_adapter_does_not_turn_missing_metadata_into_zero() -> None:
    record = usage_from_response(object(), provider="openai", model="fake", stage="test")
    assert record.input_tokens is None
    assert record.output_tokens is None
    assert record.total_tokens is None
    response = type("Response", (), {"usage": {"input_tokens": 0, "output_tokens": 0}})()
    zero = usage_from_response(response, provider="openai", model="fake", stage="test")
    assert zero.input_tokens == 0 and zero.output_tokens == 0 and zero.total_tokens == 0


def test_rate_limit_is_shared_across_repository_instances(tmp_path) -> None:
    database = Database(tmp_path / "rate.sqlite3")
    database.migrate()
    first = SecurityRepository(database, lifetime_credit_hmac_secret="test-secret")
    second = SecurityRepository(database, lifetime_credit_hmac_secret="test-secret")
    assert first.record_rate_event("user:u1", "full_analysis", limit=1)
    assert not second.record_rate_event("user:u1", "full_analysis", limit=1)
