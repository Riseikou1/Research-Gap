"""Opt-in PostgreSQL integration checks using an isolated temporary schema."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from uuid import uuid4

import psycopg
import pytest
from psycopg import sql
from psycopg.conninfo import make_conninfo

from src.persistence.database import Database
from src.persistence.models import NewAnalysis
from src.persistence.repositories import AnalysisRepository
from src.persistence.security import QuotaError, SecurityRepository


POSTGRES_TEST_URL = os.getenv("RESEARCH_GAP_TEST_DATABASE_URL", "").strip()
pytestmark = pytest.mark.skipif(
    not POSTGRES_TEST_URL,
    reason="set RESEARCH_GAP_TEST_DATABASE_URL to run PostgreSQL integration tests",
)


def test_postgres_migrations_crud_and_credit_concurrency() -> None:
    schema = "research_gap_test_" + uuid4().hex
    with psycopg.connect(POSTGRES_TEST_URL, autocommit=True) as admin:
        admin.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))

    database_url = make_conninfo(
        POSTGRES_TEST_URL,
        options=f"-c search_path={schema}",
    )
    database = Database(Path("unused.sqlite"), url=database_url)
    try:
        assert database.migrate() == [
            "0001_create_analyses",
            "0002_web_accounts_billing",
            "0003_unique_lifetime_credit",
        ]
        assert database.migrate() == []

        repository = AnalysisRepository(database)
        record = repository.create(
            NewAnalysis(
                analysis_id="postgres-analysis",
                research_idea="PostgreSQL persistence integration",
                decomposer="deterministic",
                query_generator="deterministic",
                paper_limit=10,
                configuration={"backend": "postgres"},
                owner_kind="user",
                owner_id="postgres-user",
            )
        )
        assert repository.get(record.analysis_id) == record
        assert repository.delete(record.analysis_id) is True

        security = SecurityRepository(database)

        def sync_account(_index: int):
            return security.sync_account(
                "postgres-user",
                email="postgres@example.test",
                verified=True,
            )

        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(sync_account, range(16)))
        assert security.balance("postgres-user") == 2
        assert len(
            [row for row in security.ledger("postgres-user") if row["source"] == "free_lifetime"]
        ) == 1

        def reserve(index: int):
            try:
                return security.reserve_credit("postgres-user", f"postgres-credit-{index}")
            except QuotaError:
                return None

        with ThreadPoolExecutor(max_workers=3) as pool:
            reservations = list(pool.map(reserve, range(3)))
        assert sum(value is not None for value in reservations) == 2
        assert security.balance("postgres-user") == 0
        successful = [index for index, value in enumerate(reservations) if value is not None]
        settled_id = f"postgres-credit-{successful[0]}"
        released_id = f"postgres-credit-{successful[1]}"
        assert security.settle_credit(settled_id) is True
        assert security.settle_credit(settled_id) is False
        assert security.release_credit(released_id) is True
        assert security.release_credit(released_id) is False
        assert security.balance("postgres-user") == 1

        security.touch_guest("postgres-guest", "network-key", retention_hours=24)
        assert security.guest_active("postgres-guest") is True
        repository.create(
            NewAnalysis(
                analysis_id="guest-analysis",
                research_idea="PostgreSQL guest claim",
                decomposer="deterministic",
                query_generator="deterministic",
                paper_limit=10,
                configuration={"backend": "postgres"},
                owner_kind="guest",
                owner_id="postgres-guest",
                mode="quick",
            )
        )
        assert repository.claim_guest("postgres-guest", "postgres-user") == 1
        assert repository.claim_guest("postgres-guest", "postgres-user") == 0
        assert security.record_rate_event("postgres-user", "quick", limit=1) is True
        assert security.record_rate_event("postgres-user", "quick", limit=1) is False

        security.link_stripe_customer("postgres-user", "postgres-customer")
        invoice = {
            "id": "postgres-invoice-event",
            "type": "invoice.paid",
            "created": 1,
            "data": {
                "object": {
                    "id": "postgres-invoice",
                    "customer": "postgres-customer",
                }
            },
        }
        assert security.process_payment_event(invoice, cycle_credits=5) == "processed"
        assert security.process_payment_event(invoice, cycle_credits=5) == "duplicate"
        subscription = {
            "id": "postgres-subscription-event",
            "type": "customer.subscription.updated",
            "created": 2,
            "data": {
                "object": {
                    "id": "postgres-subscription",
                    "customer": "postgres-customer",
                    "status": "active",
                }
            },
        }
        assert security.process_payment_event(subscription, cycle_credits=5) == "processed"
        assert security.balance("postgres-user") == 6
        assert repository.cleanup_expired_guests() == 0
    finally:
        with psycopg.connect(POSTGRES_TEST_URL, autocommit=True) as admin:
            admin.execute(
                sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema))
            )
