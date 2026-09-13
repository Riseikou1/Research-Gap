"""Concurrency-safe operational accounting backed by the application database."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from src.operations.usage import ProviderUsage
from src.persistence.database import Database


class ProviderBudgetExceeded(RuntimeError):
    """Raised before queueing when the configured daily provider budget is full."""


class OperationsRepository:
    def __init__(self, database: Database, *, daily_budget_usd: float = 0.0,
                 reservation_usd: float = 0.25, input_per_million_usd: float = 0.0,
                 output_per_million_usd: float = 0.0) -> None:
        self.database = database
        self.daily_budget_usd = daily_budget_usd
        self.reservation_usd = reservation_usd
        self.input_per_million_usd = input_per_million_usd
        self.output_per_million_usd = output_per_million_usd

    @property
    def budget_enabled(self) -> bool:
        return self.daily_budget_usd > 0

    def reserve_budget(self, analysis_id: str) -> str | None:
        if not self.budget_enabled:
            return None
        day = datetime.now(timezone.utc).date().isoformat()
        reservation_id = str(uuid4())
        with self.database.connect() as connection:
            with self.database.transaction(connection, lock_keys=(f"provider-budget:{day}",)):
                existing = connection.execute(
                    "SELECT reservation_id FROM provider_budget_reservations WHERE analysis_id=?",
                    (analysis_id,),
                ).fetchone()
                if existing:
                    return str(existing["reservation_id"])
                row = connection.execute(
                    """SELECT COALESCE(SUM(CASE WHEN status='settled' THEN actual_cost_usd
                           WHEN status='reserved' THEN reserved_cost_usd ELSE 0 END),0) AS committed
                       FROM provider_budget_reservations WHERE budget_day=?""",
                    (day,),
                ).fetchone()
                if float(row["committed"] or 0) + self.reservation_usd > self.daily_budget_usd:
                    raise ProviderBudgetExceeded("The daily provider spending budget is currently full.")
                connection.execute(
                    """INSERT INTO provider_budget_reservations
                       (reservation_id,analysis_id,budget_day,reserved_cost_usd,actual_cost_usd,status,created_at,updated_at)
                       VALUES(?,?,?,?,NULL,'reserved',?,?)""",
                    (reservation_id, analysis_id, day, self.reservation_usd,
                     datetime.now(timezone.utc).isoformat(), datetime.now(timezone.utc).isoformat()),
                )
        return reservation_id

    def record_usage(self, analysis_id: str, records: list[ProviderUsage]) -> float:
        total = 0.0
        fully_priced = bool(records)
        with self.database.connect() as connection:
            with self.database.transaction(connection, lock_keys=(f"provider-usage:{analysis_id}",)):
                for record in records:
                    cost = record.cost_usd
                    if cost is None and record.input_tokens is not None and record.output_tokens is not None:
                        cost = (
                            record.input_tokens * self.input_per_million_usd
                            + record.output_tokens * self.output_per_million_usd
                        ) / 1_000_000
                    if cost is not None:
                        total += cost
                    else:
                        fully_priced = False
                    connection.execute(
                        """INSERT INTO provider_usage
                           (usage_id,analysis_id,provider,model,stage,input_tokens,output_tokens,total_tokens,cost_usd,occurred_at,cache_hit)
                           VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                        (str(uuid4()), analysis_id, record.provider, record.model, record.stage,
                         record.input_tokens, record.output_tokens, record.total_tokens, cost,
                         record.timestamp.isoformat(), int(record.cache_hit)),
                    )
        return total if fully_priced else self.reservation_usd

    def settle_budget(self, analysis_id: str, actual_cost_usd: float) -> bool:
        return self._finish_budget(analysis_id, "settled", actual_cost_usd)

    def release_budget(self, analysis_id: str) -> bool:
        return self._finish_budget(analysis_id, "released", None)

    def _finish_budget(self, analysis_id: str, status: str, actual: float | None) -> bool:
        with self.database.connect() as connection:
            with self.database.transaction(connection, lock_keys=(f"provider-budget-analysis:{analysis_id}",)):
                cursor = connection.execute(
                    """UPDATE provider_budget_reservations SET status=?,actual_cost_usd=?,updated_at=?
                       WHERE analysis_id=? AND status='reserved'""",
                    (status, actual, datetime.now(timezone.utc).isoformat(), analysis_id),
                )
                return cursor.rowcount > 0

    def record_failure(self, *, request_id: str | None, analysis_id: str | None,
                       stage: str, category: str) -> None:
        with self.database.connect() as connection:
            with self.database.transaction(connection):
                connection.execute(
                    "INSERT INTO operational_failures VALUES(?,?,?,?,?,?)",
                    (str(uuid4()), request_id, analysis_id, stage[:80], category[:80],
                     datetime.now(timezone.utc).isoformat()),
                )

    def diagnostics(self) -> dict[str, object]:
        today = datetime.now(timezone.utc).date().isoformat()
        with self.database.connect() as connection:
            usage = connection.execute(
                """SELECT COUNT(*) AS request_count,SUM(input_tokens) AS input_tokens,
                          SUM(output_tokens) AS output_tokens,
                          SUM(total_tokens) AS total_tokens,SUM(cost_usd) AS cost_usd,
                          COUNT(input_tokens) AS known_input_count,
                          COUNT(output_tokens) AS known_output_count,
                          COUNT(total_tokens) AS known_total_count,
                          COUNT(cost_usd) AS priced_count,
                          SUM(CASE WHEN input_tokens IS NULL OR output_tokens IS NULL THEN 1 ELSE 0 END) AS unavailable_count
                   FROM provider_usage WHERE substr(occurred_at,1,10)=?""", (today,),
            ).fetchone()
            budget = connection.execute(
                """SELECT COALESCE(SUM(CASE WHEN status='settled' THEN actual_cost_usd
                       WHEN status='reserved' THEN reserved_cost_usd ELSE 0 END),0) AS committed
                   FROM provider_budget_reservations WHERE budget_day=?""", (today,),
            ).fetchone()
            failures = [dict(row) for row in connection.execute(
                """SELECT request_id,analysis_id,stage,category,created_at FROM operational_failures
                   ORDER BY created_at DESC LIMIT 20"""
            ).fetchall()]
            failure_counts = [dict(row) for row in connection.execute(
                """SELECT category,COUNT(*) AS count FROM operational_failures
                   WHERE substr(created_at,1,10)=? GROUP BY category ORDER BY category""",
                (today,),
            ).fetchall()]
        usage_payload = dict(usage)
        for field, count_field in (
            ("input_tokens", "known_input_count"), ("output_tokens", "known_output_count"),
            ("total_tokens", "known_total_count"), ("cost_usd", "priced_count"),
        ):
            if not usage_payload.get(count_field):
                usage_payload[field] = None
        committed = float(budget["committed"] or 0)
        return {
            "provider_usage_today": usage_payload,
            "provider_budget": {
                "enabled": self.budget_enabled, "daily_limit_usd": self.daily_budget_usd or None,
                "committed_usd": committed,
                "remaining_usd": max(0.0, self.daily_budget_usd - committed) if self.budget_enabled else None,
                "reservation_usd": self.reservation_usd if self.budget_enabled else None,
            },
            "recent_failures": failures,
            "failure_counts_today": failure_counts,
        }
