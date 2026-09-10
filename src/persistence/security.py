"""Transactional account, quota, rate-limit, subscription, and audit persistence."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from src.persistence.database import Database


def _now() -> datetime:
    return datetime.now(timezone.utc)


class QuotaError(RuntimeError):
    """A full-analysis credit could not be reserved."""


class SecurityRepository:
    def __init__(self, database: Database, *, free_credits: int = 2) -> None:
        self.database = database
        self.free_credits = free_credits

    def sync_account(self, user_id: str, *, email: str | None, verified: bool) -> dict[str, object]:
        now = _now().isoformat()
        with self.database.connect() as connection:
            with self.database.transaction(connection, lock_keys=(f"credit:{user_id}",)):
                connection.execute(
                    """INSERT INTO accounts(user_id,email,email_verified,created_at,updated_at)
                       VALUES(?,?,?,?,?) ON CONFLICT(user_id) DO UPDATE SET
                       email=COALESCE(excluded.email,accounts.email),
                       email_verified=CASE
                           WHEN accounts.email_verified=1 OR excluded.email_verified=1 THEN 1
                           ELSE 0
                       END,
                       updated_at=excluded.updated_at""",
                    (user_id, email, int(verified), now, now),
                )
                if verified:
                    connection.execute(
                        """INSERT INTO credit_ledger
                           (entry_id,user_id,amount,kind,source,payment_reference,reason,created_at)
                           VALUES(?,?,?,'grant','free_lifetime',?,'Verified free-account allowance',?)
                           ON CONFLICT DO NOTHING""",
                        (str(uuid4()), user_id, self.free_credits, f"lifetime:{user_id}", now),
                    )
                row = connection.execute(
                    "SELECT * FROM accounts WHERE user_id=?",
                    (user_id,),
                ).fetchone()
        return dict(row)

    def account(self, user_id: str) -> dict[str, object] | None:
        with self.database.connect() as connection:
            row = connection.execute("SELECT * FROM accounts WHERE user_id=?", (user_id,)).fetchone()
        return dict(row) if row else None

    def account_by_email(self, email: str) -> dict[str, object] | None:
        with self.database.connect() as connection:
            row = connection.execute("SELECT * FROM accounts WHERE lower(email)=lower(?)", (email,)).fetchone()
        return dict(row) if row else None

    def admin_account(self) -> dict[str, object] | None:
        with self.database.connect() as connection:
            row = connection.execute("SELECT * FROM accounts WHERE role='admin' ORDER BY created_at LIMIT 1").fetchone()
        return dict(row) if row else None

    def balance(self, user_id: str) -> int:
        with self.database.connect() as connection:
            row = connection.execute(
                "SELECT COALESCE(SUM(amount),0) AS balance FROM credit_ledger WHERE user_id=?", (user_id,)
            ).fetchone()
        return int(row["balance"])

    def ledger(self, user_id: str, limit: int = 100) -> list[dict[str, object]]:
        with self.database.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM credit_ledger WHERE user_id=? ORDER BY created_at DESC LIMIT ?",
                (user_id, max(1, min(limit, 100))),
            ).fetchall()
        return [dict(row) for row in rows]

    def reserve_credit(self, user_id: str, analysis_id: str) -> str:
        reservation_id, now = str(uuid4()), _now().isoformat()
        with self.database.connect() as connection:
            with self.database.transaction(
                connection,
                lock_keys=(f"reservation:{analysis_id}",),
            ):
                existing = connection.execute(
                    "SELECT reservation_id FROM credit_reservations WHERE analysis_id=?",
                    (analysis_id,),
                ).fetchone()
                if existing:
                    return str(existing["reservation_id"])
                self.database.lock(connection, f"credit:{user_id}")
                balance = connection.execute(
                    "SELECT COALESCE(SUM(amount),0) AS value FROM credit_ledger WHERE user_id=?",
                    (user_id,),
                ).fetchone()["value"]
                if int(balance) < 1:
                    raise QuotaError("No Full Gap Analysis credits remain.")
                connection.execute(
                    "INSERT INTO credit_reservations VALUES(?,?,?,'reserved',?,?)",
                    (reservation_id, user_id, analysis_id, now, now),
                )
                connection.execute(
                    """INSERT INTO credit_ledger(entry_id,user_id,amount,kind,source,analysis_id,reason,created_at)
                       VALUES(?,?,-1,'reservation','analysis_reservation',?,'Reserved before queueing',?)""",
                    (str(uuid4()), user_id, analysis_id, now),
                )
        return reservation_id

    def settle_credit(self, analysis_id: str) -> bool:
        return self._finish_reservation(analysis_id, "settled", refund=False)

    def release_credit(self, analysis_id: str) -> bool:
        return self._finish_reservation(analysis_id, "released", refund=True)

    def _finish_reservation(self, analysis_id: str, status: str, *, refund: bool) -> bool:
        now = _now().isoformat()
        with self.database.connect() as connection:
            with self.database.transaction(
                connection,
                lock_keys=(f"reservation:{analysis_id}",),
            ):
                row = connection.execute(
                    "SELECT * FROM credit_reservations WHERE analysis_id=?",
                    (analysis_id,),
                ).fetchone()
                if not row or row["status"] != "reserved":
                    return False
                self.database.lock(connection, f"credit:{row['user_id']}")
                connection.execute(
                    "UPDATE credit_reservations SET status=?,updated_at=? WHERE analysis_id=?",
                    (status, now, analysis_id),
                )
                if refund:
                    connection.execute(
                        """INSERT INTO credit_ledger
                           (entry_id,user_id,amount,kind,source,analysis_id,reason,created_at)
                           VALUES(?,?,1,'refund','analysis_refund',?,'Analysis failed before usable result',?)
                           ON CONFLICT DO NOTHING""",
                        (str(uuid4()), row["user_id"], analysis_id, now),
                    )
                else:
                    connection.execute(
                        """INSERT INTO credit_ledger
                           (entry_id,user_id,amount,kind,source,analysis_id,reason,created_at)
                           VALUES(?,?,0,'settlement','analysis_settlement',?,'Usable result completed',?)
                           ON CONFLICT DO NOTHING""",
                        (str(uuid4()), row["user_id"], analysis_id, now),
                    )
                return True

    def adjust_credit(self, admin_id: str, user_id: str, amount: int, reason: str) -> None:
        if amount == 0 or not -1000 <= amount <= 1000 or len(reason.strip()) < 3:
            raise ValueError("A non-zero bounded amount and audit reason are required.")
        now, reference = _now().isoformat(), str(uuid4())
        with self.database.connect() as connection:
            with self.database.transaction(connection, lock_keys=(f"credit:{user_id}",)):
                connection.execute(
                    """INSERT INTO credit_ledger
                       (entry_id,user_id,amount,kind,source,payment_reference,reason,created_at)
                       VALUES(?,?,?,'admin_adjustment','admin',?,?,?)""",
                    (str(uuid4()), user_id, amount, reference, reason.strip(), now),
                )
                self._audit(connection, admin_id, "credit_adjustment", user_id, reason, {"amount": amount})

    def record_rate_event(self, principal_key: str, action: str, *, limit: int, hours: int = 24) -> bool:
        now, cutoff = _now(), _now() - timedelta(hours=hours)
        with self.database.connect() as connection:
            with self.database.transaction(
                connection,
                lock_keys=(f"rate:{principal_key}:{action}",),
            ):
                count = connection.execute(
                    "SELECT COUNT(*) AS count FROM rate_events "
                    "WHERE principal_key=? AND action=? AND created_at>=?",
                    (principal_key, action, cutoff.isoformat()),
                ).fetchone()["count"]
                if int(count) >= limit:
                    return False
                connection.execute(
                    "INSERT INTO rate_events VALUES(?,?,?,?)",
                    (str(uuid4()), principal_key, action, now.isoformat()),
                )
                return True

    def touch_guest(self, guest_id: str, network_key: str, *, retention_hours: int) -> None:
        now, expiry = _now(), _now() + timedelta(hours=retention_hours)
        with self.database.connect() as connection:
            connection.execute(
                """INSERT INTO guest_sessions VALUES(?,?,?,?,?,NULL)
                   ON CONFLICT(guest_id) DO UPDATE SET last_seen_at=excluded.last_seen_at""",
                (guest_id, network_key, now.isoformat(), now.isoformat(), expiry.isoformat()),
            )
            connection.commit()

    def guest_active(self, guest_id: str) -> bool:
        with self.database.connect() as connection:
            row = connection.execute(
                "SELECT 1 FROM guest_sessions WHERE guest_id=? AND expires_at>? AND claimed_by_user_id IS NULL",
                (guest_id, _now().isoformat()),
            ).fetchone()
        return row is not None

    def profile_update(self, user_id: str, *, display_name: str | None = None, avatar_url: str | None = None) -> None:
        with self.database.connect() as connection:
            if display_name is not None:
                connection.execute(
                    "UPDATE accounts SET display_name=?,updated_at=? WHERE user_id=?",
                    (display_name.strip()[:80], _now().isoformat(), user_id),
                )
            if avatar_url is not None:
                connection.execute(
                    "UPDATE accounts SET avatar_url=?,updated_at=? WHERE user_id=?",
                    (avatar_url, _now().isoformat(), user_id),
                )
            connection.commit()

    def delete_account_data(self, user_id: str) -> None:
        """Delete research content and anonymize the retained usage/payment audit subject."""
        with self.database.connect() as connection:
            with self.database.transaction(connection, lock_keys=(f"account:{user_id}",)):
                connection.execute(
                    "DELETE FROM analyses WHERE owner_kind='user' AND owner_id=?",
                    (user_id,),
                )
                connection.execute(
                    "UPDATE accounts SET email=NULL,display_name='',avatar_url=NULL,"
                    "status='deleted',updated_at=? WHERE user_id=?",
                    (_now().isoformat(), user_id),
                )

    def set_role(self, user_id: str, role: str) -> None:
        if role not in {"user", "admin"}:
            raise ValueError("invalid role")
        with self.database.connect() as connection:
            connection.execute("UPDATE accounts SET role=?,updated_at=? WHERE user_id=?", (role, _now().isoformat(), user_id))
            connection.commit()

    def set_status(self, admin_id: str, user_id: str, status: str, reason: str) -> None:
        if status not in {"active", "suspended"} or len(reason.strip()) < 3:
            raise ValueError("Valid status and reason are required.")
        with self.database.connect() as connection:
            with self.database.transaction(connection, lock_keys=(f"account:{user_id}",)):
                connection.execute(
                    "UPDATE accounts SET status=?,updated_at=? WHERE user_id=?",
                    (status, _now().isoformat(), user_id),
                )
                self._audit(connection, admin_id, f"account_{status}", user_id, reason, {})

    def list_users(self, limit: int = 100) -> list[dict[str, object]]:
        with self.database.connect() as connection:
            rows = connection.execute(
                """SELECT a.user_id,a.email,a.display_name,a.role,a.status,a.created_at,
                   COALESCE(s.status,'none') subscription_status,
                   COALESCE((SELECT SUM(amount) FROM credit_ledger l WHERE l.user_id=a.user_id),0) balance
                   FROM accounts a LEFT JOIN subscriptions s ON s.user_id=a.user_id
                   ORDER BY a.created_at DESC LIMIT ?""", (max(1, min(limit, 200)),)
            ).fetchall()
        return [dict(row) for row in rows]

    def admin_summary(self) -> dict[str, object]:
        with self.database.connect() as connection:
            users = connection.execute("SELECT COUNT(*) c FROM accounts WHERE status!='deleted'").fetchone()["c"]
            analyses = [dict(row) for row in connection.execute(
                "SELECT status,mode,COUNT(*) count FROM analyses WHERE owner_id IS NOT NULL GROUP BY status,mode"
            ).fetchall()]
            owner_usage = [dict(row) for row in connection.execute(
                "SELECT owner_kind,mode,COUNT(*) count FROM analyses WHERE owner_id IS NOT NULL GROUP BY owner_kind,mode"
            ).fetchall()]
            result_rows = connection.execute(
                "SELECT result_json FROM analyses WHERE status='completed' AND result_json IS NOT NULL"
            ).fetchall()
            failed_events = [dict(row) for row in connection.execute(
                "SELECT event_id,event_type,error_message,processed_at FROM payment_events WHERE status='failed' ORDER BY processed_at DESC LIMIT 20"
            ).fetchall()]
            audits = [dict(row) for row in connection.execute(
                "SELECT * FROM admin_audit_log ORDER BY created_at DESC LIMIT 50"
            ).fetchall()]
        work: dict[str, float] = {}
        timings: dict[str, float] = {}
        for row in result_rows:
            try:
                result = json.loads(row["result_json"])
            except (TypeError, ValueError):
                continue
            for key, value in (result.get("work_metrics") or {}).items():
                if isinstance(value, (int, float)):
                    work[key] = work.get(key, 0.0) + float(value)
            for key, value in (result.get("stage_timings") or {}).items():
                if isinstance(value, (int, float)):
                    timings[key] = timings.get(key, 0.0) + float(value)
        return {"user_count": users, "analyses": analyses, "usage_by_principal": owner_usage,
                "aggregate_work_metrics": work, "aggregate_stage_seconds": timings,
                "provider_cost_usd": None, "provider_cost_note": "Unavailable unless provider usage metadata and pricing are recorded.",
                "failed_payment_events": failed_events, "audit_log": audits}

    def process_payment_event(self, event: dict[str, object], *, cycle_credits: int,
                              expected_price_id: str | None = None) -> str:
        """Idempotently synchronize Stripe state and grant only invoice-paid cycles."""
        event_id, event_type = str(event.get("id", "")), str(event.get("type", ""))
        created = int(event.get("created", 0) or 0)
        obj = ((event.get("data") or {}).get("object") or {}) if isinstance(event.get("data"), dict) else {}
        now = _now().isoformat()
        if not event_id or not isinstance(obj, dict):
            raise ValueError("Malformed payment event.")
        with self.database.connect() as connection:
            self.database.begin(connection, lock_keys=(f"payment:{event_id}",))
            existing_event = connection.execute("SELECT status FROM payment_events WHERE event_id=?", (event_id,)).fetchone()
            if existing_event and existing_event["status"] == "processed":
                connection.commit()
                return "duplicate"
            customer = str(obj.get("customer") or "")
            account = connection.execute("SELECT user_id FROM accounts WHERE stripe_customer_id=?", (customer,)).fetchone()
            hinted_user = str(obj.get("client_reference_id") or "")
            metadata = obj.get("metadata") if isinstance(obj.get("metadata"), dict) else {}
            hinted_user = hinted_user or str(metadata.get("research_gap_user_id") or "")
            parent = obj.get("parent") if isinstance(obj.get("parent"), dict) else {}
            subscription_details = parent.get("subscription_details") if isinstance(parent.get("subscription_details"), dict) else {}
            if not subscription_details and isinstance(obj.get("subscription_details"), dict):
                subscription_details = obj["subscription_details"]
            subscription_metadata = subscription_details.get("metadata") if isinstance(subscription_details.get("metadata"), dict) else {}
            hinted_user = hinted_user or str(subscription_metadata.get("research_gap_user_id") or "")
            if not account and hinted_user:
                account = connection.execute("SELECT user_id FROM accounts WHERE user_id=?", (hinted_user,)).fetchone()
                if account and customer:
                    connection.execute(
                        "UPDATE accounts SET stripe_customer_id=?,updated_at=? WHERE user_id=? AND stripe_customer_id IS NULL",
                        (customer, now, account["user_id"]),
                    )
            if account:
                self.database.lock(connection, f"credit:{account['user_id']}")
            status = "ignored"
            if account and event_type == "invoice.paid" and self._invoice_has_price(obj, expected_price_id):
                connection.execute(
                    """INSERT INTO credit_ledger
                       (entry_id,user_id,amount,kind,source,payment_reference,reason,created_at)
                       VALUES(?,?,?,'grant','stripe_invoice',?,'Successful billing cycle',?)
                       ON CONFLICT DO NOTHING""",
                    (str(uuid4()), account["user_id"], cycle_credits, str(obj.get("id")), now),
                )
                status = "processed"
            elif account and event_type.startswith("customer.subscription."):
                subscription_id = str(obj.get("id") or "")
                incoming_status = str(obj.get("status") or "unknown")
                period_end = obj.get("current_period_end")
                period_iso = datetime.fromtimestamp(int(period_end), timezone.utc).isoformat() if period_end else None
                connection.execute(
                    """INSERT INTO subscriptions(user_id,stripe_customer_id,stripe_subscription_id,status,current_period_end,last_event_created,updated_at)
                       VALUES(?,?,?,?,?,?,?) ON CONFLICT(user_id) DO UPDATE SET
                       stripe_customer_id=excluded.stripe_customer_id,
                       stripe_subscription_id=excluded.stripe_subscription_id,
                       status=CASE WHEN excluded.last_event_created>=subscriptions.last_event_created THEN excluded.status ELSE subscriptions.status END,
                       current_period_end=CASE WHEN excluded.last_event_created>=subscriptions.last_event_created THEN excluded.current_period_end ELSE subscriptions.current_period_end END,
                       last_event_created=CASE
                           WHEN excluded.last_event_created>=subscriptions.last_event_created
                           THEN excluded.last_event_created ELSE subscriptions.last_event_created
                       END,updated_at=excluded.updated_at""",
                    (account["user_id"], customer, subscription_id, incoming_status, period_iso, created, now),
                )
                status = "processed"
            elif account and event_type == "invoice.payment_failed":
                subscription_value = obj.get("subscription")
                if isinstance(subscription_value, dict):
                    subscription_value = subscription_value.get("id")
                subscription_value = subscription_value or subscription_details.get("subscription")
                connection.execute(
                    """INSERT INTO subscriptions
                       (user_id,stripe_customer_id,stripe_subscription_id,status,current_period_end,last_event_created,updated_at)
                       VALUES(?,?,?,'past_due',NULL,?,?) ON CONFLICT(user_id) DO UPDATE SET
                       status=CASE WHEN excluded.last_event_created>=subscriptions.last_event_created THEN 'past_due' ELSE subscriptions.status END,
                       last_event_created=CASE
                           WHEN excluded.last_event_created>=subscriptions.last_event_created
                           THEN excluded.last_event_created ELSE subscriptions.last_event_created
                       END,updated_at=excluded.updated_at""",
                    (account["user_id"], customer, str(subscription_value or "invoice:" + str(obj.get("id"))), created, now),
                )
                status = "processed"
            elif account and event_type == "checkout.session.completed":
                # Browser return and Checkout completion synchronize identity only. Credits
                # remain gated on the separately signed invoice.paid event.
                status = "processed"
            connection.execute(
                """INSERT INTO payment_events VALUES(?,?,?,?,NULL,?)
                   ON CONFLICT(event_id) DO UPDATE SET status=excluded.status,processed_at=excluded.processed_at""",
                (event_id, event_type, created, status, now),
            )
            connection.commit()
            return status

    def record_failed_payment_event(self, event: dict[str, object], error: str) -> None:
        event_id = str(event.get("id") or "")
        if not event_id:
            return
        with self.database.connect() as connection:
            with self.database.transaction(connection, lock_keys=(f"payment:{event_id}",)):
                connection.execute(
                    """INSERT INTO payment_events
                       (event_id,event_type,created_epoch,status,error_message,processed_at)
                       VALUES(?,?,?,'failed',?,?) ON CONFLICT(event_id) DO UPDATE SET
                       status='failed',error_message=excluded.error_message,
                       processed_at=excluded.processed_at
                       WHERE payment_events.status!='processed'""",
                    (
                        event_id,
                        str(event.get("type") or "unknown"),
                        int(event.get("created", 0) or 0),
                        " ".join(error.split())[:500],
                        _now().isoformat(),
                    ),
                )

    @staticmethod
    def _invoice_has_price(invoice: dict[str, object], expected: str | None) -> bool:
        if not expected:
            return True
        lines = invoice.get("lines") if isinstance(invoice.get("lines"), dict) else {}
        data = lines.get("data") if isinstance(lines.get("data"), list) else []
        for line in data:
            if not isinstance(line, dict):
                continue
            price = line.get("price") if isinstance(line.get("price"), dict) else {}
            if price.get("id") == expected:
                return True
            pricing = line.get("pricing") if isinstance(line.get("pricing"), dict) else {}
            details = pricing.get("price_details") if isinstance(pricing.get("price_details"), dict) else {}
            if details.get("price") == expected:
                return True
        return False

    def link_stripe_customer(self, user_id: str, customer_id: str) -> None:
        with self.database.connect() as connection:
            with self.database.transaction(connection, lock_keys=(f"account:{user_id}",)):
                connection.execute(
                    "UPDATE accounts SET stripe_customer_id=?,updated_at=? WHERE user_id=?",
                    (customer_id, _now().isoformat(), user_id),
                )

    @staticmethod
    def _audit(connection, admin_id: str, action: str, target_id: str | None, reason: str,
               metadata: dict[str, object]) -> None:
        connection.execute(
            "INSERT INTO admin_audit_log VALUES(?,?,?,?,?,?,?)",
            (str(uuid4()), admin_id, action, target_id, reason.strip(), json.dumps(metadata), _now().isoformat()),
        )
