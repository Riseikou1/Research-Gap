"""Deterministic Milestone 9 ownership, quota, auth, billing, and avatar checks."""

from __future__ import annotations

import json
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from threading import Event
from unittest.mock import patch

from fastapi.testclient import TestClient

from src.api.app import create_app
from src.auth import AuthenticationError, AuthIdentity, StaticAuthProvider, SupabaseAuthProvider
from src.billing import BillingError
from src.config import Settings
from src.persistence.database import Database
from src.persistence.security import QuotaError, SecurityRepository
from src.persistence.models import NewAnalysis
from src.storage import AvatarError, validate_avatar
from src.application.analysis_service import PipelineOptions, build_pipeline
from src.api.routes.analyses import _markdown_report
from src.api.safety import public_analysis_result


def result_for(record):
    if "fails" in record.research_idea:
        raise RuntimeError("deterministic failure")
    return {"mode": record.mode, "papers": [{"id": "W1", "title": "Paper"}], "gaps": []}


class FakeBilling:
    def create_checkout(self, **_kwargs): return {"id": "cs_test", "url": "https://checkout.stripe.test/session"}
    def create_portal(self, **_kwargs): return {"id": "bps_test", "url": "https://billing.stripe.test/session"}
    def verify_webhook(self, payload: bytes, signature: str):
        if signature != "valid": raise BillingError("Invalid payment webhook signature.")
        return json.loads(payload)


def settings_for(path: Path) -> Settings:
    with patch.dict(os.environ, {}, clear=True):
        settings = Settings.from_env()
    return replace(settings, analysis_database_path=path, max_analysis_workers=2,
                   web=replace(settings.web, stripe_price_id="price_test_placeholder",
                               stripe_secret_key="sk_test_fake", stripe_webhook_secret="whsec_fake"))


def wait(client: TestClient, analysis_id: str, token: str | None = None):
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    for _ in range(200):
        value = client.get(f"/analyses/{analysis_id}", headers=headers).json()
        if value.get("status") in {"completed", "failed"}: return value
        time.sleep(.01)
    raise AssertionError("job did not finish")


def test_verified_user_gets_exactly_two_and_third_is_blocked_and_failure_refunds():
    with tempfile.TemporaryDirectory() as directory:
        auth = StaticAuthProvider({"verified": AuthIdentity("u1", "u@example.test", True)})
        with TestClient(create_app(settings=settings_for(Path(directory)/"db.sqlite"), analysis_executor=result_for, auth_provider=auth)) as client:
            assert client.get("/me", headers={"Authorization": "Bearer verified"}).json()["credits"] == 2
            first = client.post("/analyses", headers={"Authorization": "Bearer verified"}, json={"research_idea":"first valid idea","mode":"full"})
            second = client.post("/analyses", headers={"Authorization": "Bearer verified"}, json={"research_idea":"second valid idea","mode":"full"})
            assert first.status_code == second.status_code == 201
            first_record = client.app.state.components.repository.get(first.json()["analysis_id"])
            assert first_record.reservation_id is not None
            assert first_record.configuration["credit_billing_decision"] == "paid_credit"
            wait(client, first.json()["analysis_id"], "verified")
            wait(client, second.json()["analysis_id"], "verified")
            blocked = client.post("/analyses", headers={"Authorization": "Bearer verified"}, json={"research_idea":"third valid idea","mode":"full"})
            assert blocked.status_code == 402
            assert client.get("/me", headers={"Authorization": "Bearer verified"}).json()["credits"] == 0
            client.app.state.components.security.adjust_credit("admin", "u1", 1, "test failure refund")
            failed = client.post("/analyses", headers={"Authorization": "Bearer verified"}, json={"research_idea":"this fails safely","mode":"full"})
            failed_record = wait(client, failed.json()["analysis_id"], "verified")
            assert failed_record["status"] == "failed"
            assert "reserved credit was returned" in failed_record["error_message"]
            assert client.get("/me", headers={"Authorization": "Bearer verified"}).json()["credits"] == 1


def test_zero_credit_admin_full_analysis_is_exempt_owned_and_persisted_without_ledger_work():
    auth = StaticAuthProvider({"admin-token": AuthIdentity("admin-user", "admin@example.test", True)})
    with tempfile.TemporaryDirectory() as directory:
        with TestClient(create_app(
            settings=settings_for(Path(directory) / "admin-exempt.sqlite"),
            analysis_executor=result_for,
            auth_provider=auth,
        )) as client:
            headers = {"Authorization": "Bearer admin-token"}
            security = client.app.state.components.security
            client.get("/me", headers=headers)
            security.adjust_credit("setup", "admin-user", -2, "zero admin test balance")
            security.set_role("admin-user", "admin")
            before_ledger = security.ledger("admin-user")

            account = client.get("/me", headers=headers).json()
            assert account["role"] == "admin"
            assert account["credit_exempt"] is True
            assert account["credits"] == 0
            assert account["plan_label"] == "Admin access"

            created = client.post("/analyses", headers=headers, json={
                "research_idea": "administrator full analysis with zero credits",
                "mode": "full",
            })
            assert created.status_code == 201
            analysis_id = created.json()["analysis_id"]
            persisted = client.app.state.components.repository.get(analysis_id)
            assert persisted.owner_kind == "user"
            assert persisted.owner_id == "admin-user"
            assert persisted.reservation_id is None
            assert persisted.configuration["credit_billing_decision"] == "administrator_credit_exempt"

            completed = wait(client, analysis_id, "admin-token")
            assert completed["status"] == "completed"
            assert security.balance("admin-user") == 0
            assert security.ledger("admin-user") == before_ledger
            with client.app.state.components.database.connect() as connection:
                reservation = connection.execute(
                    "SELECT 1 FROM credit_reservations WHERE analysis_id=?",
                    (analysis_id,),
                ).fetchone()
                analysis_ledger = connection.execute(
                    "SELECT 1 FROM credit_ledger WHERE analysis_id=?",
                    (analysis_id,),
                ).fetchone()
            assert reservation is None
            assert analysis_ledger is None
            quick = client.post("/analyses", headers=headers, json={
                "research_idea": "administrator quick search with zero credits",
                "mode": "quick",
            })
            assert quick.status_code == 201
            wait(client, quick.json()["analysis_id"], "admin-token")
            assert security.balance("admin-user") == 0
            assert security.ledger("admin-user") == before_ledger
            history = client.get("/analyses", headers=headers).json()
            assert analysis_id in [item["analysis_id"] for item in history]
            assert quick.json()["analysis_id"] in [item["analysis_id"] for item in history]

            # Exemption does not relax request or scientific provider bounds.
            assert client.post("/analyses", headers=headers, json={
                "research_idea": "bounded admin analysis",
                "mode": "full", "paper_limit": 101,
            }).status_code == 422
            assert client.post("/analyses", headers=headers, json={
                "research_idea": "", "mode": "full",
            }).status_code == 422


def test_failed_admin_analysis_has_no_refund_or_refund_wording():
    auth = StaticAuthProvider({"admin-token": AuthIdentity("admin-user", "admin@example.test", True)})
    with tempfile.TemporaryDirectory() as directory:
        with TestClient(create_app(
            settings=settings_for(Path(directory) / "admin-failure.sqlite"),
            analysis_executor=result_for,
            auth_provider=auth,
        )) as client:
            headers = {"Authorization": "Bearer admin-token"}
            security = client.app.state.components.security
            client.get("/me", headers=headers)
            security.adjust_credit("setup", "admin-user", -2, "zero admin test balance")
            security.set_role("admin-user", "admin")
            before_ledger = security.ledger("admin-user")

            created = client.post("/analyses", headers=headers, json={
                "research_idea": "administrator analysis intentionally fails",
                "mode": "full",
            })
            failed = wait(client, created.json()["analysis_id"], "admin-token")
            assert failed["status"] == "failed"
            assert "refund" not in failed["error_message"].casefold()
            assert "reserved credit" not in failed["error_message"].casefold()
            assert security.balance("admin-user") == 0
            assert security.ledger("admin-user") == before_ledger


def test_admin_cancellation_and_concurrent_submissions_never_touch_credit_ledger():
    started = Event()
    release = Event()
    auth = StaticAuthProvider({"admin-token": AuthIdentity("admin-user", "admin@example.test", True)})

    def block(record):
        started.set()
        release.wait(2)
        return result_for(record)

    with tempfile.TemporaryDirectory() as directory:
        settings = replace(
            settings_for(Path(directory) / "admin-concurrent.sqlite"),
            max_analysis_workers=1,
        )
        with TestClient(create_app(
            settings=settings,
            analysis_executor=block,
            auth_provider=auth,
        )) as client:
            headers = {"Authorization": "Bearer admin-token"}
            security = client.app.state.components.security
            client.get("/me", headers=headers)
            security.adjust_credit("setup", "admin-user", -2, "zero admin test balance")
            security.set_role("admin-user", "admin")
            before_ledger = security.ledger("admin-user")
            try:
                first = client.post("/analyses", headers=headers, json={
                    "research_idea": "first concurrent administrator analysis", "mode": "full",
                })
                assert first.status_code == 201
                assert started.wait(1)
                second = client.post("/analyses", headers=headers, json={
                    "research_idea": "second queued administrator analysis", "mode": "full",
                })
                assert second.status_code == 201
                third = client.post("/analyses", headers=headers, json={
                    "research_idea": "third bounded administrator analysis", "mode": "full",
                })
                assert third.status_code == 429
                assert client.delete(
                    f"/analyses/{second.json()['analysis_id']}", headers=headers,
                ).status_code == 204
                assert security.balance("admin-user") == 0
                assert security.ledger("admin-user") == before_ledger
            finally:
                release.set()
            wait(client, first.json()["analysis_id"], "admin-token")
            assert security.ledger("admin-user") == before_ledger


def test_forged_admin_inputs_and_metadata_fail_and_demotion_takes_effect_next_request():
    auth = StaticAuthProvider({
        "admin-looking-token": AuthIdentity(
            "ordinary-user", "owner-admin@example.test", True,
        ),
    })
    with tempfile.TemporaryDirectory() as directory:
        with TestClient(create_app(
            settings=settings_for(Path(directory) / "admin-boundary.sqlite"),
            analysis_executor=result_for,
            auth_provider=auth,
        )) as client:
            headers = {"Authorization": "Bearer admin-looking-token", "X-Admin": "true"}
            security = client.app.state.components.security
            client.get("/me", headers=headers)
            security.adjust_credit("setup", "ordinary-user", -2, "zero ordinary balance")

            forged = client.post("/analyses", headers=headers, json={
                "research_idea": "forged administrator request field", "mode": "full",
                "is_admin": True,
            })
            assert forged.status_code == 422
            assert client.post("/analyses", headers=headers, json={
                "research_idea": "metadata cannot provide administrator access", "mode": "full",
            }).status_code == 402

            security.set_role("ordinary-user", "admin")
            exempt = client.post("/analyses", headers=headers, json={
                "research_idea": "server role permits this administrator analysis", "mode": "full",
            })
            assert exempt.status_code == 201
            wait(client, exempt.json()["analysis_id"], "admin-looking-token")
            security.set_role("ordinary-user", "user")
            blocked = client.post("/analyses", headers=headers, json={
                "research_idea": "demoted administrator now needs a credit", "mode": "full",
            })
            assert blocked.status_code == 402


def test_supabase_role_metadata_cannot_enable_administrator_credit_exemption():
    provider = SupabaseAuthProvider("https://auth.example.test")
    claims = {
        "sub": "metadata-user",
        "email": "owner-admin@example.test",
        "email_verified": True,
        "user_metadata": {"role": "admin", "is_admin": True},
        "app_metadata": {"role": "admin"},
    }
    signing_key = type("SigningKey", (), {"key": "test-public-key"})()
    with tempfile.TemporaryDirectory() as directory:
        with (
            patch.object(
                provider._jwks,
                "get_signing_key_from_jwt",
                return_value=signing_key,
            ),
            patch("src.auth.jwt.decode", return_value=claims),
            TestClient(create_app(
                settings=settings_for(Path(directory) / "supabase-metadata.sqlite"),
                analysis_executor=result_for,
                auth_provider=provider,
            )) as client,
        ):
            headers = {"Authorization": "Bearer forged-metadata-token"}
            security = client.app.state.components.security
            account = client.get("/me", headers=headers)
            assert account.status_code == 200
            assert account.json()["role"] == "user"
            assert account.json()["credit_exempt"] is False
            security.adjust_credit("setup", "metadata-user", -2, "zero ordinary balance")

            response = client.post("/analyses", headers=headers, json={
                "research_idea": "Supabase role metadata must not bypass credits",
                "mode": "full",
            })
            assert response.status_code == 402


def test_concurrent_credit_reservations_cannot_overspend():
    with tempfile.TemporaryDirectory() as directory:
        database=Database(Path(directory)/"quota.sqlite");database.migrate();security=SecurityRepository(database)
        security.sync_account("u",email="u@example.test",verified=True)
        security.reserve_credit("u","used-one")
        def reserve(index:int):
            try:return security.reserve_credit("u",f"concurrent-{index}")
            except QuotaError:return None
        with ThreadPoolExecutor(max_workers=2) as pool: outcomes=list(pool.map(reserve,[1,2]))
        assert sum(value is not None for value in outcomes)==1
        assert security.balance("u")==0


def test_concurrent_account_sync_grants_lifetime_credit_once():
    with tempfile.TemporaryDirectory() as directory:
        database = Database(Path(directory) / "lifetime.sqlite")
        database.migrate()
        security = SecurityRepository(database)

        def sync(_index: int):
            return security.sync_account("same-user", email="u@example.test", verified=True)

        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(sync, range(16)))

        assert security.balance("same-user") == 2
        with database.connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS count FROM credit_ledger "
                "WHERE user_id=? AND source='free_lifetime'",
                ("same-user",),
            ).fetchone()
        assert row["count"] == 1


def test_same_normalized_email_never_receives_second_lifetime_grant():
    with tempfile.TemporaryDirectory() as directory:
        database = Database(Path(directory) / "identity.sqlite")
        database.migrate()
        security = SecurityRepository(database, lifetime_credit_hmac_secret="dedicated-test-secret")
        security.sync_account("old-user", email="  Person@Example.Test ", verified=True)
        security.delete_account_data("old-user")
        security.sync_account("new-user", email="person@example.test", verified=True)
        assert security.balance("old-user") == 2
        assert security.balance("new-user") == 0
        with database.connect() as connection:
            marker = connection.execute("SELECT * FROM lifetime_credit_identities").fetchall()
        assert len(marker) == 1
        assert "person@example.test" not in str(dict(marker[0])).lower()


def test_concurrent_different_users_with_same_email_only_grant_once():
    with tempfile.TemporaryDirectory() as directory:
        database = Database(Path(directory) / "same-email.sqlite")
        database.migrate()
        security = SecurityRepository(database, lifetime_credit_hmac_secret="dedicated-test-secret")
        def sync(index: int):
            security.sync_account(f"user-{index}", email="Same@Example.Test", verified=True)
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(sync, [1, 2]))
        assert security.balance("user-1") + security.balance("user-2") == 2


def test_credit_settlement_and_refund_are_idempotent():
    with tempfile.TemporaryDirectory() as directory:
        database = Database(Path(directory) / "credit-lifecycle.sqlite")
        database.migrate()
        security = SecurityRepository(database)
        security.sync_account("u", email="u@example.test", verified=True)

        security.reserve_credit("u", "settled-analysis")
        assert security.settle_credit("settled-analysis") is True
        assert security.settle_credit("settled-analysis") is False
        assert security.balance("u") == 1

        security.reserve_credit("u", "failed-analysis")
        assert security.release_credit("failed-analysis") is True
        assert security.release_credit("failed-analysis") is False
        assert security.balance("u") == 1


def test_guest_quick_limit_is_private_and_network_bounded():
    with tempfile.TemporaryDirectory() as directory:
        auth=StaticAuthProvider({})
        app=create_app(settings=settings_for(Path(directory)/"guest.sqlite"),analysis_executor=result_for,auth_provider=auth)
        with TestClient(app) as first:
            created=first.post("/analyses",json={"research_idea":"guest quick idea","mode":"quick"})
            assert created.status_code==201
            analysis_id=created.json()["analysis_id"]
            assert first.get(f"/analyses/{analysis_id}").status_code==200
            assert first.post("/analyses",json={"research_idea":"another quick idea","mode":"quick"}).status_code==429
            assert first.post("/analyses",json={"research_idea":"guest full idea","mode":"full"}).status_code==401
            first.cookies.clear()
            assert first.get(f"/analyses/{analysis_id}").status_code == 404


def test_user_ownership_legacy_policy_unverified_and_admin_enforcement():
    identities={"a":AuthIdentity("ua","a@example.test",True),"b":AuthIdentity("ub","b@example.test",True),"uv":AuthIdentity("uv","v@example.test",False)}
    with tempfile.TemporaryDirectory() as directory:
        with TestClient(create_app(settings=settings_for(Path(directory)/"owners.sqlite"),analysis_executor=result_for,auth_provider=StaticAuthProvider(identities))) as client:
            client.app.state.components.repository.create(NewAnalysis(
                analysis_id="legacy", research_idea="pre migration private record",
                decomposer="deterministic", query_generator="deterministic", paper_limit=10,
                configuration={"pipeline_version":"m8-v1"},
            ))
            assert client.get("/analyses/legacy",headers={"Authorization":"Bearer a"}).status_code==404
            created=client.post("/analyses",headers={"Authorization":"Bearer a"},json={"research_idea":"private owner idea","mode":"full"}).json()
            assert client.get(f"/analyses/{created['analysis_id']}",headers={"Authorization":"Bearer b"}).status_code==404
            assert client.delete(f"/analyses/{created['analysis_id']}",headers={"Authorization":"Bearer b"}).status_code==404
            assert client.post("/analyses",headers={"Authorization":"Bearer uv"},json={"research_idea":"unverified full idea","mode":"full"}).status_code==403
            assert client.get("/admin/summary",headers={"Authorization":"Bearer a"}).status_code==403
            assert client.patch("/profile",headers={"Authorization":"Bearer a"},json={"display_name":"Me","role":"admin"}).status_code==422
            security=client.app.state.components.security;security.set_role("ua","admin")
            adjusted=client.post("/admin/credits",headers={"Authorization":"Bearer a"},json={"user_id":"ub","amount":1,"reason":"manual support correction"})
            assert adjusted.status_code==200
            assert client.get("/admin/summary",headers={"Authorization":"Bearer a"}).json()["audit_log"][0]["action"]=="credit_adjustment"


def test_signed_billing_webhook_grants_once_and_checkout_return_grants_nothing():
    auth=StaticAuthProvider({"u":AuthIdentity("u","u@example.test",True)})
    with tempfile.TemporaryDirectory() as directory:
        with TestClient(create_app(settings=settings_for(Path(directory)/"stripe.sqlite"),analysis_executor=result_for,auth_provider=auth,billing_provider=FakeBilling())) as client:
            headers={"Authorization":"Bearer u"};client.get("/me",headers=headers)
            security=client.app.state.components.security;security.link_stripe_customer("u","cus_test")
            assert client.post("/billing/checkout",headers=headers).status_code==200
            assert security.balance("u")==2
            event={"id":"evt_paid","type":"invoice.paid","created":1,"data":{"object":{"id":"in_test","customer":"cus_test","lines":{"data":[{"price":{"id":"price_test_placeholder"}}]}}}}
            assert client.post("/billing/webhook",content=json.dumps(event),headers={"Stripe-Signature":"bad"}).status_code==400
            assert client.post("/billing/webhook",content=json.dumps(event),headers={"Stripe-Signature":"valid"}).json()["status"]=="processed"
            assert client.post("/billing/webhook",content=json.dumps(event),headers={"Stripe-Signature":"valid"}).json()["status"]=="duplicate"
            assert security.balance("u")==7
            cancelled={"id":"evt_cancel","type":"customer.subscription.deleted","created":3,"data":{"object":{"id":"sub_test","customer":"cus_test","status":"canceled"}}}
            failed={"id":"evt_failed","type":"invoice.payment_failed","created":2,"data":{"object":{"id":"in_failed","customer":"cus_test","subscription":"sub_test"}}}
            assert client.post("/billing/webhook",content=json.dumps(cancelled),headers={"Stripe-Signature":"valid"}).json()["status"]=="processed"
            assert client.post("/billing/webhook",content=json.dumps(failed),headers={"Stripe-Signature":"valid"}).json()["status"]=="processed"
            with client.app.state.components.database.connect() as connection:
                subscription=connection.execute("SELECT status FROM subscriptions WHERE user_id='u'").fetchone()
            assert subscription["status"]=="canceled"  # older failed-payment delivery cannot overwrite cancellation


def test_avatar_signature_and_size_validation():
    assert validate_avatar(b"\x89PNG\r\n\x1a\nbody")=="image/png"
    try: validate_avatar(b"not an image")
    except AvatarError: pass
    else: raise AssertionError("invalid avatar accepted")
    try: validate_avatar(b"\x89PNG\r\n\x1a\n"+b"x"*2_000_001)
    except AvatarError: pass
    else: raise AssertionError("oversized avatar accepted")


def test_quick_pipeline_never_initializes_paid_openai_components():
    with tempfile.TemporaryDirectory() as directory:
        settings = replace(settings_for(Path(directory)/"quick.sqlite"), openai_api_key="sk-test-present")
        with patch("src.application.analysis_service.OpenAIEmbeddingProvider") as embedding, \
             patch("src.application.analysis_service.PaperExtractor") as extractor, \
             patch("src.application.analysis_service.OpenAIDecomposer") as decomposer, \
             patch("src.application.analysis_service.OpenAIQueryGenerator") as generator:
            pipeline = build_pipeline(PipelineOptions(quick=True), settings)
        embedding.assert_not_called(); extractor.assert_not_called(); decomposer.assert_not_called(); generator.assert_not_called()
        assert pipeline.extractor is None and pipeline.gap_verifier is None


def test_full_text_pipeline_option_builds_a_full_text_client():
    with tempfile.TemporaryDirectory() as directory:
        settings = settings_for(Path(directory)/"full-text.sqlite")
        pipeline = build_pipeline(PipelineOptions(include_evidence=True, full_text=True), settings)
        assert pipeline.extractor is not None
        assert pipeline.extractor.full_text_client is not None


def test_guest_can_claim_only_its_own_still_valid_history():
    auth = StaticAuthProvider({"u": AuthIdentity("u", "u@example.test", True)})
    with tempfile.TemporaryDirectory() as directory:
        with TestClient(create_app(settings=settings_for(Path(directory)/"claim.sqlite"), analysis_executor=result_for, auth_provider=auth)) as client:
            created = client.post("/analyses", json={"research_idea":"claimable guest idea", "mode":"quick"})
            assert created.status_code == 201
            claimed = client.post("/account/claim-guest", headers={"Authorization":"Bearer u"})
            assert claimed.json()["claimed"] == 1
            history = client.get("/analyses", headers={"Authorization":"Bearer u"}).json()
            assert [item["analysis_id"] for item in history] == [created.json()["analysis_id"]]
            assert client.post("/account/claim-guest", headers={"Authorization":"Bearer u"}).json()["claimed"] == 0


def test_account_deletion_removes_auth_identity_and_reregistration_gets_no_free_grant():
    auth = StaticAuthProvider({"old": AuthIdentity("old", "same@example.test", True)})
    with tempfile.TemporaryDirectory() as directory:
        with TestClient(create_app(settings=settings_for(Path(directory)/"delete.sqlite"), analysis_executor=result_for, auth_provider=auth)) as client:
            headers = {"Authorization": "Bearer old"}
            assert client.get("/me", headers=headers).json()["credits"] == 2
            assert client.delete("/account", headers=headers).status_code == 204
            assert client.get("/me", headers=headers).status_code == 401
            with client.app.state.components.database.connect() as connection:
                account = connection.execute("SELECT * FROM accounts WHERE user_id='old'").fetchone()
            assert account["status"] == "deleted" and account["email"] is None
            auth.identities["new"] = AuthIdentity("new", " SAME@example.test ", True)
            assert client.get("/me", headers={"Authorization":"Bearer new"}).json()["credits"] == 0


def test_active_subscription_blocks_account_deletion():
    auth = StaticAuthProvider({"u": AuthIdentity("u", "u@example.test", True)})
    with tempfile.TemporaryDirectory() as directory:
        with TestClient(create_app(settings=settings_for(Path(directory)/"active-sub.sqlite"), analysis_executor=result_for, auth_provider=auth)) as client:
            headers = {"Authorization":"Bearer u"}; client.get("/me", headers=headers)
            with client.app.state.components.database.connect() as connection:
                connection.execute(
                    "INSERT INTO subscriptions(user_id,status,last_event_created,updated_at) VALUES(?,?,?,?)",
                    ("u", "active", 0, "2026-01-01T00:00:00+00:00"),
                ); connection.commit()
            response = client.delete("/account", headers=headers)
            assert response.status_code == 409
            assert "subscription" in response.json()["detail"].lower()
            assert client.get("/me", headers=headers).status_code == 200


def test_auth_deletion_failure_keeps_local_account_active():
    class FailingAuth(StaticAuthProvider):
        def delete_user(self, user_id: str) -> None:
            raise AuthenticationError("technical provider body")
    auth = FailingAuth({"u": AuthIdentity("u", "u@example.test", True)})
    with tempfile.TemporaryDirectory() as directory:
        with TestClient(create_app(settings=settings_for(Path(directory)/"delete-fail.sqlite"), analysis_executor=result_for, auth_provider=auth)) as client:
            headers = {"Authorization":"Bearer u"}; client.get("/me", headers=headers)
            response = client.delete("/account", headers=headers)
            assert response.status_code == 503
            assert response.json()["detail"] == "Account deletion is temporarily unavailable."
            assert client.get("/me", headers=headers).status_code == 200


def test_public_result_hides_provider_errors_and_markdown_uses_real_contract():
    auth = StaticAuthProvider({"u": AuthIdentity("u", "u@example.test", True)})
    def unsafe_result(record):
        return {"mode":"full", "full_text_requested":True,
                "papers":[{"id":"W1","title":"Paper","publication_year":2025}],
                "evidence":[], "gaps":[], "retrieval_failures":[{"error":"OpenAI traceback secret"}],
                "extraction_failures":["http://internal:9000 stack trace"]}
    with tempfile.TemporaryDirectory() as directory:
        with TestClient(create_app(settings=settings_for(Path(directory)/"safe.sqlite"), analysis_executor=unsafe_result, auth_provider=auth)) as client:
            headers={"Authorization":"Bearer u"}
            created=client.post("/analyses",headers=headers,json={"research_idea":"safe public result","mode":"full"}).json()
            record=wait(client,created["analysis_id"],"u")
            serialized=json.dumps(record)
            assert "traceback" not in serialized and "internal:9000" not in serialized
            assert record["result"]["failure_summary"] == {"retrieval":1,"extraction":1}
            assert record["result"]["extraction_coverage"] == {
                "selected_for_report": 1,
                "requested_for_extraction": 1,
                "successful_evidence_records": 0,
                "failed_extractions": 1,
                "not_requested_for_extraction": 0,
                "partial": True,
            }
            assert "could not be evaluated" in record["result"]["coverage_messages"][1]
    report = _markdown_report("An idea", "full", {
        "full_text_requested": True,
        "idea_assessment":{"label":"uncertain","rationale":"Coverage is bounded."},
        "papers":[{"id":"W1","title":"Correct year paper","publication_year":2025}],
        "evidence":[{"paper_id":"W1","research_objective":{"value":"Study the problem"},
                     "limitations":[{"value":"Small cohort"}],"future_work":[]}],
        "gaps":[{"title":"A candidate","description":"Description","rationale":"Rationale","supporting_paper_ids":["W1"]}],
    })
    assert "Coverage is bounded." in report and "Correct year paper (2025)" in report
    assert "Small cohort" in report and "paper.get('year')" not in report


def test_final_coverage_accounts_once_and_separates_attempt_failures():
    records = [
        {
            "paper_id": "full", "title": "Full", "full_text_requested": True,
            "full_text_attempted": True, "final_evidence_level": "full_text",
            "full_text_status": "usable", "full_text_extraction_succeeded": True,
            "full_text_source_format": "pdf", "truncated": False,
            "inspected_section_types": ["methods"], "fallback_explanation": None,
            "final_state": "success", "failure_category": None,
        },
        {
            "paper_id": "fetch", "title": "Fetch", "full_text_requested": True,
            "full_text_attempted": True, "final_evidence_level": "abstract_fallback",
            "full_text_status": "fetch_failed", "full_text_extraction_succeeded": False,
            "full_text_source_format": "pdf", "truncated": False,
            "inspected_section_types": [],
            "fallback_explanation": "Full-text fetch failed; abstract fallback succeeded.",
            "final_state": "success", "failure_category": None,
        },
        {
            "paper_id": "meta", "title": "Metadata", "full_text_requested": True,
            "full_text_attempted": False, "final_evidence_level": "metadata_only",
            "full_text_status": "unavailable", "full_text_extraction_succeeded": False,
            "full_text_source_format": None, "truncated": False,
            "inspected_section_types": [],
            "fallback_explanation": "Open full text was unavailable; title metadata fallback succeeded.",
            "final_state": "success", "failure_category": None,
        },
        {
            "paper_id": "invalid", "title": "Invalid", "full_text_requested": True,
            "full_text_attempted": True, "final_evidence_level": "none",
            "full_text_status": "parse_failed", "full_text_extraction_succeeded": False,
            "full_text_source_format": "pdf", "truncated": False,
            "inspected_section_types": [],
            "fallback_explanation": "Full-text parsing failed and the abstract fallback did not produce validated evidence.",
            "final_state": "failure",
            "failure_category": "model_schema_evidence_validation",
        },
    ]
    public = public_analysis_result({
        "papers": [{"id": item["paper_id"]} for item in records],
        "evidence": [{"paper_id": item["paper_id"]} for item in records[:3]],
        "paper_coverage": records,
        "extraction_failures": ["private provider detail"],
    })
    coverage = public["extraction_coverage"]
    self_total = (
        coverage["full_text_successes"]
        + coverage["abstract_successes"]
        + coverage["abstract_fallback_successes"]
        + coverage["metadata_only_successes"]
        + coverage["final_failures"]
    )
    assert coverage["requested_for_extraction"] == self_total == 4
    assert coverage["successful_evidence_records"] == 3
    assert coverage["failed_extractions"] == 1
    assert public["full_text_attempt_summary"]["fetch_failures"] == 1
    assert public["full_text_attempt_summary"]["parse_failures"] == 1
    assert public["full_text_attempt_summary"]["model_schema_evidence_validation_failures"] == 1
    assert "extraction_failures" not in public


def test_markdown_renders_supported_scientific_sections_and_provenance_only():
    full_text = {
        "source": "full_text", "confidence": 0.9,
        "section_type": "results", "section_heading": "Results", "section_id": "s-results",
    }
    report = _markdown_report("An idea", "full", {
        "full_text_requested": True,
        "papers": [
            {"id": "W1", "title": "Canonical paper", "publication_year": 2025,
             "doi": "10.1/example"},
            {"id": "W1-alias", "title": "Canonical-paper", "publication_year": 2025,
             "doi": "https://doi.org/10.1/EXAMPLE"},
        ],
        "evidence": [{
            "paper_id": "W1", "title": "Canonical paper", "study_type": "empirical",
            "population_or_setting": [{
                "value": "Enterprise workflows", "evidence_text": "enterprise workflows",
                "source": "abstract", "confidence": 0.9,
            }],
            "datasets": [{
                "value": "EnterpriseFlow-500", "evidence_text": "EnterpriseFlow-500",
                **full_text,
            }],
            "sample_size": [{
                "value": "500 examples", "evidence_text": "500 examples", **full_text,
            }],
            "comparison_or_baseline": [{
                "value": "GPT-4 baseline", "evidence_text": "GPT-4 baseline", **full_text,
            }],
            "evaluation_metrics": [{
                "value": "Exact Match", "evidence_text": "Exact Match", **full_text,
            }],
            "main_findings": [{
                "value": "Hallucinations decreased", "evidence_text": "Hallucinations decreased",
                **full_text,
            }],
            "limitations": [], "future_work": [],
        }],
        "paper_coverage": [{
            "paper_id": "W1", "title": "Canonical paper", "full_text_requested": True,
            "full_text_attempted": True, "final_evidence_level": "full_text",
            "full_text_status": "usable", "full_text_extraction_succeeded": True,
            "full_text_source_format": "pdf", "truncated": False,
            "inspected_section_types": ["results"], "fallback_explanation": None,
            "final_state": "success", "failure_category": None,
        }],
        "gaps": [],
    })
    for heading in (
        "Populations and settings", "Datasets and data modalities", "Sample sizes",
        "Comparisons and baselines", "Evaluation metrics", "Main findings",
        "Per-paper coverage",
    ):
        assert f"## {heading}" in report
    assert "> EnterpriseFlow-500" in report
    assert "Evidence source: full text; heading: Results; section type: results; section ID: s-results." in report
    assert "Evidence source: abstract." in report
    assert "full text: usable (PDF)" in report
    assert "\n## Limitations and future work\n" not in report
    relevant = report.split("## Relevant papers", 1)[1].split("## Coverage limitations", 1)[0]
    assert relevant.count("Canonical paper") == 1


def test_full_text_request_is_persisted_for_the_pipeline_executor():
    auth = StaticAuthProvider({"u": AuthIdentity("u", "u@example.test", True)})
    seen: dict[str, object] = {}
    def inspect(record):
        seen.update(record.configuration)
        return {"mode":"full", "full_text_requested":record.configuration.get("full_text"),
                "papers":[], "evidence":[], "gaps":[]}
    with tempfile.TemporaryDirectory() as directory:
        with TestClient(create_app(settings=settings_for(Path(directory)/"full-text-flag.sqlite"), analysis_executor=inspect, auth_provider=auth)) as client:
            created = client.post("/analyses", headers={"Authorization":"Bearer u"}, json={
                "research_idea":"full text propagation", "mode":"full", "full_text":True,
            })
            record = wait(client, created.json()["analysis_id"], "u")
        assert seen["full_text"] is True
        assert record["result"]["full_text_requested"] is True
