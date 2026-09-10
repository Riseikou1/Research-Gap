"""Deterministic Milestone 9 ownership, quota, auth, billing, and avatar checks."""

from __future__ import annotations

import json
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from src.api.app import create_app
from src.auth import AuthIdentity, StaticAuthProvider
from src.billing import BillingError
from src.config import Settings
from src.persistence.database import Database
from src.persistence.security import QuotaError, SecurityRepository
from src.persistence.models import NewAnalysis
from src.storage import AvatarError, validate_avatar
from src.application.analysis_service import PipelineOptions, build_pipeline


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
            wait(client, first.json()["analysis_id"], "verified")
            wait(client, second.json()["analysis_id"], "verified")
            blocked = client.post("/analyses", headers={"Authorization": "Bearer verified"}, json={"research_idea":"third valid idea","mode":"full"})
            assert blocked.status_code == 402
            assert client.get("/me", headers={"Authorization": "Bearer verified"}).json()["credits"] == 0
            client.app.state.components.security.adjust_credit("admin", "u1", 1, "test failure refund")
            failed = client.post("/analyses", headers={"Authorization": "Bearer verified"}, json={"research_idea":"this fails safely","mode":"full"})
            assert wait(client, failed.json()["analysis_id"], "verified")["status"] == "failed"
            assert client.get("/me", headers={"Authorization": "Bearer verified"}).json()["credits"] == 1


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
