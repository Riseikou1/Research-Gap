"""Server-owned placeholder plan Checkout, portal, and signed webhook routes."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from src.api.dependencies import require_user
from src.billing import BillingError

router = APIRouter(prefix="/billing", tags=["billing"])


@router.get("/plan")
def plan(request: Request) -> dict[str, object]:
    web = request.app.state.components.settings.web
    return {"price_usd": web.paid_price_usd, "interval": "month", "credits_per_cycle": web.paid_cycle_credits,
            "test_mode": web.stripe_test_mode, "configured": bool(web.stripe_price_id)}


@router.post("/checkout")
def checkout(request: Request) -> dict[str, str]:
    principal = require_user(request)
    components = request.app.state.components
    provider, web = components.billing_provider, components.settings.web
    if not provider or not web.stripe_price_id:
        raise HTTPException(status_code=503, detail="Stripe test Checkout is not configured.")
    if not components.security.record_rate_event(f"user:{principal.principal_id}", "checkout", limit=5):
        raise HTTPException(status_code=429, detail="Checkout rate limit reached.")
    account = components.security.account(principal.principal_id) or {}
    try:
        return provider.create_checkout(
            customer_id=account.get("stripe_customer_id"), email=principal.email,
            user_id=principal.principal_id, price_id=web.stripe_price_id,
            success_url=f"{web.app_url}/pricing?checkout=returned",
            cancel_url=f"{web.app_url}/pricing?checkout=cancelled",
        )
    except BillingError as exc:
        raise HTTPException(status_code=502, detail=str(exc))


@router.post("/portal")
def portal(request: Request) -> dict[str, str]:
    principal = require_user(request)
    components = request.app.state.components
    account = components.security.account(principal.principal_id) or {}
    customer = account.get("stripe_customer_id")
    if not components.billing_provider or not customer:
        raise HTTPException(status_code=409, detail="No Stripe customer is linked to this account.")
    try:
        return components.billing_provider.create_portal(
            customer_id=str(customer), return_url=f"{components.settings.web.app_url}/pricing",
        )
    except BillingError as exc:
        raise HTTPException(status_code=502, detail=str(exc))


@router.post("/webhook")
async def webhook(request: Request) -> dict[str, str]:
    components = request.app.state.components
    if not components.billing_provider:
        raise HTTPException(status_code=503, detail="Stripe webhooks are not configured.")
    declared = request.headers.get("content-length")
    if declared and (not declared.isdigit() or int(declared) > 1_000_000):
        raise HTTPException(status_code=413, detail="Webhook payload is too large.")
    buffer = bytearray()
    async for chunk in request.stream():
        buffer.extend(chunk)
        if len(buffer) > 1_000_000:
            raise HTTPException(status_code=413, detail="Webhook payload is too large.")
    payload = bytes(buffer)
    try:
        event = components.billing_provider.verify_webhook(payload, request.headers.get("stripe-signature", ""))
    except BillingError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    try:
        outcome = components.security.process_payment_event(
            event, cycle_credits=components.settings.web.paid_cycle_credits,
            expected_price_id=components.settings.web.stripe_price_id,
        )
    except ValueError as exc:
        components.security.record_failed_payment_event(event, str(exc))
        raise HTTPException(status_code=400, detail=str(exc))
    return {"status": outcome}
