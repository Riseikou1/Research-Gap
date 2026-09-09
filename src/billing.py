"""Narrow Stripe test-mode Checkout, portal, and webhook boundary."""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from typing import Protocol
from urllib import error, parse, request


class BillingError(RuntimeError):
    pass


class BillingProvider(Protocol):
    def create_checkout(self, *, customer_id: str | None, email: str | None, user_id: str,
                        price_id: str, success_url: str, cancel_url: str) -> dict[str, str]: ...
    def create_portal(self, *, customer_id: str, return_url: str) -> dict[str, str]: ...
    def verify_webhook(self, payload: bytes, signature: str) -> dict[str, object]: ...


class StripeProvider:
    def __init__(self, secret_key: str, webhook_secret: str | None = None) -> None:
        self.secret_key = secret_key
        self.webhook_secret = webhook_secret

    def _post(self, endpoint: str, values: dict[str, str]) -> dict[str, object]:
        body = parse.urlencode(values).encode()
        req = request.Request(
            f"https://api.stripe.com/v1/{endpoint}", data=body, method="POST",
            headers={"Authorization": f"Bearer {self.secret_key}",
                     "Content-Type": "application/x-www-form-urlencoded"},
        )
        try:
            with request.urlopen(req, timeout=20) as response:
                return json.loads(response.read())
        except (error.URLError, ValueError) as exc:
            raise BillingError("The payment provider request failed.") from exc

    def create_checkout(self, *, customer_id: str | None, email: str | None, user_id: str,
                        price_id: str, success_url: str, cancel_url: str) -> dict[str, str]:
        values = {
            "mode": "subscription", "line_items[0][price]": price_id,
            "line_items[0][quantity]": "1", "success_url": success_url,
            "cancel_url": cancel_url, "client_reference_id": user_id,
            "subscription_data[metadata][research_gap_user_id]": user_id,
        }
        if customer_id:
            values["customer"] = customer_id
        elif email:
            values["customer_email"] = email
        payload = self._post("checkout/sessions", values)
        return {"id": str(payload["id"]), "url": str(payload["url"])}

    def create_portal(self, *, customer_id: str, return_url: str) -> dict[str, str]:
        payload = self._post("billing_portal/sessions", {"customer": customer_id, "return_url": return_url})
        return {"id": str(payload["id"]), "url": str(payload["url"])}

    def verify_webhook(self, payload: bytes, signature: str) -> dict[str, object]:
        if not self.webhook_secret:
            raise BillingError("STRIPE_WEBHOOK_SECRET is not configured.")
        parts: dict[str, list[str]] = {}
        for item in signature.split(","):
            key, _, value = item.partition("=")
            parts.setdefault(key, []).append(value)
        try:
            timestamp = int(parts["t"][0])
        except (KeyError, ValueError, IndexError) as exc:
            raise BillingError("Invalid payment webhook signature.") from exc
        if abs(int(time.time()) - timestamp) > 300:
            raise BillingError("Expired payment webhook signature.")
        expected = hmac.new(
            self.webhook_secret.encode(), str(timestamp).encode() + b"." + payload, hashlib.sha256
        ).hexdigest()
        if not any(hmac.compare_digest(expected, candidate) for candidate in parts.get("v1", [])):
            raise BillingError("Invalid payment webhook signature.")
        try:
            event = json.loads(payload)
        except (ValueError, UnicodeDecodeError) as exc:
            raise BillingError("Invalid payment webhook payload.") from exc
        if not isinstance(event, dict):
            raise BillingError("Invalid payment webhook payload.")
        return event
