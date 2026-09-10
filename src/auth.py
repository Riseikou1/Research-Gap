"""Managed-auth boundary and signed guest-session identities."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
from dataclasses import dataclass
from typing import Protocol
from urllib import error, parse, request
from uuid import uuid4

import jwt
from jwt import PyJWKClient


class AuthenticationError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class AuthIdentity:
    user_id: str
    email: str | None
    email_verified: bool


@dataclass(frozen=True, slots=True)
class Principal:
    kind: str
    principal_id: str
    email: str | None = None
    email_verified: bool = False
    role: str = "user"
    status: str = "active"

    @property
    def signed_in(self) -> bool:
        return self.kind in {"user", "local"}


class AuthProvider(Protocol):
    def verify(self, token: str) -> AuthIdentity: ...
    def create_admin_user(self, email: str, password: str, display_name: str) -> AuthIdentity: ...
    def delete_user(self, user_id: str) -> None: ...


class SupabaseAuthProvider:
    """Verify Supabase JWTs through project JWKS and create users through the admin API."""

    def __init__(self, url: str, *, audience: str = "authenticated", issuer: str | None = None,
                 service_role_key: str | None = None) -> None:
        self.url = url.rstrip("/")
        self.audience = audience
        self.issuer = issuer or f"{self.url}/auth/v1"
        self.service_role_key = service_role_key
        self._jwks = PyJWKClient(f"{self.url}/auth/v1/.well-known/jwks.json", cache_keys=True)

    def verify(self, token: str) -> AuthIdentity:
        try:
            key = self._jwks.get_signing_key_from_jwt(token)
            claims = jwt.decode(
                token, key.key, algorithms=["RS256", "ES256"], audience=self.audience,
                issuer=self.issuer, options={"require": ["exp", "sub"]},
            )
        except Exception as exc:
            raise AuthenticationError("Invalid or expired access token.") from exc
        email = claims.get("email")
        metadata = claims.get("user_metadata") if isinstance(claims.get("user_metadata"), dict) else {}
        verified = bool(claims.get("email_confirmed_at") or claims.get("email_verified") or metadata.get("email_verified"))
        return AuthIdentity(str(claims["sub"]), str(email) if email else None, verified)

    def create_admin_user(self, email: str, password: str, display_name: str) -> AuthIdentity:
        if not self.service_role_key:
            raise AuthenticationError("SUPABASE_SERVICE_ROLE_KEY is required for administrator bootstrap.")
        body = json.dumps({
            "email": email, "password": password, "email_confirm": True,
            "user_metadata": {"display_name": display_name},
        }).encode()
        req = request.Request(
            f"{self.url}/auth/v1/admin/users", data=body, method="POST",
            headers={"Content-Type": "application/json", "apikey": self.service_role_key,
                     "Authorization": f"Bearer {self.service_role_key}"},
        )
        try:
            with request.urlopen(req, timeout=20) as response:
                payload = json.loads(response.read())
        except (error.URLError, ValueError) as exc:
            raise AuthenticationError("The managed-auth administrator account could not be created.") from exc
        return AuthIdentity(str(payload["id"]), str(payload.get("email") or email), True)

    def delete_user(self, user_id: str) -> None:
        """Delete an Auth identity through the privileged server-only Admin API."""
        if not self.service_role_key:
            raise AuthenticationError("Managed authentication deletion is unavailable.")
        req = request.Request(
            f"{self.url}/auth/v1/admin/users/{parse.quote(user_id, safe='')}", method="DELETE",
            headers={"apikey": self.service_role_key,
                     "Authorization": f"Bearer {self.service_role_key}"},
        )
        try:
            with request.urlopen(req, timeout=20):
                return
        except error.HTTPError as exc:
            if exc.code == 404:
                return
            raise AuthenticationError("Managed authentication deletion failed.") from exc
        except error.URLError as exc:
            raise AuthenticationError("Managed authentication deletion failed.") from exc


class StaticAuthProvider:
    """Deterministic provider for local tests; it never accepts arbitrary production tokens."""

    def __init__(self, identities: dict[str, AuthIdentity]) -> None:
        self.identities = identities

    def verify(self, token: str) -> AuthIdentity:
        try:
            return self.identities[token]
        except KeyError as exc:
            raise AuthenticationError("Invalid or expired access token.") from exc

    def create_admin_user(self, email: str, password: str, display_name: str) -> AuthIdentity:
        identity = AuthIdentity(str(uuid4()), email, True)
        self.identities[f"admin:{identity.user_id}"] = identity
        return identity

    def delete_user(self, user_id: str) -> None:
        self.identities = {
            token: identity for token, identity in self.identities.items()
            if identity.user_id != user_id
        }


def sign_guest_id(guest_id: str, secret: str) -> str:
    encoded = base64.urlsafe_b64encode(guest_id.encode()).decode().rstrip("=")
    signature = hmac.new(secret.encode(), encoded.encode(), hashlib.sha256).hexdigest()
    return f"{encoded}.{signature}"


def verify_guest_cookie(value: str | None, secret: str) -> str | None:
    if not value or "." not in value:
        return None
    encoded, signature = value.rsplit(".", 1)
    expected = hmac.new(secret.encode(), encoded.encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(signature, expected):
        return None
    try:
        return base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4)).decode()
    except (ValueError, UnicodeDecodeError):
        return None


def network_rate_key(address: str, secret: str) -> str:
    """Stable pseudonymous network key; no raw address is persisted."""
    return hmac.new(secret.encode(), f"network:{address}".encode(), hashlib.sha256).hexdigest()
