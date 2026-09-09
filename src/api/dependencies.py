"""Request authentication and server-owned principal resolution."""

from __future__ import annotations

from uuid import uuid4

from fastapi import HTTPException, Request, status

from src.auth import AuthenticationError, Principal, network_rate_key, verify_guest_cookie

GUEST_COOKIE = "research_gap_guest"


def principal_for(request: Request, *, force_guest: bool = False) -> Principal:
    components = request.app.state.components
    authorization = request.headers.get("authorization", "")
    if authorization:
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() != "bearer" or not token or components.auth_provider is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid authentication token.")
        try:
            identity = components.auth_provider.verify(token)
        except AuthenticationError:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired access token.")
        account = components.security.sync_account(
            identity.user_id, email=identity.email, verified=identity.email_verified,
        )
        if account["status"] != "active":
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="This account is suspended.")
        return Principal(
            "user", identity.user_id, identity.email, bool(account["email_verified"]),
            str(account["role"]), str(account["status"]),
        )
    settings = components.settings.web
    cookie = request.cookies.get(GUEST_COOKIE)
    guest_id = verify_guest_cookie(cookie, settings.guest_cookie_secret) if settings else None
    if guest_id and components.security.guest_active(guest_id):
        return Principal("guest", guest_id)
    if components.trusted_local_mode and not force_guest:
        return Principal("local", "local-development", email_verified=True)
    guest_id = str(uuid4())
    request.state.new_guest_id = guest_id
    address = request.client.host if request.client else "unknown"
    network = network_rate_key(address, settings.guest_cookie_secret)
    components.security.touch_guest(guest_id, network, retention_hours=settings.guest_retention_hours)
    return Principal("guest", guest_id)


def require_user(request: Request) -> Principal:
    principal = principal_for(request)
    if principal.kind != "user":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Sign in is required.")
    return principal


def require_admin(request: Request) -> Principal:
    principal = require_user(request)
    if principal.role != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Administrator access is required.")
    return principal
