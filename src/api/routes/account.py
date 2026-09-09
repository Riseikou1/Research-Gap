"""Current-account profile, quota, guest claiming, avatar, and deletion APIs."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from src.api.dependencies import GUEST_COOKIE, principal_for, require_user
from src.auth import verify_guest_cookie
from src.storage import AvatarError, validate_avatar

router = APIRouter(tags=["account"])


class ProfileUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)
    display_name: str = Field(min_length=1, max_length=80)


@router.get("/me")
def me(request: Request) -> dict[str, object]:
    principal = principal_for(request)
    if principal.kind != "user":
        return {"kind": principal.kind, "signed_in": False, "verified": False,
                "role": "user", "credits": 0, "profile": None}
    components = request.app.state.components
    account = components.security.account(principal.principal_id) or {}
    return {
        "kind": "user", "signed_in": True, "verified": principal.email_verified,
        "role": principal.role, "credits": components.security.balance(principal.principal_id),
        "profile": {"display_name": account.get("display_name", ""),
                    "avatar_url": account.get("avatar_url"), "email": account.get("email")},
        "plan": {"test_mode": components.settings.web.stripe_test_mode,
                 "price_usd": components.settings.web.paid_price_usd,
                 "credits_per_cycle": components.settings.web.paid_cycle_credits},
    }


@router.patch("/profile")
def update_profile(payload: ProfileUpdate, request: Request) -> dict[str, str]:
    principal = require_user(request)
    request.app.state.components.security.profile_update(principal.principal_id, display_name=payload.display_name)
    return {"display_name": payload.display_name}


@router.post("/profile/avatar")
async def upload_avatar(request: Request) -> dict[str, str]:
    principal = require_user(request)
    declared = request.headers.get("content-length")
    if declared and (not declared.isdigit() or int(declared) > 2_000_000):
        raise HTTPException(status_code=413, detail="Avatar exceeds the 2 MB limit.")
    buffer = bytearray()
    async for chunk in request.stream():
        buffer.extend(chunk)
        if len(buffer) > 2_000_000:
            raise HTTPException(status_code=413, detail="Avatar exceeds the 2 MB limit.")
    content = bytes(buffer)
    try:
        media_type = validate_avatar(content)
    except AvatarError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    storage = request.app.state.components.avatar_storage
    if storage is None:
        raise HTTPException(status_code=503, detail="Avatar storage is not configured.")
    try:
        url = storage.upload(principal.principal_id, content, media_type)
    except AvatarError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    request.app.state.components.security.profile_update(principal.principal_id, avatar_url=url)
    return {"avatar_url": url}


@router.post("/account/claim-guest")
def claim_guest(request: Request) -> dict[str, int]:
    principal = require_user(request)
    settings = request.app.state.components.settings.web
    guest_id = verify_guest_cookie(request.cookies.get(GUEST_COOKIE), settings.guest_cookie_secret)
    if not guest_id:
        return {"claimed": 0}
    claimed = request.app.state.components.repository.claim_guest(guest_id, principal.principal_id)
    return {"claimed": claimed}


@router.delete("/account", status_code=204)
def delete_account(request: Request) -> None:
    principal = require_user(request)
    components = request.app.state.components
    if components.repository.count_active_for_owner("user", principal.principal_id):
        raise HTTPException(status_code=409, detail="Wait for active analyses to finish before deleting the account.")
    if components.avatar_storage:
        components.avatar_storage.delete(principal.principal_id)
    components.security.delete_account_data(principal.principal_id)
