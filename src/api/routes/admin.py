"""Server-authorized owner operations; no public role assignment exists."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from src.api.dependencies import require_admin

router = APIRouter(prefix="/admin", tags=["admin"])


class CreditAdjustment(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)
    user_id: str = Field(min_length=1, max_length=128)
    amount: int = Field(ge=-1000, le=1000)
    reason: str = Field(min_length=3, max_length=500)


class StatusChange(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)
    status: str
    reason: str = Field(min_length=3, max_length=500)


@router.get("/summary")
def summary(request: Request) -> dict[str, object]:
    require_admin(request)
    return request.app.state.components.security.admin_summary()


@router.get("/users")
def users(request: Request) -> list[dict[str, object]]:
    require_admin(request)
    return request.app.state.components.security.list_users()


@router.post("/credits")
def adjust_credits(payload: CreditAdjustment, request: Request) -> dict[str, int]:
    admin = require_admin(request)
    if payload.amount == 0:
        raise HTTPException(status_code=422, detail="Credit adjustment cannot be zero.")
    security = request.app.state.components.security
    if security.account(payload.user_id) is None:
        raise HTTPException(status_code=404, detail="User not found.")
    security.adjust_credit(admin.principal_id, payload.user_id, payload.amount, payload.reason)
    return {"credits": security.balance(payload.user_id)}


@router.post("/users/{user_id}/status")
def update_status(user_id: str, payload: StatusChange, request: Request) -> dict[str, str]:
    admin = require_admin(request)
    if payload.status not in {"active", "suspended"}:
        raise HTTPException(status_code=422, detail="Status must be active or suspended.")
    if user_id == admin.principal_id and payload.status == "suspended":
        raise HTTPException(status_code=409, detail="An administrator cannot suspend their current account.")
    request.app.state.components.security.set_status(
        admin.principal_id, user_id, payload.status, payload.reason,
    )
    return {"status": payload.status}
