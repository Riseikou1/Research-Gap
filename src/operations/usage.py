"""Normalized provider usage records without prompt or response content."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ProviderUsage(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)

    provider: str = Field(min_length=1, max_length=40)
    model: str = Field(min_length=1, max_length=120)
    stage: str = Field(min_length=1, max_length=80)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    cost_usd: float | None = Field(default=None, ge=0)
    timestamp: datetime
    cache_hit: bool = False


def usage_from_response(response: Any, *, provider: str, model: str, stage: str) -> ProviderUsage:
    usage = getattr(response, "usage", None)

    def value(name: str) -> int | None:
        raw = getattr(usage, name, None)
        if raw is None and isinstance(usage, dict):
            raw = usage.get(name)
        return raw if isinstance(raw, int) and not isinstance(raw, bool) and raw >= 0 else None

    input_tokens = value("input_tokens")
    if input_tokens is None:
        input_tokens = value("prompt_tokens")
    output_tokens = value("output_tokens")
    if output_tokens is None:
        output_tokens = value("completion_tokens")
    total_tokens = value("total_tokens")
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens
    return ProviderUsage(
        provider=provider, model=model, stage=stage,
        input_tokens=input_tokens, output_tokens=output_tokens,
        total_tokens=total_tokens, timestamp=datetime.now(timezone.utc),
    )
