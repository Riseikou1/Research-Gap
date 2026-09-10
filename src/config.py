"""Validated runtime configuration for the Research GAP pipeline."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

DEFAULT_OPENAI_MODEL = "gpt-5.6-luna"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"


def openai_api_key() -> str | None:
    return os.getenv("OPENAI_API_KEY", "").strip() or None


def openai_model() -> str:
    return os.getenv("OPENAI_MODEL", "").strip() or DEFAULT_OPENAI_MODEL


def openai_extraction_model() -> str:
    return os.getenv("OPENAI_EXTRACTION_MODEL", "").strip() or openai_model()


def openai_embedding_model() -> str:
    return os.getenv("OPENAI_EMBEDDING_MODEL", "").strip() or DEFAULT_EMBEDDING_MODEL


def cache_dir() -> Path:
    """Resolve the cache when a component is built, not when its module is imported."""
    value = os.getenv("RESEARCH_GAP_CACHE_DIR", "").strip()
    return Path(value).expanduser() if value else PROJECT_ROOT / "data" / "cache"


def database_path() -> Path:
    """Return the durable analysis-history database path."""
    value = os.getenv("RESEARCH_GAP_DATABASE_PATH", "").strip()
    return Path(value).expanduser() if value else PROJECT_ROOT / "data" / "research_gap.sqlite3"

def database_url() -> str | None:
    """Return the PostgreSQL connection URL when configured."""
    return os.getenv("DATABASE_URL", "").strip() or None

class ConfigurationError(ValueError):
    """Raised when environment configuration is malformed or out of bounds."""


@dataclass(frozen=True)
class OpenAlexSettings:
    api_key: str | None
    mailto: str | None
    per_route_limit: int
    max_candidates: int
    max_workers: int
    timeout_seconds: float
    max_retries: int
    retrieval_cache_ttl_seconds: float


@dataclass(frozen=True)
class RankingSettings:
    embedding_model: str
    embedding_batch_size: int
    lexical_weight: float
    semantic_weight: float
    constraint_weight: float
    semantic_fallback: Literal["lexical", "error"]


@dataclass(frozen=True)
class FullTextSettings:
    timeout_seconds: float
    max_bytes: int
    max_document_chars: int
    max_section_chars: int
    max_chunk_chars: int
    max_context_chars: int
    max_redirects: int
    negative_cache_ttl_seconds: float


@dataclass(frozen=True)
class WebSettings:
    app_url: str
    allowed_origins: tuple[str, ...]
    auth_url: str | None
    auth_anon_key: str | None
    auth_service_role_key: str | None
    auth_jwt_audience: str
    auth_jwt_issuer: str | None
    guest_cookie_secret: str
    secure_cookies: bool
    guest_retention_hours: int
    guest_quick_limit: int
    user_quick_daily_limit: int
    max_active_analyses_per_principal: int
    free_lifetime_credits: int
    paid_cycle_credits: int
    paid_price_usd: float
    stripe_secret_key: str | None
    stripe_webhook_secret: str | None
    stripe_price_id: str | None
    stripe_test_mode: bool
    acknowledge_live_pricing: bool
    trusted_local_mode: bool


@dataclass(frozen=True)
class Settings:
    openai_api_key: str | None
    openai_model: str
    extraction_model: str
    openalex: OpenAlexSettings
    ranking: RankingSettings
    full_text: FullTextSettings
    evidence_limit: int
    extraction_workers: int
    extraction_batch_size: int
    cache_directory: Path = field(default_factory=cache_dir)
    database_url: str | None = None
    analysis_database_path: Path = field(default_factory=database_path)
    max_analysis_workers: int = 2
    web: WebSettings | None = None

    @classmethod
    def from_env(cls) -> "Settings":
        def integer(name: str, default: int, *, minimum: int = 1, maximum: int | None = None) -> int:
            try:
                value = int(os.getenv(name, str(default)))
            except ValueError as exc:
                raise ConfigurationError(f"{name} must be an integer") from exc
            if value < minimum:
                requirement = "positive" if minimum == 1 else f"at least {minimum}"
                raise ConfigurationError(f"{name} must be {requirement}")
            if maximum is not None and value > maximum:
                raise ConfigurationError(f"{name} must not exceed {maximum}")
            return value

        def number(
            name: str,
            default: float,
            *,
            positive: bool = False,
            maximum: float | None = None,
        ) -> float:
            try:
                value = float(os.getenv(name, str(default)))
            except ValueError as exc:
                raise ConfigurationError(f"{name} must be a number") from exc
            if not math.isfinite(value):
                raise ConfigurationError(f"{name} must be finite")
            if value < 0 or (positive and value == 0):
                requirement = "positive" if positive else "non-negative"
                raise ConfigurationError(f"{name} must be {requirement}")
            if maximum is not None and value > maximum:
                raise ConfigurationError(f"{name} must not exceed {maximum:g}")
            return value

        lexical = number("RESEARCH_GAP_LEXICAL_WEIGHT", 0.4)
        semantic = number("RESEARCH_GAP_SEMANTIC_WEIGHT", 0.6)
        if lexical == 0 and semantic == 0:
            raise ConfigurationError("weights cannot both be zero")
        if not math.isfinite(lexical + semantic):
            raise ConfigurationError("combined ranking weights must be finite")
        constraint_weight = number("RESEARCH_GAP_CONSTRAINT_WEIGHT", 0.15)
        if constraint_weight > 1:
            raise ConfigurationError("RESEARCH_GAP_CONSTRAINT_WEIGHT must be between 0 and 1")
        fallback = os.getenv("RESEARCH_GAP_SEMANTIC_FALLBACK", "lexical").strip()
        if fallback not in {"lexical", "error"}:
            raise ConfigurationError("RESEARCH_GAP_SEMANTIC_FALLBACK must be 'lexical' or 'error'")

        app_url = os.getenv("RESEARCH_GAP_APP_URL", "http://localhost:3000").strip().rstrip("/")
        origins = tuple(
            origin.strip().rstrip("/") for origin in
            os.getenv("RESEARCH_GAP_ALLOWED_ORIGINS", app_url).split(",") if origin.strip()
        )
        if "*" in origins:
            raise ConfigurationError("RESEARCH_GAP_ALLOWED_ORIGINS cannot contain '*' with credentials")
        secure_cookies = os.getenv("RESEARCH_GAP_SECURE_COOKIES", "false").lower() in {"1", "true", "yes"}
        stripe_secret = os.getenv("STRIPE_SECRET_KEY", "").strip() or None
        stripe_test_mode = not bool(stripe_secret and stripe_secret.startswith("sk_live_"))
        acknowledge_live = os.getenv("RESEARCH_GAP_ACKNOWLEDGE_LIVE_PRICING", "false").lower() in {"1", "true", "yes"}
        price_id = os.getenv("STRIPE_PRICE_ID", "").strip() or None
        if not stripe_test_mode and not acknowledge_live:
            raise ConfigurationError(
                "Live Stripe keys require RESEARCH_GAP_ACKNOWLEDGE_LIVE_PRICING=true; "
                "the $1/5-credit plan is test placeholder pricing"
            )
        guest_secret = os.getenv("RESEARCH_GAP_GUEST_COOKIE_SECRET", "local-development-only-change-me")
        if secure_cookies and guest_secret == "local-development-only-change-me":
            raise ConfigurationError("Set a strong RESEARCH_GAP_GUEST_COOKIE_SECRET before secure production use")
        web = WebSettings(
            app_url=app_url,
            allowed_origins=origins,
            auth_url=os.getenv("SUPABASE_URL", "").strip().rstrip("/") or None,
            auth_anon_key=os.getenv("SUPABASE_ANON_KEY", "").strip() or None,
            auth_service_role_key=os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip() or None,
            auth_jwt_audience=os.getenv("SUPABASE_JWT_AUDIENCE", "authenticated").strip(),
            auth_jwt_issuer=os.getenv("SUPABASE_JWT_ISSUER", "").strip().rstrip("/") or None,
            guest_cookie_secret=guest_secret,
            secure_cookies=secure_cookies,
            guest_retention_hours=integer("RESEARCH_GAP_GUEST_RETENTION_HOURS", 72),
            guest_quick_limit=integer("RESEARCH_GAP_GUEST_QUICK_LIMIT", 1),
            user_quick_daily_limit=integer("RESEARCH_GAP_USER_QUICK_DAILY_LIMIT", 10),
            max_active_analyses_per_principal=integer("RESEARCH_GAP_MAX_ACTIVE_PER_PRINCIPAL", 2),
            free_lifetime_credits=2,
            paid_cycle_credits=integer("RESEARCH_GAP_PAID_CYCLE_CREDITS", 5),
            paid_price_usd=number("RESEARCH_GAP_PAID_PRICE_DISPLAY_USD", 1.0, positive=True),
            stripe_secret_key=stripe_secret,
            stripe_webhook_secret=os.getenv("STRIPE_WEBHOOK_SECRET", "").strip() or None,
            stripe_price_id=price_id,
            stripe_test_mode=stripe_test_mode,
            acknowledge_live_pricing=acknowledge_live,
            trusted_local_mode=os.getenv("RESEARCH_GAP_TRUSTED_LOCAL_MODE", "false").lower() in {"1", "true", "yes"},
        )
        return cls(
            openai_api_key=openai_api_key(), openai_model=openai_model(),
            extraction_model=openai_extraction_model(),
            openalex=OpenAlexSettings(
                api_key=os.getenv("OPENALEX_API_KEY") or None,
                mailto=os.getenv("OPENALEX_MAILTO") or None,
                per_route_limit=integer("OPENALEX_CANDIDATE_LIMIT", 20, maximum=100),
                max_candidates=integer("RESEARCH_GAP_MAX_CANDIDATES", 100, maximum=500),
                max_workers=integer("RESEARCH_GAP_RETRIEVAL_WORKERS", 4, maximum=16),
                timeout_seconds=number("OPENALEX_TIMEOUT_SECONDS", 20.0, positive=True),
                max_retries=integer("OPENALEX_MAX_RETRIES", 2, minimum=0),
                retrieval_cache_ttl_seconds=number(
                    "RESEARCH_GAP_RETRIEVAL_CACHE_TTL_SECONDS", 21600, positive=True,
                ),
            ),
            ranking=RankingSettings(
                embedding_model=openai_embedding_model(),
                embedding_batch_size=integer("RESEARCH_GAP_EMBEDDING_BATCH_SIZE", 100),
                lexical_weight=lexical, semantic_weight=semantic,
                constraint_weight=constraint_weight, semantic_fallback=fallback,
            ),
            full_text=FullTextSettings(
                timeout_seconds=number(
                    "RESEARCH_GAP_FULL_TEXT_TIMEOUT_SECONDS", 12.0,
                    positive=True, maximum=120,
                ),
                max_bytes=integer("RESEARCH_GAP_FULL_TEXT_MAX_BYTES", 8_000_000, maximum=25_000_000),
                max_document_chars=integer("RESEARCH_GAP_FULL_TEXT_MAX_DOCUMENT_CHARS", 120_000, maximum=500_000),
                max_section_chars=integer("RESEARCH_GAP_FULL_TEXT_MAX_SECTION_CHARS", 20_000, maximum=100_000),
                max_chunk_chars=integer("RESEARCH_GAP_FULL_TEXT_MAX_CHUNK_CHARS", 8_000, maximum=30_000),
                max_context_chars=integer("RESEARCH_GAP_FULL_TEXT_MAX_CONTEXT_CHARS", 30_000, maximum=100_000),
                max_redirects=integer("RESEARCH_GAP_FULL_TEXT_MAX_REDIRECTS", 3, minimum=0, maximum=10),
                negative_cache_ttl_seconds=number(
                    "RESEARCH_GAP_FULL_TEXT_NEGATIVE_CACHE_TTL_SECONDS", 3600,
                    positive=True, maximum=604800,
                ),
            ),
            evidence_limit=integer("RESEARCH_GAP_EVIDENCE_LIMIT", 10, minimum=0),
            extraction_workers=integer("RESEARCH_GAP_EXTRACTION_WORKERS", 4),
            extraction_batch_size=integer("RESEARCH_GAP_EXTRACTION_BATCH_SIZE", 3),
            cache_directory=cache_dir(),
            database_url=database_url(),
            analysis_database_path=database_path(),
            max_analysis_workers=integer("RESEARCH_GAP_MAX_ANALYSIS_WORKERS", 2, maximum=8),
            web=web,
        )
