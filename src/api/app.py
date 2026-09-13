"""FastAPI application factory and local application instance."""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Callable

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes.analyses import router as analyses_router
from src.api.routes.health import router as health_router
from src.api.routes.account import router as account_router
from src.api.routes.admin import router as admin_router
from src.api.routes.billing import router as billing_router
from src.auth import AuthProvider, SupabaseAuthProvider, sign_guest_id
from src.billing import BillingProvider, StripeProvider
from src.application.analysis_service import AnalysisService
from src.application.jobs import AnalysisJobRunner
from src.config import Settings
from src.persistence.database import Database
from src.persistence.models import AnalysisRecord
from src.persistence.repositories import AnalysisRepository
from src.persistence.security import SecurityRepository
from src.storage import AvatarStorage, SupabaseAvatarStorage
from src.operations import OperationsRepository, ProviderUsage
from src.operations.logging import (
    configure_structured_logging, request_id_context, safe_category,
)
from uuid import uuid4
import logging
import re
import time

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ApiComponents:
    settings: Settings
    database: Database
    repository: AnalysisRepository
    service: AnalysisService
    runner: AnalysisJobRunner
    security: SecurityRepository
    auth_provider: AuthProvider | None
    billing_provider: BillingProvider | None
    avatar_storage: AvatarStorage | None
    trusted_local_mode: bool
    operations: OperationsRepository


def create_app(
    *,
    settings: Settings | None = None,
    analysis_executor: Callable[[AnalysisRecord], dict[str, object]] | None = None,
    auth_provider: AuthProvider | None = None,
    billing_provider: BillingProvider | None = None,
    avatar_storage: AvatarStorage | None = None,
) -> FastAPI:
    runtime_settings = settings or Settings.from_env()
    configure_structured_logging()
    database = Database(
        runtime_settings.analysis_database_path,
        url=runtime_settings.database_url,
    )
    repository = AnalysisRepository(database)
    web_settings = runtime_settings.web
    security = SecurityRepository(
        database, free_credits=web_settings.free_lifetime_credits,
        lifetime_credit_hmac_secret=web_settings.lifetime_credit_hmac_secret,
    )
    service = AnalysisService(runtime_settings)
    operation_settings = runtime_settings.operations
    operations = OperationsRepository(
        database,
        daily_budget_usd=operation_settings.daily_provider_budget_usd,
        reservation_usd=operation_settings.provider_budget_reservation_usd,
        input_per_million_usd=operation_settings.openai_input_per_million_usd,
        output_per_million_usd=operation_settings.openai_output_per_million_usd,
    )

    configured_auth = auth_provider
    if configured_auth is None and web_settings.auth_url:
        configured_auth = SupabaseAuthProvider(
            web_settings.auth_url, audience=web_settings.auth_jwt_audience,
            issuer=web_settings.auth_jwt_issuer,
            service_role_key=web_settings.auth_service_role_key,
        )
    configured_billing = billing_provider
    if configured_billing is None and web_settings.stripe_secret_key:
        configured_billing = StripeProvider(
            web_settings.stripe_secret_key, web_settings.stripe_webhook_secret,
        )
    configured_storage = avatar_storage
    if configured_storage is None and web_settings.auth_url and web_settings.auth_service_role_key:
        configured_storage = SupabaseAvatarStorage(web_settings.auth_url, web_settings.auth_service_role_key)

    def execute(record: AnalysisRecord) -> dict[str, object]:
        if analysis_executor is not None:
            payload = analysis_executor(record)
            operations.settle_budget(
                record.analysis_id, operation_settings.provider_budget_reservation_usd,
            )
            return payload
        progress = lambda stage, details=None: repository.update_stage(record.analysis_id, stage, details)
        payload = service.run(
            record.research_idea,
            decomposer=record.decomposer,
            query_generator=record.query_generator,
            paper_limit=record.paper_limit,
            full_text=record.configuration.get("full_text") is True,
            mode=record.mode,
            progress=progress,
        )
        raw_usage = payload.pop("provider_usage", [])
        usage = [ProviderUsage.model_validate(item, strict=False) for item in raw_usage if isinstance(item, dict)]
        actual_cost = operations.record_usage(record.analysis_id, usage)
        operations.settle_budget(record.analysis_id, actual_cost)
        LOGGER.info(
            "provider usage recorded requests=%d", len(usage),
            extra={"stage": "provider_accounting", "outcome": "completed",
                   "provider": "openai", "cache_status": "provider_calls_only",
                   "usage": {"records": len(usage), "estimated_cost_usd": actual_cost}},
        )
        return payload

    runner = AnalysisJobRunner(
        repository,
        execute,
        max_workers=runtime_settings.max_analysis_workers,
    )

    def settle_reserved_credit(analysis_id: str) -> bool:
        record = repository.get(analysis_id)
        if record is None or record.reservation_id is None:
            return False
        return security.settle_credit(analysis_id)

    def release_reserved_credit(analysis_id: str) -> bool:
        operations.release_budget(analysis_id)
        try:
            operations.record_failure(
                request_id=None, analysis_id=analysis_id,
                stage="analysis_job", category="analysis_job_failure",
            )
        except Exception:
            pass
        record = repository.get(analysis_id)
        if record is None or record.reservation_id is None:
            return False
        return security.release_credit(analysis_id)

    runner.on_success = settle_reserved_credit
    runner.on_failure = release_reserved_credit
    components = ApiComponents(
        runtime_settings, database, repository, service, runner, security,
        configured_auth, configured_billing, configured_storage,
        web_settings.trusted_local_mode or (analysis_executor is not None and auth_provider is None),
        operations,
    )

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        if runtime_settings.operations.auto_migrate:
            database.migrate()
        repository.cleanup_expired_guests()
        runner.recover()
        yield
        runner.shutdown()

    application = FastAPI(title="Research GAP", version="10.0", lifespan=lifespan)
    application.state.components = components
    application.add_middleware(
        CORSMiddleware, allow_origins=list(web_settings.allowed_origins), allow_credentials=True,
        allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "Stripe-Signature", "X-CSRF-Token", "X-Request-ID"],
        expose_headers=["X-Request-ID"],
    )

    @application.middleware("http")
    async def request_context(request: Request, call_next):
        supplied = request.headers.get("x-request-id", "")
        request_id = supplied if re.fullmatch(r"[A-Za-z0-9._-]{1,80}", supplied) else str(uuid4())
        request.state.request_id = request_id
        token = request_id_context.set(request_id)
        started = time.perf_counter()
        try:
            response = await call_next(request)
            response.headers["X-Request-ID"] = request_id
            LOGGER.info(
                "request completed method=%s path=%s status=%s user=%s",
                request.method, request.url.path, response.status_code,
                getattr(request.state, "safe_user_id", None),
                extra={"safe_user_id": getattr(request.state, "safe_user_id", None),
                       "stage": "http_request", "duration_ms": round((time.perf_counter()-started)*1000, 1), "outcome": "completed"},
            )
            return response
        finally:
            request_id_context.reset(token)

    @application.exception_handler(Exception)
    async def unexpected_error(request: Request, error: Exception):
        request_id = getattr(request.state, "request_id", str(uuid4()))
        LOGGER.error("unexpected API error category=%s", safe_category(error), extra={"outcome": "failed"})
        try:
            operations.record_failure(request_id=request_id, analysis_id=None, stage="api", category=safe_category(error))
        except Exception:
            pass
        return JSONResponse(status_code=500, content={
            "detail": "An unexpected error occurred.", "request_id": request_id,
        }, headers={"X-Request-ID": request_id})

    @application.middleware("http")
    async def security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        response.headers["Content-Security-Policy"] = "default-src 'none'; frame-ancestors 'none'"
        guest_id = getattr(request.state, "new_guest_id", None)
        if guest_id:
            response.set_cookie(
                "research_gap_guest", sign_guest_id(guest_id, web_settings.guest_cookie_secret),
                httponly=True, secure=web_settings.secure_cookies, samesite="lax",
                max_age=web_settings.guest_retention_hours * 3600, path="/",
            )
        return response

    application.include_router(health_router)
    application.include_router(analyses_router)
    application.include_router(account_router)
    application.include_router(billing_router)
    application.include_router(admin_router)
    return application


app = create_app()
