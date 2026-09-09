"""FastAPI application factory and local application instance."""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Callable

from fastapi import FastAPI, Request
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


def create_app(
    *,
    settings: Settings | None = None,
    analysis_executor: Callable[[AnalysisRecord], dict[str, object]] | None = None,
    auth_provider: AuthProvider | None = None,
    billing_provider: BillingProvider | None = None,
    avatar_storage: AvatarStorage | None = None,
) -> FastAPI:
    runtime_settings = settings or Settings.from_env()
    database = Database(runtime_settings.analysis_database_path)
    repository = AnalysisRepository(database)
    web_settings = runtime_settings.web
    security = SecurityRepository(database, free_credits=web_settings.free_lifetime_credits)
    service = AnalysisService(runtime_settings)

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
            return analysis_executor(record)
        progress = lambda stage, details=None: repository.update_stage(record.analysis_id, stage, details)
        return service.run(
            record.research_idea,
            decomposer=record.decomposer,
            query_generator=record.query_generator,
            paper_limit=record.paper_limit,
            full_text=record.configuration.get("full_text") is True,
            mode=record.mode,
            progress=progress,
        )

    runner = AnalysisJobRunner(
        repository,
        execute,
        max_workers=runtime_settings.max_analysis_workers,
    )
    runner.on_success = security.settle_credit
    runner.on_failure = security.release_credit
    components = ApiComponents(
        runtime_settings, database, repository, service, runner, security,
        configured_auth, configured_billing, configured_storage,
        web_settings.trusted_local_mode or (analysis_executor is not None and auth_provider is None),
    )

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        database.migrate()
        repository.cleanup_expired_guests()
        runner.recover()
        yield
        runner.shutdown()

    application = FastAPI(title="Research GAP", version="9.0", lifespan=lifespan)
    application.state.components = components
    application.add_middleware(
        CORSMiddleware, allow_origins=list(web_settings.allowed_origins), allow_credentials=True,
        allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "Stripe-Signature", "X-CSRF-Token"],
    )

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
