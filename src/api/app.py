"""FastAPI application factory and local application instance."""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Callable

from fastapi import FastAPI

from src.api.routes.analyses import router as analyses_router
from src.api.routes.health import router as health_router
from src.application.analysis_service import AnalysisService
from src.application.jobs import AnalysisJobRunner
from src.config import Settings
from src.persistence.database import Database
from src.persistence.models import AnalysisRecord
from src.persistence.repositories import AnalysisRepository


@dataclass(frozen=True, slots=True)
class ApiComponents:
    settings: Settings
    database: Database
    repository: AnalysisRepository
    service: AnalysisService
    runner: AnalysisJobRunner


def create_app(
    *,
    settings: Settings | None = None,
    analysis_executor: Callable[[AnalysisRecord], dict[str, object]] | None = None,
) -> FastAPI:
    runtime_settings = settings or Settings.from_env()
    database = Database(runtime_settings.analysis_database_path)
    repository = AnalysisRepository(database)
    service = AnalysisService(runtime_settings)

    def execute(record: AnalysisRecord) -> dict[str, object]:
        if analysis_executor is not None:
            return analysis_executor(record)
        return service.run(
            record.research_idea,
            decomposer=record.decomposer,
            query_generator=record.query_generator,
            paper_limit=record.paper_limit,
            full_text=record.configuration.get("full_text") is True,
        )

    runner = AnalysisJobRunner(
        repository,
        execute,
        max_workers=runtime_settings.max_analysis_workers,
    )
    components = ApiComponents(runtime_settings, database, repository, service, runner)

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        database.migrate()
        runner.recover()
        yield
        runner.shutdown()

    application = FastAPI(title="Research GAP", version="8.0", lifespan=lifespan)
    application.state.components = components
    application.include_router(health_router)
    application.include_router(analyses_router)
    return application


app = create_app()
