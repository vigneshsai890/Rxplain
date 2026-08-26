"""FastAPI application factory and ASGI entrypoint for RxPlain."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import structlog
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict

from app.adapters.extract import RuleBasedExtractor
from app.adapters.ocr import DeterministicFakeOcr, OcrClient
from app.adapters.openfda import NullOpenFdaClient, OpenFdaClient, PostgresCachedOpenFdaClient
from app.adapters.rxnorm import PostgresRxNormResolver
from app.adapters.rxnorm_seed import seed_if_empty
from app.adapters.vision import NvidiaVisionOcr
from app.config import AppEnv, Settings, get_settings
from app.db import Database
from app.web.logging import configure_logging
from app.web.middleware import RequestIdMiddleware, SecurityHeadersMiddleware
from app.web.routes import Services, router

_LOGGER = structlog.stdlib.get_logger("app.main")

STATIC_DIR = Path(__file__).resolve().parent / "web" / "static"


class HealthResponse(BaseModel):
    """Liveness payload returned by ``GET /healthz``."""

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    status: Literal["ok"]
    environment: AppEnv


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build a configured FastAPI application.

    Args:
        settings: Configuration to use. Defaults to the process singleton.

    Returns:
        An application with logging, middleware and routes wired.
    """
    resolved = settings if settings is not None else get_settings()
    configure_logging(resolved.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        """Open the database, build the services, and tear them down after."""
        database = Database(resolved)
        await database.open()
        await database.apply_migrations()
        # A container with an empty database would resolve nothing and report
        # every drug as unverified, so the committed seed is loaded on first run.
        await seed_if_empty(database, resolved.rxnorm_seed_path)
        openfda: OpenFdaClient = (
            PostgresCachedOpenFdaClient(database, resolved)
            if resolved.openfda_enabled
            else NullOpenFdaClient()
        )
        # A key switches transcription to the hosted vision model; without one
        # the fixture replayer keeps the app runnable offline and in CI.
        ocr: OcrClient = (
            NvidiaVisionOcr(resolved)
            if resolved.nvidia_api_key is not None
            else DeterministicFakeOcr(resolved.ocr_fixture_dir)
        )
        app.state.services = Services(
            settings=resolved,
            ocr=ocr,
            extractor=RuleBasedExtractor(),
            resolver=PostgresRxNormResolver(database, resolved),
            openfda=openfda,
        )
        _LOGGER.info(
            "startup_complete",
            environment=resolved.app_env,
            openfda_enabled=resolved.openfda_enabled,
            transcription=type(ocr).__name__,
        )
        try:
            yield
        finally:
            await database.close()

    # The interactive docs load Swagger/ReDoc assets from a CDN, which
    # `default-src 'self'` blocks. Serving a permanently broken page would be
    # worse than not serving one, so they are disabled.
    app = FastAPI(
        title="RxPlain",
        version="0.1.0",
        docs_url=None,
        redoc_url=None,
        lifespan=lifespan,
    )

    # add_middleware prepends, so the last registered runs outermost: the
    # correlation ID is bound before any other layer can log.
    app.add_middleware(SecurityHeadersMiddleware, settings=resolved)
    app.add_middleware(RequestIdMiddleware)

    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    app.include_router(router)

    @app.get("/healthz", response_model=HealthResponse, summary="Liveness probe")
    async def healthz() -> HealthResponse:
        """Report that the process is up. Touches no external dependency."""
        return HealthResponse(status="ok", environment=resolved.app_env)

    return app


app = create_app()
