"""FastAPI application factory and ASGI entrypoint for RxPlain."""

from __future__ import annotations

from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict

from app.config import AppEnv, Settings, get_settings
from app.web.logging import configure_logging
from app.web.middleware import RequestIdMiddleware, SecurityHeadersMiddleware


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
        An application with logging and middleware already wired.
    """
    resolved = settings if settings is not None else get_settings()
    configure_logging(resolved.log_level)

    # The interactive docs load Swagger/ReDoc assets from a CDN, which
    # `default-src 'self'` blocks. Serving a permanently broken page would be
    # worse than not serving one, so they are disabled.
    app = FastAPI(
        title="RxPlain",
        version="0.1.0",
        docs_url=None,
        redoc_url=None,
    )

    # add_middleware prepends, so the last registered runs outermost: the
    # correlation ID is bound before any other layer can log.
    app.add_middleware(SecurityHeadersMiddleware, settings=resolved)
    app.add_middleware(RequestIdMiddleware)

    @app.get("/healthz", response_model=HealthResponse, summary="Liveness probe")
    async def healthz() -> HealthResponse:
        """Report that the process is up. Touches no external dependency."""
        return HealthResponse(status="ok", environment=resolved.app_env)

    return app


app = create_app()
