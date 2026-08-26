"""Shared pytest fixtures."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.config import Settings, get_settings
from app.main import create_app

SETTINGS_ENV_VARS = (
    "APP_ENV",
    "CONTENT_SECURITY_POLICY",
    "HSTS_MAX_AGE_SECONDS",
    "LOG_LEVEL",
    "PORT",
    "REFERRER_POLICY",
)


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Strip settings env vars so a developer's shell cannot alter results."""
    for name in SETTINGS_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def settings() -> Settings:
    """Fully specified settings, independent of the ambient environment."""
    return Settings(
        app_env="test",
        log_level="INFO",
        port=8080,
        content_security_policy="default-src 'self'",
        hsts_max_age_seconds=63_072_000,
        referrer_policy="no-referrer",
    )


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    """A test client bound to a freshly built application."""
    with TestClient(create_app(settings)) as test_client:
        yield test_client
