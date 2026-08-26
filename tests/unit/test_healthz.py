"""Tests for the ``GET /healthz`` liveness endpoint."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import HealthResponse, create_app


def test_healthz_returns_ok(client: TestClient) -> None:
    """The probe succeeds and reports the active environment."""
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "environment": "test"}


def test_healthz_payload_matches_schema(client: TestClient) -> None:
    """The response body round-trips through the strict response model."""
    payload = HealthResponse.model_validate(client.get("/healthz").json())
    assert payload.status == "ok"
    assert payload.environment == "test"


def test_healthz_content_type_is_json(client: TestClient) -> None:
    """The probe advertises JSON so proxies do not sniff the body."""
    assert client.get("/healthz").headers["content-type"] == "application/json"


def test_healthz_reflects_configured_environment() -> None:
    """The reported environment is read from settings, not hardcoded."""
    settings = Settings(app_env="staging", log_level="ERROR")
    with TestClient(create_app(settings)) as production_client:
        assert production_client.get("/healthz").json()["environment"] == "staging"


def test_unknown_path_is_404(client: TestClient) -> None:
    """No catch-all route exists in phase 1."""
    assert client.get("/no-such-route").status_code == 404
