"""Tests asserting the hardening headers on every response."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app

EXPECTED_HEADERS = {
    "content-security-policy": "default-src 'self'; img-src 'self' data:; form-action 'self'",
    "referrer-policy": "no-referrer",
    "strict-transport-security": "max-age=63072000; includeSubDomains",
    "x-content-type-options": "nosniff",
    "x-frame-options": "DENY",
}


@pytest.mark.parametrize("path", ["/healthz", "/no-such-route"])
@pytest.mark.parametrize(("header", "value"), sorted(EXPECTED_HEADERS.items()))
def test_header_present_on_every_response(
    client: TestClient, path: str, header: str, value: str
) -> None:
    """Hardening headers apply to success and error responses alike."""
    assert client.get(path).headers[header] == value


def test_no_header_is_missing(client: TestClient) -> None:
    """Guards against a header being dropped from the middleware."""
    headers = client.get("/healthz").headers
    assert set(EXPECTED_HEADERS) <= {name.lower() for name in headers}


def test_configurable_headers_come_from_settings() -> None:
    """CSP, HSTS max-age and Referrer-Policy are configuration, not code."""
    settings = Settings(
        content_security_policy="default-src 'none'",
        hsts_max_age_seconds=120,
        referrer_policy="same-origin",
    )
    with TestClient(create_app(settings)) as custom_client:
        headers = custom_client.get("/healthz").headers
    assert headers["content-security-policy"] == "default-src 'none'"
    assert headers["strict-transport-security"] == "max-age=120; includeSubDomains"
    assert headers["referrer-policy"] == "same-origin"


def test_fixed_headers_are_not_configurable() -> None:
    """XCTO and XFO have no safe alternative value, so they stay constant."""
    with TestClient(create_app(Settings(app_env="test"))) as default_client:
        headers = default_client.get("/healthz").headers
    assert headers["x-content-type-options"] == "nosniff"
    assert headers["x-frame-options"] == "DENY"
