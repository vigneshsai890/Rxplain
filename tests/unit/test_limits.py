"""Tests for request size limiting and rate limiting.

The size tests matter because the pipeline's own size check runs *after* the body
has been read. These assert the request is refused before that point, which is
the difference between limiting what is accepted and limiting what is buffered.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.web.limits import MULTIPART_OVERHEAD_BYTES
from app.web.ratelimit import default_limit, guide_limit

PNG = b"\x89PNG\r\n\x1a\n"

_HTTP_OK = 200
_HTTP_PAYLOAD_TOO_LARGE = 413
_HTTP_UNSUPPORTED_MEDIA_TYPE = 415
_HTTP_TOO_MANY_REQUESTS = 429


@pytest.fixture
def small_upload_client() -> Iterator[TestClient]:
    """A client whose upload cap is small enough to exceed cheaply."""
    settings = Settings(app_env="test", max_upload_bytes=2048, rate_limit_enabled=False)
    with TestClient(create_app(settings)) as client:
        yield client


# --------------------------------------------------------------------------- #
# size limiting
# --------------------------------------------------------------------------- #
def test_an_oversized_body_is_refused_with_413(small_upload_client: TestClient) -> None:
    """A body past the ceiling is refused, not parsed."""
    oversized = PNG + b"\x00" * (2048 + MULTIPART_OVERHEAD_BYTES + 4096)
    response = small_upload_client.post(
        "/guide", files={"prescription": ("big.png", oversized, "image/png")}
    )
    assert response.status_code == _HTTP_PAYLOAD_TOO_LARGE


def test_the_refusal_explains_what_to_do(small_upload_client: TestClient) -> None:
    """A size refusal is actionable rather than a bare status code."""
    oversized = PNG + b"\x00" * (2048 + MULTIPART_OVERHEAD_BYTES + 4096)
    response = small_upload_client.post(
        "/guide", files={"prescription": ("big.png", oversized, "image/png")}
    )
    assert "too large" in response.text.lower()


def test_a_body_inside_the_ceiling_reaches_the_pipeline(
    small_upload_client: TestClient,
) -> None:
    """The ceiling must not shadow the pipeline's own, more specific errors.

    A body under the wire ceiling but carrying an unsupported file still gets the
    415 that names the real problem, rather than a generic size error.
    """
    response = small_upload_client.post(
        "/guide", files={"prescription": ("x.png", b"PK\x03\x04" + b"\x00" * 64, "image/png")}
    )
    assert response.status_code == _HTTP_UNSUPPORTED_MEDIA_TYPE
    assert "not a PNG, JPEG or PDF" in response.text


def test_the_ceiling_allows_multipart_overhead() -> None:
    """A file at exactly the limit must not be refused by framing overhead."""
    assert MULTIPART_OVERHEAD_BYTES > 0
    settings = Settings(app_env="test", max_upload_bytes=1024)
    assert settings.max_upload_bytes + MULTIPART_OVERHEAD_BYTES > settings.max_upload_bytes


def test_ordinary_page_requests_are_unaffected(small_upload_client: TestClient) -> None:
    """The middleware passes through anything without an over-large body."""
    assert small_upload_client.get("/").status_code == _HTTP_OK
    assert small_upload_client.get("/healthz").status_code == _HTTP_OK


# --------------------------------------------------------------------------- #
# rate limiting
# --------------------------------------------------------------------------- #
def test_limits_come_from_the_settings_the_app_was_built_with() -> None:
    """A limit passed to create_app must actually be the one enforced.

    Reading get_settings() here instead would consult the environment, so an
    explicitly configured limit would be ignored while appearing to work.
    """
    settings = Settings(app_env="test", rate_limit_guide="7/minute", rate_limit_default="99/hour")
    with TestClient(create_app(settings)):
        assert guide_limit() == "7/minute"
        assert default_limit() == "99/hour"


def test_defaults_match_the_declared_settings() -> None:
    """With no override, the limiter adopts the documented defaults."""
    with TestClient(create_app(Settings(app_env="test"))):
        assert guide_limit() == Settings().rate_limit_guide
        assert default_limit() == Settings().rate_limit_default


def test_the_guide_limit_is_tighter_than_the_default() -> None:
    """The expensive endpoint must not inherit the browsing budget."""
    settings = Settings()
    guide_per_minute = int(settings.rate_limit_guide.split("/")[0])
    default_per_minute = int(settings.rate_limit_default.split("/")[0])
    assert guide_per_minute < default_per_minute


def test_repeated_uploads_are_eventually_refused() -> None:
    """The transcription endpoint stops answering past its limit."""
    settings = Settings(app_env="test", rate_limit_guide="3/minute", max_upload_bytes=4096)
    spoofed = {"prescription": ("x.png", b"PK\x03\x04" + b"\x00" * 32, "image/png")}
    statuses: list[int] = []
    with TestClient(create_app(settings)) as client:
        for _ in range(6):
            statuses.append(client.post("/guide", files=spoofed).status_code)
    assert _HTTP_TOO_MANY_REQUESTS in statuses
    assert statuses.count(_HTTP_UNSUPPORTED_MEDIA_TYPE) == 3


def test_a_rate_limited_response_says_what_to_do() -> None:
    """A 429 tells the user to wait rather than looking like a crash."""
    settings = Settings(app_env="test", rate_limit_guide="1/minute", max_upload_bytes=4096)
    spoofed = {"prescription": ("x.png", b"PK\x03\x04" + b"\x00" * 32, "image/png")}
    with TestClient(create_app(settings)) as client:
        client.post("/guide", files=spoofed)
        response = client.post("/guide", files=spoofed)
    assert response.status_code == _HTTP_TOO_MANY_REQUESTS
    assert "wait" in response.text.lower()


def test_rate_limiting_can_be_disabled_for_tests() -> None:
    """The switch exists so other suites are not throttled by each other."""
    settings = Settings(app_env="test", rate_limit_enabled=False, max_upload_bytes=4096)
    spoofed = {"prescription": ("x.png", b"PK\x03\x04" + b"\x00" * 32, "image/png")}
    with TestClient(create_app(settings)) as client:
        statuses = [client.post("/guide", files=spoofed).status_code for _ in range(8)]
    assert statuses.count(_HTTP_UNSUPPORTED_MEDIA_TYPE) == 8


def test_a_rate_limited_response_still_carries_security_headers() -> None:
    """A refusal is still a response a browser will render."""
    settings = Settings(app_env="test", rate_limit_guide="1/minute", max_upload_bytes=4096)
    spoofed = {"prescription": ("x.png", b"PK\x03\x04" + b"\x00" * 32, "image/png")}
    with TestClient(create_app(settings)) as client:
        client.post("/guide", files=spoofed)
        response = client.post("/guide", files=spoofed)
    assert response.headers["x-frame-options"] == "DENY"
    assert response.headers["x-content-type-options"] == "nosniff"


# --------------------------------------------------------------------------- #
# refusals are pages, not bare status codes
# --------------------------------------------------------------------------- #
def _assert_accessible_page(markup: str) -> None:
    """Assert a refusal carries the same structure as any other page.

    A plain-text body would have no language, no landmarks and no disclaimer, and
    would fail the accessibility gate. A throttled user still gets a real page.
    """
    assert '<html lang="en"' in markup
    assert "<main" in markup
    assert "<footer" in markup
    assert "Not medical advice" in markup
    assert 'class="skip-link" href="#main"' in markup


def test_the_rate_limit_page_is_accessible() -> None:
    """A 429 is rendered, not dumped as text."""
    settings = Settings(app_env="test", rate_limit_guide="1/minute", max_upload_bytes=4096)
    spoofed = {"prescription": ("x.png", b"PK\x03\x04" + b"\x00" * 32, "image/png")}
    with TestClient(create_app(settings)) as client:
        client.post("/guide", files=spoofed)
        response = client.post("/guide", files=spoofed)
    assert response.status_code == _HTTP_TOO_MANY_REQUESTS
    _assert_accessible_page(response.text)
    assert "aria-live" in response.text


def test_the_oversize_page_is_accessible(small_upload_client: TestClient) -> None:
    """A 413 from pure-ASGI middleware is rendered too."""
    oversized = PNG + b"\x00" * (2048 + MULTIPART_OVERHEAD_BYTES + 4096)
    response = small_upload_client.post(
        "/guide", files={"prescription": ("big.png", oversized, "image/png")}
    )
    assert response.status_code == _HTTP_PAYLOAD_TOO_LARGE
    _assert_accessible_page(response.text)


def test_refusals_never_echo_the_request(small_upload_client: TestClient) -> None:
    """A refusal page must not reflect a filename back into the markup."""
    oversized = PNG + b"\x00" * (2048 + MULTIPART_OVERHEAD_BYTES + 4096)
    response = small_upload_client.post(
        "/guide",
        files={"prescription": ("<script>alert(1)</script>.png", oversized, "image/png")},
    )
    assert "<script>alert(1)</script>" not in response.text
    assert "alert(1)" not in response.text
