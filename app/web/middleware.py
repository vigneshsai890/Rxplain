"""HTTP middleware: request correlation IDs and hardened response headers."""

from __future__ import annotations

import re
import time
import uuid

import structlog
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp

from app.config import Settings

REQUEST_ID_HEADER = "x-request-id"
"""Header used to accept and echo the correlation ID."""

UNMATCHED_ROUTE = "<unmatched>"
"""Logged in place of a raw path when no route matched (may hold user input)."""

# Deliberately narrow: an inbound ID is reflected into a response header and
# into logs, so anything with control characters, whitespace or excess length
# is discarded in favour of a fresh UUID.
_REQUEST_ID_PATTERN = re.compile(r"[A-Za-z0-9._:-]{8,128}")

_LOGGER = structlog.stdlib.get_logger("app.web.request")


def _coerce_request_id(raw: str | None) -> str:
    """Return a safe correlation ID, minting a UUID4 when input is unusable."""
    if raw is not None and _REQUEST_ID_PATTERN.fullmatch(raw) is not None:
        return raw
    return str(uuid.uuid4())


def _route_template(request: Request) -> str:
    """Return the matched route's path template, never raw user input.

    Logging ``request.url.path`` would leak whatever a client sent, which on a
    404 is arbitrary text. The route template is a fixed string from our own
    source, so it is safe to record.
    """
    route_path: object = getattr(request.scope.get("route"), "path", None)
    if isinstance(route_path, str):
        return route_path
    return UNMATCHED_ROUTE


class RequestIdMiddleware(BaseHTTPMiddleware):
    """Bind a correlation ID to the log context and echo it to the client."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Wrap one request in a fresh, request-scoped logging context."""
        structlog.contextvars.clear_contextvars()
        request_id = _coerce_request_id(request.headers.get(REQUEST_ID_HEADER))
        structlog.contextvars.bind_contextvars(request_id=request_id)
        started = time.perf_counter()
        try:
            response = await call_next(request)
            response.headers[REQUEST_ID_HEADER] = request_id
            _LOGGER.info(
                "http_request",
                method=request.method,
                route=_route_template(request),
                status_code=response.status_code,
                duration_ms=round((time.perf_counter() - started) * 1000, 3),
            )
            return response
        finally:
            structlog.contextvars.clear_contextvars()


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Attach a fixed set of hardening headers to every response."""

    def __init__(self, app: ASGIApp, settings: Settings) -> None:
        """Precompute the header set from validated settings.

        Args:
            app: The downstream ASGI application.
            settings: Validated runtime configuration.
        """
        super().__init__(app)
        self._headers: dict[str, str] = {
            "content-security-policy": settings.content_security_policy,
            "referrer-policy": settings.referrer_policy,
            "strict-transport-security": (
                f"max-age={settings.hsts_max_age_seconds}; includeSubDomains"
            ),
            "x-content-type-options": "nosniff",
            "x-frame-options": "DENY",
        }

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Apply the hardening headers to the downstream response."""
        response = await call_next(request)
        for name, value in self._headers.items():
            response.headers[name] = value
        return response
