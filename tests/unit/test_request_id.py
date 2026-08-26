"""Tests for the request-ID middleware."""

from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient
from starlette.requests import Request

from app.web.middleware import REQUEST_ID_HEADER, UNMATCHED_ROUTE, _route_template


def test_request_id_is_generated_when_absent(client: TestClient) -> None:
    """A missing header yields a fresh UUID4 in the response."""
    value = client.get("/healthz").headers[REQUEST_ID_HEADER]
    assert uuid.UUID(value).version == 4


def test_request_id_is_echoed_when_valid(client: TestClient) -> None:
    """A well-formed inbound ID is preserved for cross-service correlation."""
    supplied = "trace-0123456789abcdef"
    response = client.get("/healthz", headers={REQUEST_ID_HEADER: supplied})
    assert response.headers[REQUEST_ID_HEADER] == supplied


@pytest.mark.parametrize(
    "supplied",
    [
        "short",  # below the 8-character minimum
        "x" * 129,  # above the 128-character maximum
        "has spaces",  # header injection / log-forging surface
        "semi;colon",
        "<script>alert(1)</script>",
        "",
    ],
)
def test_hostile_request_id_is_replaced(client: TestClient, supplied: str) -> None:
    """Unusable input is discarded rather than reflected back."""
    response = client.get("/healthz", headers={REQUEST_ID_HEADER: supplied})
    returned = response.headers[REQUEST_ID_HEADER]
    assert returned != supplied
    assert uuid.UUID(returned).version == 4


def test_each_request_gets_a_distinct_id(client: TestClient) -> None:
    """IDs are per-request, not per-process."""
    first = client.get("/healthz").headers[REQUEST_ID_HEADER]
    second = client.get("/healthz").headers[REQUEST_ID_HEADER]
    assert first != second


def test_error_responses_also_carry_an_id(client: TestClient) -> None:
    """A 404 is still traceable in the logs."""
    response = client.get("/no-such-route")
    assert response.status_code == 404
    assert uuid.UUID(response.headers[REQUEST_ID_HEADER]).version == 4


def test_route_template_falls_back_for_unmatched_scope() -> None:
    """Without a matched route there is no template, so nothing is logged."""
    request = Request({"type": "http", "method": "GET", "path": "/x", "headers": []})
    assert _route_template(request) == UNMATCHED_ROUTE
