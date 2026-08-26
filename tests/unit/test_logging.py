"""Tests for JSON logging and the request log record."""

from __future__ import annotations

import json
import logging

import pytest
import structlog
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.web.logging import configure_logging
from app.web.middleware import REQUEST_ID_HEADER, UNMATCHED_ROUTE


def _json_lines(raw: str) -> list[dict[str, object]]:
    """Parse captured stdout into log records, failing loudly on non-JSON."""
    records: list[dict[str, object]] = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        parsed = json.loads(line)
        assert isinstance(parsed, dict), f"non-object log line: {line!r}"
        records.append(parsed)
    return records


def _requests(records: list[dict[str, object]]) -> list[dict[str, object]]:
    """Select the ``http_request`` records from a capture."""
    return [record for record in records if record.get("message") == "http_request"]


def test_structlog_emits_json_to_stdout(capsys: pytest.CaptureFixture[str]) -> None:
    """A structlog call produces one parseable JSON object with a timestamp."""
    configure_logging("INFO")
    structlog.stdlib.get_logger("test").info("hello", answer=42)
    records = _json_lines(capsys.readouterr().out)
    assert len(records) == 1
    assert records[0]["message"] == "hello"
    assert records[0]["answer"] == 42
    assert records[0]["level"] == "info"
    assert isinstance(records[0]["timestamp"], str)


def test_stdlib_loggers_are_also_json(capsys: pytest.CaptureFixture[str]) -> None:
    """Third-party libraries using stdlib logging share the JSON handler."""
    configure_logging("INFO")
    logging.getLogger("some.library").warning("legacy message")
    records = _json_lines(capsys.readouterr().out)
    assert records[0]["message"] == "legacy message"
    assert records[0]["logger"] == "some.library"


def test_log_level_is_honoured(capsys: pytest.CaptureFixture[str]) -> None:
    """Records below the configured level are dropped."""
    configure_logging("ERROR")
    logging.getLogger("some.library").info("should not appear")
    assert capsys.readouterr().out == ""


def test_unknown_log_level_raises() -> None:
    """An unmapped level name fails fast rather than defaulting silently."""
    with pytest.raises(KeyError):
        configure_logging("VERBOSE")


@pytest.mark.parametrize("library", ["httpcore", "httpx", "httpx2", "urllib3"])
def test_url_logging_libraries_are_clamped(
    library: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """HTTP clients log full URLs at INFO, which could carry identifiers."""
    configure_logging("INFO")
    logging.getLogger(library).info("HTTP Request: GET /patients/jane-doe")
    assert capsys.readouterr().out == ""


def test_clamped_libraries_still_report_problems(capsys: pytest.CaptureFixture[str]) -> None:
    """Suppression stops at WARNING so real failures remain visible."""
    configure_logging("INFO")
    logging.getLogger("httpx").warning("connection reset")
    records = _json_lines(capsys.readouterr().out)
    assert records[0]["message"] == "connection reset"


def test_uvicorn_access_log_is_silenced(capsys: pytest.CaptureFixture[str]) -> None:
    """The access log prints raw request lines, so it emits nothing at all."""
    configure_logging("INFO")
    logging.getLogger("uvicorn.access").info('1.2.3.4 - "GET /patients/jane-doe HTTP/1.1" 404')
    assert capsys.readouterr().out == ""


def test_uvicorn_error_log_is_json(capsys: pytest.CaptureFixture[str]) -> None:
    """Server lifecycle records still reach the shared JSON handler."""
    configure_logging("INFO")
    logging.getLogger("uvicorn.error").info("Application startup complete.")
    records = _json_lines(capsys.readouterr().out)
    assert records[0]["logger"] == "uvicorn.error"
    assert records[0]["message"] == "Application startup complete."


def test_request_log_carries_the_correlation_id(
    settings: Settings, capsys: pytest.CaptureFixture[str]
) -> None:
    """The logged request_id matches the ID returned to the client."""
    with TestClient(create_app(settings)) as client:
        response = client.get("/healthz")
    records = _requests(_json_lines(capsys.readouterr().out))
    assert len(records) == 1
    assert records[0]["request_id"] == response.headers[REQUEST_ID_HEADER]
    assert records[0]["method"] == "GET"
    assert records[0]["route"] == "/healthz"
    assert records[0]["status_code"] == 200
    assert isinstance(records[0]["duration_ms"], float)


def test_request_log_omits_raw_paths(
    settings: Settings, capsys: pytest.CaptureFixture[str]
) -> None:
    """Unmatched paths are user input, so the template placeholder is logged."""
    identifying_path = "/patients/jane-doe-1982-national-id-000000000"
    with TestClient(create_app(settings)) as client:
        assert client.get(identifying_path).status_code == 404
    captured = capsys.readouterr().out
    records = _requests(_json_lines(captured))
    assert len(records) == 1
    assert records[0]["route"] == UNMATCHED_ROUTE
    assert "jane-doe" not in captured


def test_context_is_not_leaked_between_requests(
    settings: Settings, capsys: pytest.CaptureFixture[str]
) -> None:
    """Each request log has its own ID; context vars are cleared afterwards."""
    with TestClient(create_app(settings)) as client:
        client.get("/healthz")
        client.get("/healthz")
    records = _requests(_json_lines(capsys.readouterr().out))
    assert len({str(record["request_id"]) for record in records}) == 2
    structlog.stdlib.get_logger("test").info("outside")
    trailing = _json_lines(capsys.readouterr().out)
    assert "request_id" not in trailing[0]
