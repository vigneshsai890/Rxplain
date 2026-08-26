"""Structured logging configuration.

The service emits exactly one JSON object per log record on stdout, including
records produced by third-party libraries and by uvicorn, so a log shipper
never has to parse two formats.

No record may contain patient data. Callers log identifiers, route templates
and counts; raw request paths, bodies, headers and OCR text are never passed to
a logger.
"""

from __future__ import annotations

import logging
import sys

import structlog
from structlog.types import Processor

_UVICORN_LOGGERS = ("uvicorn", "uvicorn.error")

# uvicorn's access log prints the raw request line, so a 404 would record
# whatever path a client sent. The request-ID middleware already emits one
# structured record per request using the route template, so this logger is
# silenced outright rather than reformatted.
_ACCESS_LOGGER = "uvicorn.access"

# HTTP client libraries log the full request URL at INFO. A URL can carry
# identifiers, so these are clamped to WARNING: the no-PII rule outranks their
# verbosity. Anything genuinely wrong still surfaces as a warning or error.
_URL_LOGGING_LIBRARIES = ("httpcore", "httpx", "httpx2", "urllib3")


def _shared_processors() -> list[Processor]:
    """Return processors applied to structlog and stdlib records alike."""
    return [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
    ]


def configure_logging(log_level: str) -> None:
    """Install a single JSON stdout handler for the whole process.

    Args:
        log_level: A name accepted by :mod:`logging`, e.g. ``"INFO"``.

    Raises:
        KeyError: If ``log_level`` is not a known level name.
    """
    level = logging.getLevelNamesMapping()[log_level]
    shared = _shared_processors()

    structlog.configure(
        processors=[*shared, structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=False,
    )

    formatter: logging.Formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.processors.dict_tracebacks,
            structlog.processors.EventRenamer("message"),
            structlog.processors.JSONRenderer(sort_keys=True),
        ],
    )
    handler: logging.Handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)

    # uvicorn calls dictConfig() before importing the app and attaches its own
    # plain-text handlers with propagate=False. Undo that so its records reach
    # the JSON handler above.
    for name in _UVICORN_LOGGERS:
        server_logger = logging.getLogger(name)
        server_logger.handlers.clear()
        server_logger.propagate = True
        server_logger.setLevel(level)

    # Enforced here rather than via `uvicorn --no-access-log` so the guarantee
    # holds for every entrypoint, including a bare `uvicorn app.main:app`.
    access_logger = logging.getLogger(_ACCESS_LOGGER)
    access_logger.handlers.clear()
    access_logger.propagate = False

    for name in _URL_LOGGING_LIBRARIES:
        logging.getLogger(name).setLevel(max(level, logging.WARNING))
