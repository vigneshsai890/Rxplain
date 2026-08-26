"""Request size limiting, applied before the body is read.

Why this exists as middleware rather than a check in the handler
---------------------------------------------------------------
``app.domain.ingest.validate_upload`` rejects a payload above
``max_upload_bytes``, but by the time it runs the body has already been read.
That bounds what the pipeline *accepts*, not what the process *buffers*: a
multi-gigabyte upload is consumed into memory or a spool file first and only then
refused. The size cap therefore did nothing against resource exhaustion.

This middleware closes that in two ways, because a client controls which applies:

* **``Content-Length`` present** - the header is compared against the cap and an
  oversized request is refused with 413 before the body is touched at all.
* **Chunked, no ``Content-Length``** - the header cannot be trusted to exist, so
  the ASGI ``receive`` callable is wrapped and bytes are counted as they arrive.
  The request is aborted the moment the running total exceeds the cap, so the
  most that is ever buffered is one chunk beyond the limit.

The cap here is deliberately larger than ``max_upload_bytes``: a multipart body
carries boundary and header overhead around the file, so a legitimate upload at
exactly the file limit is slightly bigger on the wire. The overhead allowance
keeps the specific, useful "that file is larger than 10 MB" error coming from the
pipeline, while this layer only stops requests that are abusive rather than
merely too large.
"""

from __future__ import annotations

from typing import Final

import structlog
from starlette.datastructures import Headers
from starlette.responses import HTMLResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

_LOGGER = structlog.stdlib.get_logger("app.web.limits")

# Headroom for multipart framing around the file itself.
MULTIPART_OVERHEAD_BYTES: Final = 64 * 1024

_HTTP_PAYLOAD_TOO_LARGE: Final = 413


class RequestSizeLimitMiddleware:
    """Refuses over-large request bodies before they are buffered."""

    def __init__(self, app: ASGIApp, max_body_bytes: int) -> None:
        """Bind the middleware to a byte ceiling.

        Args:
            app: The downstream ASGI application.
            max_body_bytes: Largest request body accepted, in bytes.
        """
        self._app = app
        self._max_body_bytes = max_body_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Enforce the ceiling for HTTP requests, passing others through."""
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        headers = Headers(scope=scope)
        declared = headers.get("content-length")
        if declared is not None and declared.isdigit() and int(declared) > self._max_body_bytes:
            _LOGGER.info(
                "request_rejected_by_declared_size",
                declared_bytes=int(declared),
                limit_bytes=self._max_body_bytes,
            )
            await self._reject(send)
            return

        received = 0

        async def counting_receive() -> Message:
            """Count body bytes and refuse once the ceiling is passed."""
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                body = message.get("body", b"")
                if isinstance(body, bytes):
                    received += len(body)
                if received > self._max_body_bytes:
                    _LOGGER.info(
                        "request_rejected_by_streamed_size",
                        received_bytes=received,
                        limit_bytes=self._max_body_bytes,
                    )
                    # Presenting the stream as finished stops the parser reading
                    # further while the handler unwinds.
                    return {"type": "http.disconnect"}
            return message

        await self._app(scope, counting_receive, send)

    @staticmethod
    async def _reject(send: Send) -> None:
        """Send a 413 as a full page, without invoking the application."""
        # Imported here: the routes module builds the template environment, and
        # importing it at module scope would make the two modules circular.
        from app.web.routes import render_refusal  # noqa: PLC0415

        response = HTMLResponse(
            render_refusal(
                heading="That upload is too large",
                message=("The file you chose is bigger than RxPlain accepts, so it was not read."),
                steps=[
                    "Take a new photo instead of sending a scan at full size.",
                    "If your phone offers a smaller image size, choose that.",
                    "A clear photo of just the label is enough.",
                ],
            ),
            status_code=_HTTP_PAYLOAD_TOO_LARGE,
        )
        await response({"type": "http"}, _no_body, send)


async def _no_body() -> Message:
    """Receive callable for a response sent without reading a request."""
    return {"type": "http.request", "body": b"", "more_body": False}
