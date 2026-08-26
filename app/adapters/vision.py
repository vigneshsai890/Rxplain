"""Prescription transcription with a hosted vision model (NVIDIA NIM).

This plugs into the same :class:`~app.adapters.ocr.OcrClient` seam as the fixture
replayer, so everything downstream - the confidence gates, RxNorm resolution,
verbatim traceability, provenance tagging - applies unchanged.

What this path gives up, and why it is labelled
-----------------------------------------------
A vision model returns text. It does not return what a real OCR engine returns:
a per-character bounding box and an independently measured confidence. Two
consequences are recorded honestly rather than papered over.

**Confidence is self-reported.** The model is asked to wrap any word it is unsure
of in ``??word??``. Marked words are scored below
``ocr_min_token_confidence`` so the existing per-token gate flags them, which
demotes the affected medication to "ask your pharmacist". This is the model's own
uncertainty signal, not a measurement, and not a number invented here. A
confident-but-wrong transcription will not be caught by it - that is what RxNorm
resolution and dose containment are for.

**Boxes are reading order, not pixels.** Bounding boxes are derived from line and
word position so spans remain well-formed and orderable. They locate text within
the reading order of the page, not at measured coordinates, and are therefore not
used to crop or highlight the image.

PDFs are refused on this path. A vision model takes a raster image, so a PDF
would have to be rendered first; that is not implemented, and sending the raw
bytes produces an opaque API error rather than a useful one.

The transcription prompt forbids correcting, expanding or reordering anything.
That matters most for dose shorthand: expanding "OD" or "HS" is an
interpretation, and both appear on the ISMP list of error-prone abbreviations
("OD" is *once daily* in some regions and *right eye* in others). This adapter
transcribes; it does not interpret.
"""

from __future__ import annotations

import base64
import json
import re
from typing import Final

import httpx2
import structlog

from app.config import Settings
from app.domain.ingest import (
    BoundingBox,
    IngestError,
    OcrDocument,
    OcrPage,
    OcrToken,
    UploadedFile,
)

_LOGGER = structlog.stdlib.get_logger("app.adapters.vision")

# Page geometry assumed when deriving reading-order boxes. Only ratios matter.
_ASSUMED_WIDTH: Final = 1000
_ASSUMED_HEIGHT: Final = 1400

_UNSURE = re.compile(r"\?\?(.+?)\?\?")

TRANSCRIPTION_PROMPT: Final = (
    "You are transcribing a photograph of a medical prescription.\n"
    "\n"
    "Rules you must follow exactly:\n"
    "1. Transcribe every line of text you can see, verbatim, one line per output line.\n"
    "2. Do NOT correct spelling. Do NOT expand abbreviations. Do NOT reorder anything.\n"
    "   Leave shorthand such as OD, BD, TDS, HS, PC, AC, SOS, PRN exactly as written.\n"
    "3. Do NOT add any text that is not visible in the image.\n"
    "4. If you are unsure of a word, wrap that word in double question marks, "
    "like ??word??. Mark every word you are unsure of.\n"
    "5. Output only the transcribed lines. No preamble, no commentary, no markdown.\n"
    "6. If the image is not a prescription or you can read no text, output exactly: "
    "NO_TEXT_FOUND"
)


class VisionTranscriptionError(IngestError):
    """The vision model could not produce a usable transcription."""


class NotAPrescriptionError(VisionTranscriptionError):
    """The model reported that the image contains no readable text."""


class UnsupportedForVisionError(VisionTranscriptionError):
    """The upload is a format the vision model cannot accept.

    Vision models take raster images. A PDF has to be rendered to an image first,
    which this build does not do, so a PDF is refused here with a message the user
    can act on rather than being sent to the model to fail obscurely.
    """


def _tokenise(lines: list[str], *, confident: float, unsure: float) -> tuple[OcrToken, ...]:
    """Turn transcribed lines into tokens with reading-order boxes.

    Args:
        lines: Transcribed lines, possibly containing ``??word??`` markers.
        confident: Confidence assigned to unmarked words.
        unsure: Confidence assigned to marked words.

    Returns:
        Tokens in reading order.
    """
    tokens: list[OcrToken] = []
    row_count = max(len(lines), 1)
    row_height = 1.0 / (row_count + 1)
    index = 0

    for row, raw_line in enumerate(lines):
        words = raw_line.split()
        if not words:
            continue
        top = min(row * row_height, 1.0 - row_height)
        bottom = min(top + row_height * 0.7, 1.0)
        span = 0.9 / max(len(words), 1)
        for column, raw_word in enumerate(words):
            match = _UNSURE.search(raw_word)
            text = _UNSURE.sub(r"\1", raw_word)
            if not text:
                continue
            left = min(0.05 + column * span, 0.99)
            right = min(left + span * 0.9, 1.0)
            if right <= left:
                right = min(left + 0.001, 1.0)
            index += 1
            tokens.append(
                OcrToken(
                    token_id=f"p1-t{index:03d}",
                    text=text,
                    confidence=unsure if match is not None else confident,
                    box=BoundingBox(left=left, top=top, right=right, bottom=bottom),
                )
            )
    return tuple(tokens)


class NvidiaVisionOcr:
    """Transcribes prescriptions with a hosted NVIDIA vision model."""

    def __init__(self, settings: Settings) -> None:
        """Bind the client to its endpoint, model and confidence settings.

        Raises:
            ValueError: If no API key is configured.
        """
        if settings.nvidia_api_key is None:
            raise ValueError("NvidiaVisionOcr requires nvidia_api_key to be set")
        self._key = settings.nvidia_api_key.get_secret_value()
        self._base_url = settings.nvidia_base_url.rstrip("/")
        self._model = settings.nvidia_vision_model
        self._timeout = settings.nvidia_timeout_seconds
        self._confident = settings.vision_confident_token_confidence
        self._unsure = settings.vision_unsure_token_confidence

    async def analyze(self, uploaded: UploadedFile, payload: bytes) -> OcrDocument:
        """Transcribe ``payload`` and return a normalised document.

        Args:
            uploaded: Validated metadata for the payload.
            payload: The raw image bytes.

        Returns:
            The transcribed document, carrying ``uploaded.document_id``.

        Raises:
            VisionTranscriptionError: On a transport failure, an unexpected
                status, or a response with no usable text.
            NotAPrescriptionError: If the model reports no readable text.
        """
        if uploaded.media_type == "application/pdf":
            raise UnsupportedForVisionError(
                "the transcription model accepts photographs, not PDF files"
            )
        text = await self._transcribe(uploaded.media_type, payload)
        if text.strip() == "NO_TEXT_FOUND":
            raise NotAPrescriptionError("the model found no readable text in this image")

        lines = [line for line in (raw.strip() for raw in text.splitlines()) if line]
        tokens = _tokenise(lines, confident=self._confident, unsure=self._unsure)
        if not tokens:
            raise VisionTranscriptionError("the transcription contained no words")

        flagged = sum(1 for token in tokens if token.confidence == self._unsure)
        _LOGGER.info(
            "vision_transcription_complete",
            document_id=uploaded.document_id,
            lines=len(lines),
            tokens=len(tokens),
            self_reported_unsure=flagged,
        )
        return OcrDocument(
            document_id=uploaded.document_id,
            media_type=uploaded.media_type,
            pages=(
                OcrPage(
                    page_number=1,
                    width_px=_ASSUMED_WIDTH,
                    height_px=_ASSUMED_HEIGHT,
                    tokens=tokens,
                ),
            ),
        )

    async def _transcribe(self, media_type: str, payload: bytes) -> str:
        """Call the model and return its raw text output.

        Raises:
            VisionTranscriptionError: On any transport or protocol failure.
        """
        encoded = base64.b64encode(payload).decode("ascii")
        body = {
            "model": self._model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": TRANSCRIPTION_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{media_type};base64,{encoded}"},
                        },
                    ],
                }
            ],
            # Temperature 0 so the same image transcribes the same way twice; a
            # guide that changes between refreshes is not auditable.
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": 1200,
            "stream": False,
        }
        try:
            async with httpx2.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(
                    f"{self._base_url}/chat/completions",
                    json=body,
                    headers={"Authorization": f"Bearer {self._key}"},
                )
        except httpx2.HTTPError as exc:
            _LOGGER.warning("vision_request_failed", error=type(exc).__name__)
            raise VisionTranscriptionError("the transcription service is unavailable") from exc

        if response.status_code != httpx2.codes.OK:
            # The body may echo request content, so only the status is logged.
            _LOGGER.warning("vision_unexpected_status", status_code=response.status_code)
            raise VisionTranscriptionError(
                f"the transcription service returned {response.status_code}"
            )

        return _extract_content(response.text)


def _extract_content(raw: str) -> str:
    """Pull the assistant message out of a chat-completions response.

    Raises:
        VisionTranscriptionError: If the response is not shaped as expected.
    """
    try:
        body = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise VisionTranscriptionError("the transcription response was not JSON") from exc
    if not isinstance(body, dict):
        raise VisionTranscriptionError("the transcription response was not an object")
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        raise VisionTranscriptionError("the transcription response had no choices")
    first = choices[0]
    if not isinstance(first, dict):
        raise VisionTranscriptionError("the transcription choice was not an object")
    message = first.get("message")
    if not isinstance(message, dict):
        raise VisionTranscriptionError("the transcription choice had no message")
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise VisionTranscriptionError("the transcription was empty")
    return content
