"""Ingestion: validate an upload by content, then normalise OCR output.

The media type is decided by magic bytes, never by the filename or the
client-supplied content type. A ``.png`` extension on a PDF - or on a ZIP - must
not change how the bytes are handled.

Every model here is frozen and strict. Once a document has been ingested,
nothing downstream can alter a token, a confidence or a bounding box, so the
provenance chain rooted in these tokens cannot be rewritten later.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

if TYPE_CHECKING:
    from app.adapters.ocr import OcrClient

MediaType = Literal["image/png", "image/jpeg", "application/pdf"]

# (signature, media type). Checked against the leading bytes of the payload.
_MAGIC_SIGNATURES: tuple[tuple[bytes, MediaType], ...] = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"%PDF-", "application/pdf"),
)

_STRICT = ConfigDict(strict=True, frozen=True, extra="forbid")


class IngestError(Exception):
    """Base class for rejections raised while validating an upload."""


class EmptyUploadError(IngestError):
    """The upload contained no bytes."""


class FileTooLargeError(IngestError):
    """The upload exceeded the configured size cap."""


class UnsupportedMediaTypeError(IngestError):
    """The leading bytes matched no supported format."""


class MalformedOcrDocumentError(IngestError):
    """The OCR adapter returned a document that violates its invariants."""


class BoundingBox(BaseModel):
    """A rectangle in page-relative coordinates, each side in ``[0, 1]``."""

    model_config = _STRICT

    left: float = Field(ge=0.0, le=1.0)
    top: float = Field(ge=0.0, le=1.0)
    right: float = Field(ge=0.0, le=1.0)
    bottom: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _check_ordering(self) -> Self:
        """Reject inverted rectangles, which would crop the wrong region."""
        if self.right <= self.left or self.bottom <= self.top:
            raise ValueError("bounding box must have positive width and height")
        return self

    def union(self, other: BoundingBox) -> BoundingBox:
        """Return the smallest box containing both rectangles."""
        return BoundingBox(
            left=min(self.left, other.left),
            top=min(self.top, other.top),
            right=max(self.right, other.right),
            bottom=max(self.bottom, other.bottom),
        )


class OcrToken(BaseModel):
    """One recognised word, with the confidence and location it came from."""

    model_config = _STRICT

    token_id: str = Field(min_length=1, pattern=r"^p\d+-t\d+$")
    text: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    box: BoundingBox


class OcrPage(BaseModel):
    """A single page of a document. May legitimately contain no tokens."""

    model_config = _STRICT

    page_number: int = Field(ge=1)
    width_px: int = Field(gt=0)
    height_px: int = Field(gt=0)
    tokens: tuple[OcrToken, ...]


class OcrDocument(BaseModel):
    """Normalised OCR output: the single root of every provenance chain."""

    model_config = _STRICT

    document_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    media_type: MediaType
    pages: tuple[OcrPage, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _check_identifiers(self) -> Self:
        """Enforce sequential page numbers and globally unique token IDs."""
        expected = tuple(range(1, len(self.pages) + 1))
        if tuple(page.page_number for page in self.pages) != expected:
            raise ValueError("page numbers must run 1..n in order")
        token_ids = [token.token_id for page in self.pages for token in page.tokens]
        if len(token_ids) != len(set(token_ids)):
            raise ValueError("token IDs must be unique across the document")
        for page in self.pages:
            for token in page.tokens:
                if not token.token_id.startswith(f"p{page.page_number}-"):
                    raise ValueError(f"token {token.token_id} does not belong to its page")
        return self

    @property
    def tokens(self) -> tuple[OcrToken, ...]:
        """Every token in the document, in page then reading order."""
        return tuple(token for page in self.pages for token in page.tokens)

    @property
    def mean_confidence(self) -> float:
        """Mean token confidence, or ``0.0`` for a document with no tokens."""
        tokens = self.tokens
        if not tokens:
            return 0.0
        return sum(token.confidence for token in tokens) / len(tokens)


class SourceSpan(BaseModel):
    """Verbatim reference from displayed text back to the tokens behind it.

    ``text`` is always the concatenation of the referenced tokens, so a span can
    never claim wording that OCR did not produce.
    """

    model_config = _STRICT

    span_id: str = Field(min_length=1)
    page_number: int = Field(ge=1)
    token_ids: tuple[str, ...] = Field(min_length=1)
    text: str = Field(min_length=1)
    box: BoundingBox


class UploadedFile(BaseModel):
    """A validated upload, identified by the digest of its own bytes."""

    model_config = _STRICT

    document_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    media_type: MediaType
    size_bytes: int = Field(ge=1)


def detect_media_type(payload: bytes) -> MediaType | None:
    """Return the media type implied by the payload's leading bytes."""
    for signature, media_type in _MAGIC_SIGNATURES:
        if payload.startswith(signature):
            return media_type
    return None


def validate_upload(payload: bytes, *, max_bytes: int) -> UploadedFile:
    """Check size and true format of an upload.

    Args:
        payload: The raw uploaded bytes.
        max_bytes: Inclusive upper bound on payload size.

    Returns:
        The validated upload, keyed by the SHA-256 of ``payload``.

    Raises:
        EmptyUploadError: If ``payload`` is empty.
        FileTooLargeError: If ``payload`` exceeds ``max_bytes``.
        UnsupportedMediaTypeError: If the magic bytes match no supported format.
    """
    if not payload:
        raise EmptyUploadError("upload contained no bytes")
    if len(payload) > max_bytes:
        raise FileTooLargeError(f"upload is {len(payload)} bytes, limit is {max_bytes}")
    media_type = detect_media_type(payload)
    if media_type is None:
        raise UnsupportedMediaTypeError("file contents are not a PNG, JPEG or PDF")
    return UploadedFile(
        document_id=hashlib.sha256(payload).hexdigest(),
        media_type=media_type,
        size_bytes=len(payload),
    )


def span_from_tokens(page_number: int, tokens: Sequence[OcrToken]) -> SourceSpan:
    """Build a span covering ``tokens``, deriving its text and box from them.

    Args:
        page_number: The page the tokens were read from.
        tokens: One or more tokens in reading order.

    Returns:
        A span whose text is the space-joined token text.

    Raises:
        ValueError: If ``tokens`` is empty.
    """
    if not tokens:
        raise ValueError("a source span needs at least one token")
    box = tokens[0].box
    for token in tokens[1:]:
        box = box.union(token.box)
    return SourceSpan(
        span_id=f"p{page_number}:{tokens[0].token_id}..{tokens[-1].token_id}",
        page_number=page_number,
        token_ids=tuple(token.token_id for token in tokens),
        text=" ".join(token.text for token in tokens),
        box=box,
    )


async def ingest(payload: bytes, *, ocr: OcrClient, max_bytes: int) -> OcrDocument:
    """Validate an upload and return its normalised OCR document.

    Args:
        payload: The raw uploaded bytes.
        ocr: The OCR client to analyse the document with.
        max_bytes: Inclusive upper bound on payload size.

    Returns:
        The normalised document.

    Raises:
        IngestError: If the upload is rejected, or if the OCR adapter returns a
            document whose identity does not match the bytes that were sent.
    """
    uploaded = validate_upload(payload, max_bytes=max_bytes)
    document = await ocr.analyze(uploaded, payload)
    if document.document_id != uploaded.document_id:
        raise MalformedOcrDocumentError("OCR document ID does not match the uploaded bytes")
    if document.media_type != uploaded.media_type:
        raise MalformedOcrDocumentError("OCR media type does not match the uploaded bytes")
    return document
