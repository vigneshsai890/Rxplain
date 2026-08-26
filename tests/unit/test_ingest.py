"""Tests for :mod:`app.domain.ingest`."""

from __future__ import annotations

import hashlib

import pytest
from pydantic import ValidationError

from app.adapters.ocr import DeterministicFakeOcr, FixtureNotFoundError
from app.domain.ingest import (
    BoundingBox,
    EmptyUploadError,
    FileTooLargeError,
    MalformedOcrDocumentError,
    OcrDocument,
    OcrPage,
    OcrToken,
    UnsupportedMediaTypeError,
    detect_media_type,
    ingest,
    span_from_tokens,
    validate_upload,
)

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
JPEG_MAGIC = b"\xff\xd8\xff"
PDF_MAGIC = b"%PDF-"


def _box(left: float = 0.1, top: float = 0.1) -> BoundingBox:
    """Return a small valid box anchored at the given corner."""
    return BoundingBox(left=left, top=top, right=left + 0.05, bottom=top + 0.02)


def _token(index: int, text: str, confidence: float, page: int = 1) -> OcrToken:
    """Return a token with a well-formed identifier for ``page``."""
    return OcrToken(
        token_id=f"p{page}-t{index:03d}",
        text=text,
        confidence=confidence,
        box=_box(left=0.1 + index * 0.01),
    )


def _document(*tokens: OcrToken, payload: bytes = PNG_MAGIC + b"body") -> OcrDocument:
    """Return a single-page document whose ID matches ``payload``."""
    return OcrDocument(
        document_id=hashlib.sha256(payload).hexdigest(),
        media_type="image/png",
        pages=(OcrPage(page_number=1, width_px=800, height_px=600, tokens=tokens),),
    )


# --------------------------------------------------------------------------- #
# magic-byte detection
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        (PNG_MAGIC + b"rest", "image/png"),
        (JPEG_MAGIC + b"\xe0rest", "image/jpeg"),
        (PDF_MAGIC + b"1.7", "application/pdf"),
    ],
)
def test_detects_supported_formats(payload: bytes, expected: str) -> None:
    """Each supported format is recognised from its leading bytes."""
    assert detect_media_type(payload) == expected


def test_spoofed_extension_is_rejected() -> None:
    """A ZIP named .png is rejected: only the bytes are consulted."""
    zip_bytes = b"PK\x03\x04" + b"\x00" * 64
    with pytest.raises(UnsupportedMediaTypeError):
        validate_upload(zip_bytes, max_bytes=1024)


def test_html_disguised_as_pdf_is_rejected() -> None:
    """Text content that merely mentions PDF does not pass the magic check."""
    with pytest.raises(UnsupportedMediaTypeError):
        validate_upload(b"<html>%PDF-1.7</html>", max_bytes=1024)


def test_truncated_png_signature_is_rejected() -> None:
    """A partial signature is not a match."""
    with pytest.raises(UnsupportedMediaTypeError):
        validate_upload(PNG_MAGIC[:-1] + b"junk", max_bytes=1024)


# --------------------------------------------------------------------------- #
# size and emptiness
# --------------------------------------------------------------------------- #
def test_oversized_file_is_rejected() -> None:
    """A payload above the cap is refused before any OCR work happens."""
    payload = PNG_MAGIC + b"\x00" * 2048
    with pytest.raises(FileTooLargeError, match="limit is 512"):
        validate_upload(payload, max_bytes=512)


def test_file_exactly_at_the_cap_is_accepted() -> None:
    """The cap is inclusive, so a boundary payload is allowed."""
    payload = PNG_MAGIC + b"\x00" * 8
    assert validate_upload(payload, max_bytes=len(payload)).size_bytes == len(payload)


def test_empty_upload_is_rejected() -> None:
    """Zero bytes is a distinct failure from an unsupported type."""
    with pytest.raises(EmptyUploadError):
        validate_upload(b"", max_bytes=1024)


def test_upload_id_is_the_content_digest() -> None:
    """Identity comes from the bytes, so the same file always has one ID."""
    payload = PNG_MAGIC + b"content"
    uploaded = validate_upload(payload, max_bytes=1024)
    assert uploaded.document_id == hashlib.sha256(payload).hexdigest()
    assert uploaded.media_type == "image/png"


# --------------------------------------------------------------------------- #
# model invariants
# --------------------------------------------------------------------------- #
def test_inverted_bounding_box_is_rejected() -> None:
    """A zero or negative area box would crop the wrong region."""
    with pytest.raises(ValidationError, match="positive width and height"):
        BoundingBox(left=0.5, top=0.5, right=0.5, bottom=0.6)


def test_bounding_box_union_covers_both() -> None:
    """Union is used to build a span box from its tokens."""
    merged = BoundingBox(left=0.1, top=0.1, right=0.2, bottom=0.2).union(
        BoundingBox(left=0.15, top=0.05, right=0.4, bottom=0.18)
    )
    assert (merged.left, merged.top, merged.right, merged.bottom) == (0.1, 0.05, 0.4, 0.2)


def test_duplicate_token_ids_are_rejected() -> None:
    """Provenance refs must be unique or a citation could point at two places."""
    duplicate = _token(1, "METFORMIN", 0.99)
    with pytest.raises(ValidationError, match="unique"):
        _document(duplicate, duplicate)


def test_token_belonging_to_another_page_is_rejected() -> None:
    """A token ID encodes its page; a mismatch means the mapping is corrupt."""
    with pytest.raises(ValidationError, match="does not belong to its page"):
        _document(_token(1, "METFORMIN", 0.99, page=2))


def test_non_sequential_pages_are_rejected() -> None:
    """Page numbers must run 1..n so a page reference is unambiguous."""
    with pytest.raises(ValidationError, match=r"1\.\.n"):
        OcrDocument(
            document_id="a" * 64,
            media_type="image/png",
            pages=(
                OcrPage(page_number=1, width_px=10, height_px=10, tokens=()),
                OcrPage(page_number=3, width_px=10, height_px=10, tokens=()),
            ),
        )


def test_document_with_no_pages_is_rejected() -> None:
    """A document must have at least one page to be a document."""
    with pytest.raises(ValidationError):
        OcrDocument(document_id="a" * 64, media_type="image/png", pages=())


def test_empty_page_is_a_valid_document() -> None:
    """A blank page is structurally valid; the confidence gate decides on it."""
    document = _document()
    assert document.pages[0].tokens == ()
    assert document.mean_confidence == 0.0


def test_models_are_frozen() -> None:
    """Nothing downstream can rewrite a token's confidence."""
    token = _token(1, "METFORMIN", 0.99)
    field = "confidence"
    with pytest.raises(ValidationError):
        setattr(token, field, 0.1)


# --------------------------------------------------------------------------- #
# spans
# --------------------------------------------------------------------------- #
def test_span_text_is_the_token_text() -> None:
    """A span cannot claim wording OCR did not produce."""
    tokens = (_token(1, "500", 0.9), _token(2, "MG", 0.9))
    span = span_from_tokens(1, tokens)
    assert span.text == "500 MG"
    assert span.token_ids == ("p1-t001", "p1-t002")
    assert span.span_id == "p1:p1-t001..p1-t002"


def test_span_box_spans_all_tokens() -> None:
    """The span box is the union of its token boxes, for cropping."""
    tokens = (_token(1, "500", 0.9), _token(5, "MG", 0.9))
    span = span_from_tokens(1, tokens)
    assert span.box.left == pytest.approx(tokens[0].box.left)
    assert span.box.right == pytest.approx(tokens[1].box.right)


def test_span_requires_at_least_one_token() -> None:
    """An empty span would be an untraceable citation."""
    with pytest.raises(ValueError, match="at least one token"):
        span_from_tokens(1, ())


# --------------------------------------------------------------------------- #
# pipeline
# --------------------------------------------------------------------------- #
@pytest.mark.anyio
async def test_ingest_returns_the_fixture_document(
    fake_ocr: DeterministicFakeOcr, prescription_bytes: dict[str, bytes]
) -> None:
    """The committed image flows through validation into its OCR document."""
    payload = prescription_bytes["clean_three_drugs"]
    document = await ingest(payload, ocr=fake_ocr, max_bytes=10 * 1024 * 1024)
    assert document.document_id == hashlib.sha256(payload).hexdigest()
    assert document.media_type == "image/png"
    assert len(document.tokens) == 37


@pytest.mark.anyio
async def test_ingest_rejects_unknown_document(fake_ocr: DeterministicFakeOcr) -> None:
    """An image with no fixture fails loudly rather than returning nothing."""
    with pytest.raises(FixtureNotFoundError):
        await ingest(PNG_MAGIC + b"unseen", ocr=fake_ocr, max_bytes=1024)


@pytest.mark.anyio
async def test_ingest_rejects_mismatched_document_id(
    prescription_bytes: dict[str, bytes],
) -> None:
    """An adapter cannot swap in a document belonging to different bytes."""

    class _WrongDocumentOcr:
        """Returns a document whose ID does not match the upload."""

        async def analyze(self, uploaded: object, payload: bytes) -> OcrDocument:
            """Ignore the upload and return an unrelated document."""
            del uploaded, payload
            return _document(_token(1, "METFORMIN", 0.99), payload=b"different bytes")

    with pytest.raises(MalformedOcrDocumentError, match="document ID"):
        await ingest(
            prescription_bytes["clean_three_drugs"],
            ocr=_WrongDocumentOcr(),
            max_bytes=10 * 1024 * 1024,
        )
