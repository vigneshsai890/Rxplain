"""Tests for :mod:`app.safety.gates`."""

from __future__ import annotations

import pytest

from app.config import Settings
from app.domain.ingest import BoundingBox, OcrDocument, OcrPage, OcrToken
from app.safety.gates import evaluate_document


def _token(index: int, confidence: float, page: int = 1) -> OcrToken:
    """Return a token with the given confidence."""
    left = 0.05 + index * 0.01
    return OcrToken(
        token_id=f"p{page}-t{index:03d}",
        text=f"word{index}",
        confidence=confidence,
        box=BoundingBox(left=left, top=0.1, right=left + 0.005, bottom=0.12),
    )


def _document(*confidences: float) -> OcrDocument:
    """Return a one-page document whose tokens carry ``confidences``."""
    tokens = tuple(_token(i + 1, c) for i, c in enumerate(confidences))
    return OcrDocument(
        document_id="b" * 64,
        media_type="image/png",
        pages=(OcrPage(page_number=1, width_px=800, height_px=600, tokens=tokens),),
    )


def test_high_confidence_document_is_accepted(settings: Settings) -> None:
    """A clean scan passes both gates with nothing flagged."""
    report = evaluate_document(_document(0.99, 0.97, 0.95), settings)
    assert report.accepted
    assert report.rejection_reason is None
    assert report.low_confidence_token_ids == frozenset()
    assert report.mean_confidence == pytest.approx(0.97)


def test_low_mean_confidence_is_rejected(settings: Settings) -> None:
    """Mean below the document threshold rejects the whole document."""
    report = evaluate_document(_document(0.70, 0.75, 0.79), settings)
    assert not report.accepted
    assert report.rejection_reason == "mean_confidence_below_threshold"


def test_document_exactly_at_the_threshold_is_accepted(settings: Settings) -> None:
    """The document gate rejects below the threshold, not at it."""
    report = evaluate_document(_document(0.80, 0.80), settings)
    assert report.mean_confidence == pytest.approx(0.80)
    assert report.accepted


def test_empty_page_is_rejected(settings: Settings) -> None:
    """An unreadable page must not silently produce an empty guide."""
    document = OcrDocument(
        document_id="c" * 64,
        media_type="image/png",
        pages=(OcrPage(page_number=1, width_px=800, height_px=600, tokens=()),),
    )
    report = evaluate_document(document, settings)
    assert not report.accepted
    assert report.rejection_reason == "no_tokens"
    assert report.token_count == 0


def test_blank_second_page_does_not_reject_a_good_document(settings: Settings) -> None:
    """Only a document with no tokens at all is rejected for emptiness."""
    document = OcrDocument(
        document_id="d" * 64,
        media_type="image/png",
        pages=(
            OcrPage(
                page_number=1,
                width_px=800,
                height_px=600,
                tokens=(_token(1, 0.99), _token(2, 0.98)),
            ),
            OcrPage(page_number=2, width_px=800, height_px=600, tokens=()),
        ),
    )
    assert evaluate_document(document, settings).accepted


def test_low_confidence_tokens_are_flagged_not_fatal(settings: Settings) -> None:
    """A single bad word flags that token while the document proceeds."""
    report = evaluate_document(_document(0.99, 0.55, 0.98, 0.97), settings)
    assert report.accepted
    assert report.low_confidence_token_ids == frozenset({"p1-t002"})
    assert report.is_flagged("p1-t002")
    assert not report.is_flagged("p1-t001")


def test_token_exactly_at_the_threshold_is_not_flagged(settings: Settings) -> None:
    """The token gate flags below the threshold, not at it."""
    report = evaluate_document(_document(0.99, 0.60, 0.99), settings)
    assert report.low_confidence_token_ids == frozenset()


def test_thresholds_come_from_configuration() -> None:
    """Raising the thresholds changes the verdict with no code change."""
    document = _document(0.90, 0.88, 0.92)
    assert evaluate_document(document, Settings(app_env="test")).accepted
    strict = Settings(
        app_env="test",
        ocr_min_mean_confidence=0.95,
        ocr_min_token_confidence=0.91,
    )
    report = evaluate_document(document, strict)
    assert not report.accepted
    assert report.low_confidence_token_ids == frozenset({"p1-t001", "p1-t002"})


def test_report_records_the_thresholds_it_used(settings: Settings) -> None:
    """A stored decision stays auditable after configuration changes."""
    report = evaluate_document(_document(0.99), settings)
    assert report.mean_confidence_threshold == pytest.approx(0.80)
    assert report.token_confidence_threshold == pytest.approx(0.60)
    assert report.document_id == "b" * 64
