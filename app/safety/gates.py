"""Confidence gates applied to an OCR document before anything is extracted.

Two independent checks, both driven by configuration rather than literals:

* **Document gate** - a document whose mean token confidence falls below
  ``ocr_min_mean_confidence`` is rejected outright. A document with no tokens
  is rejected too: an unreadable page must not silently produce an empty guide.
* **Token gate** - individual tokens below ``ocr_min_token_confidence`` are
  flagged. Flagged tokens do not reject the document, but any medication line
  that depends on one is downgraded to an unverified item downstream.

A rejection is a safe outcome. The gate never repairs, re-scores or interpolates
a confidence value.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_serializer

from app.config import Settings
from app.domain.ingest import OcrDocument

RejectionReason = Literal["no_tokens", "mean_confidence_below_threshold"]

_STRICT = ConfigDict(strict=True, frozen=True, extra="forbid")


class GateReport(BaseModel):
    """Outcome of the confidence gates for one document."""

    model_config = _STRICT

    document_id: str = Field(min_length=1)
    token_count: int = Field(ge=0)
    mean_confidence: float = Field(ge=0.0, le=1.0)
    mean_confidence_threshold: float = Field(ge=0.0, le=1.0)
    token_confidence_threshold: float = Field(ge=0.0, le=1.0)
    low_confidence_token_ids: frozenset[str]
    rejection_reason: RejectionReason | None

    @field_serializer("low_confidence_token_ids")
    def _serialise_flagged_tokens(self, value: frozenset[str]) -> list[str]:
        """Emit flagged tokens in sorted order.

        A ``frozenset`` iterates in hash order, which varies between processes.
        Sorting keeps a stored gate decision reproducible, which is what makes it
        usable as an audit record and comparable in a golden test.
        """
        return sorted(value)

    @property
    def accepted(self) -> bool:
        """True when the document may proceed to extraction."""
        return self.rejection_reason is None

    def is_flagged(self, token_id: str) -> bool:
        """True when ``token_id`` fell below the per-token threshold."""
        return token_id in self.low_confidence_token_ids


def evaluate_document(document: OcrDocument, settings: Settings) -> GateReport:
    """Apply both confidence gates to ``document``.

    Args:
        document: The normalised OCR document.
        settings: Supplies both thresholds.

    Returns:
        A report recording the thresholds used, so a stored decision can be
        re-audited even after configuration changes.
    """
    tokens = document.tokens
    mean_confidence = document.mean_confidence
    low_confidence = frozenset(
        token.token_id for token in tokens if token.confidence < settings.ocr_min_token_confidence
    )

    reason: RejectionReason | None = None
    if not tokens:
        reason = "no_tokens"
    elif mean_confidence < settings.ocr_min_mean_confidence:
        reason = "mean_confidence_below_threshold"

    return GateReport(
        document_id=document.document_id,
        token_count=len(tokens),
        mean_confidence=mean_confidence,
        mean_confidence_threshold=settings.ocr_min_mean_confidence,
        token_confidence_threshold=settings.ocr_min_token_confidence,
        low_confidence_token_ids=low_confidence,
        rejection_reason=reason,
    )
