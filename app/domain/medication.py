"""Medication candidates read off a prescription, and RxNorm resolution results.

A :class:`MedicationLine` holds nothing but verbatim OCR text plus the spans it
came from. Extractors may select and slice, never paraphrase: every string here
must be reproducible from the tokens named in its span.
"""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.domain.ingest import SourceSpan

_STRICT = ConfigDict(strict=True, frozen=True, extra="forbid")

MatchTier = Literal["exact", "normalized", "approximate", "truncated"]
UnresolvedReason = Literal["no_candidate", "below_similarity_threshold", "empty_query"]


class MedicationLine(BaseModel):
    """One candidate medication, with per-field provenance.

    ``dose_text`` and ``frequency_text`` are optional because a prescription may
    omit them. Each is paired with its span: text without a span would be an
    unciteable claim, so the pairing is validated.
    """

    model_config = _STRICT

    page_number: int = Field(ge=1)
    drug_text: str = Field(min_length=1)
    drug_span: SourceSpan
    line_span: SourceSpan
    dose_text: str | None = None
    dose_span: SourceSpan | None = None
    frequency_text: str | None = None
    frequency_span: SourceSpan | None = None

    @model_validator(mode="after")
    def _check_span_pairing(self) -> Self:
        """Every populated text field must carry the span it came from."""
        if (self.dose_text is None) != (self.dose_span is None):
            raise ValueError("dose_text and dose_span must be present together")
        if (self.frequency_text is None) != (self.frequency_span is None):
            raise ValueError("frequency_text and frequency_span must be present together")
        return self

    @property
    def source_spans(self) -> tuple[SourceSpan, ...]:
        """Every span backing this line, drug first."""
        spans = [self.drug_span]
        if self.dose_span is not None:
            spans.append(self.dose_span)
        if self.frequency_span is not None:
            spans.append(self.frequency_span)
        return tuple(spans)


class ResolvedConcept(BaseModel):
    """A drug string that matched a real RxNorm concept."""

    model_config = _STRICT

    query: str = Field(min_length=1)
    rxcui: str = Field(min_length=1, pattern=r"^\d+$")
    tty: str = Field(min_length=1)
    matched_str: str = Field(min_length=1)
    match_tier: MatchTier
    similarity: float = Field(ge=0.0, le=1.0)


class UnresolvedConcept(BaseModel):
    """A drug string that RxNorm could not confirm.

    Carries why, so the interface can distinguish "we found nothing" from "we
    found something too dissimilar to trust" when telling a patient to ask a
    pharmacist.
    """

    model_config = _STRICT

    query: str
    reason: UnresolvedReason
    best_similarity: float = Field(ge=0.0, le=1.0)


RxNormResolution = ResolvedConcept | UnresolvedConcept
