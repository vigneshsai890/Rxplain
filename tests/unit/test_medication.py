"""Tests for :mod:`app.domain.medication`.

The pairing rules matter because a populated text field without its span is an
unciteable claim: it would reach the verifier with nothing to check it against.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.domain.ingest import BoundingBox, SourceSpan
from app.domain.medication import MedicationLine, ResolvedConcept, UnresolvedConcept

BOX = BoundingBox(left=0.1, top=0.1, right=0.4, bottom=0.14)


def _span(span_id: str, text: str) -> SourceSpan:
    """Build a span carrying ``text``."""
    return SourceSpan(span_id=span_id, page_number=1, token_ids=("p1-t001",), text=text, box=BOX)


def test_drug_only_line_is_valid() -> None:
    """A prescription may name a drug with no strength or instruction."""
    line = MedicationLine(
        page_number=1,
        drug_text="METFORMIN",
        drug_span=_span("s-drug", "METFORMIN"),
        line_span=_span("s-line", "METFORMIN"),
    )
    assert line.dose_text is None
    assert line.source_spans == (line.drug_span,)


def test_source_spans_lists_every_backing_span() -> None:
    """The provenance list covers each populated field, drug first."""
    line = MedicationLine(
        page_number=1,
        drug_text="METFORMIN",
        drug_span=_span("s-drug", "METFORMIN"),
        dose_text="500 MG",
        dose_span=_span("s-dose", "500 MG"),
        frequency_text="twice daily",
        frequency_span=_span("s-freq", "twice daily"),
        line_span=_span("s-line", "METFORMIN 500 MG twice daily"),
    )
    assert [span.span_id for span in line.source_spans] == ["s-drug", "s-dose", "s-freq"]


def test_dose_text_without_a_span_is_rejected() -> None:
    """An unciteable dose cannot be represented."""
    with pytest.raises(ValidationError, match="dose_text and dose_span"):
        MedicationLine(
            page_number=1,
            drug_text="METFORMIN",
            drug_span=_span("s-drug", "METFORMIN"),
            dose_text="500 MG",
            line_span=_span("s-line", "METFORMIN 500 MG"),
        )


def test_dose_span_without_text_is_rejected() -> None:
    """A span with nothing claimed against it is equally invalid."""
    with pytest.raises(ValidationError, match="dose_text and dose_span"):
        MedicationLine(
            page_number=1,
            drug_text="METFORMIN",
            drug_span=_span("s-drug", "METFORMIN"),
            dose_span=_span("s-dose", "500 MG"),
            line_span=_span("s-line", "METFORMIN 500 MG"),
        )


def test_frequency_text_without_a_span_is_rejected() -> None:
    """The same rule applies to frequency."""
    with pytest.raises(ValidationError, match="frequency_text and frequency_span"):
        MedicationLine(
            page_number=1,
            drug_text="METFORMIN",
            drug_span=_span("s-drug", "METFORMIN"),
            frequency_text="twice daily",
            line_span=_span("s-line", "METFORMIN twice daily"),
        )


def test_frequency_span_without_text_is_rejected() -> None:
    """A frequency span with no claim is invalid."""
    with pytest.raises(ValidationError, match="frequency_text and frequency_span"):
        MedicationLine(
            page_number=1,
            drug_text="METFORMIN",
            drug_span=_span("s-drug", "METFORMIN"),
            frequency_span=_span("s-freq", "twice daily"),
            line_span=_span("s-line", "METFORMIN twice daily"),
        )


@pytest.mark.parametrize("rxcui", ["", "abc", "68O9", "6809x", "-1"])
def test_resolved_concept_requires_a_numeric_rxcui(rxcui: str) -> None:
    """An RxCUI is always digits; anything else is not an RxNorm identifier."""
    with pytest.raises(ValidationError):
        ResolvedConcept(
            query="metformin",
            rxcui=rxcui,
            tty="IN",
            matched_str="metformin",
            match_tier="exact",
            similarity=1.0,
        )


def test_unresolved_concept_accepts_empty_query() -> None:
    """Strength-only text has an empty search key and must still be reportable."""
    unresolved = UnresolvedConcept(query="", reason="empty_query", best_similarity=0.0)
    assert unresolved.reason == "empty_query"
