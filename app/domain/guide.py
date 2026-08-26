"""Composition of the patient-facing guide, with provenance on every line.

Two kinds of text reach a patient, and both carry a
:class:`Provenance` tag naming where they came from:

* **Guide sentences** are written by RxPlain from verified fields. The wording is
  ours; every *fact* inside is substituted from a verified value. A sentence about
  a name is tagged ``rxnorm``; one about a dose or frequency is tagged
  ``ocr_span`` and names the span it came from.
* **Label quotes** are passages of an openFDA drug label reproduced verbatim,
  tagged ``openfda`` with the section they came from. They are never reworded.
  Rewording manufacturer safety text risks changing its meaning, so it is shown
  as-is, attributed, behind a disclosure.

Post-validation
---------------
:func:`validate_provenance` re-reads the finished guide as an adversary would.
Any line containing something dose-shaped - a number next to a unit - must:

1. be tagged ``ocr_span`` or ``openfda`` (a ``rxnorm`` tag cannot justify a dose),
   and
2. actually contain that dose in the source text it cites.

Check 2 is the one that matters. A tag alone proves nothing: it is a claim about
provenance, and this compares the claim against the cited text. A sentence that
says "500 MG" while citing a span reading "50 MG" fails here even though it is
tagged correctly.

A guide that fails is composed once more and re-checked. If it fails again it is
discarded and the patient is shown nothing for that document, which is the
required safe failure. With the deterministic composer shipped here the second
attempt is identical to the first, so a failure is really a permanent one; the
retry exists for a generative composer, whose second attempt may differ.
"""

from __future__ import annotations

import re
from typing import Final, Literal

import structlog
from pydantic import BaseModel, ConfigDict, Field

from app.adapters.openfda import OpenFdaClient, OpenFdaLabel
from app.domain.medication import MatchTier
from app.safety.verifier import PatientGuide, UnverifiedItem, VerifiedMedication

_LOGGER = structlog.stdlib.get_logger("app.domain.guide")

_STRICT = ConfigDict(strict=True, frozen=True, extra="forbid")

MAX_COMPOSE_ATTEMPTS: Final = 2

ProvenanceSource = Literal["ocr_span", "rxnorm", "openfda"]

# Anything shaped like a strength. Deliberately broad: over-matching costs an
# extra provenance check, under-matching would let a dose through unchecked.
_DOSE_PATTERN = re.compile(
    r"\b\d+(?:[.,]\d+)?\s*"
    r"(?:mg|mcg|ug|g|gm|kg|ml|l|units?|iu|meq|mmol|%|tablets?|capsules?|hours?|hrs?|times?|days?)\b",
    re.IGNORECASE,
)

# Leading section headings openFDA embeds in the section body, e.g.
# "1 INDICATIONS AND USAGE Metformin hydrochloride tablets are...". Trimming the
# heading is presentational; the untrimmed text stays available in the quote.
_LABEL_HEADING = re.compile(
    r"^\s*\d*(?:\.\d+)?\s*"
    r"(?:INDICATIONS\s*(?:AND|&)\s*USAGE"
    r"|INFORMATION\s+FOR\s+PATIENTS"
    r"|WARNINGS\s+AND\s+(?:CAUTIONS|PRECAUTIONS)"
    r"|BOXED\s+WARNING"
    r"|WARNING)\s*[:.]?\s*",
    re.IGNORECASE,
)


class Provenance(BaseModel):
    """Machine-checkable origin of one piece of patient-facing text."""

    model_config = _STRICT

    source: ProvenanceSource
    ref: str = Field(min_length=1)


class GuideSentence(BaseModel):
    """One sentence of guidance, and the source that justifies it."""

    model_config = _STRICT

    text: str = Field(min_length=1)
    provenance: Provenance
    source_text: str = Field(min_length=1)
    """Verbatim text of the cited source, shown in the disclosure and checked
    by :func:`validate_provenance`."""


class LabelQuote(BaseModel):
    """A verbatim passage of an openFDA label."""

    model_config = _STRICT

    title: str = Field(min_length=1)
    text: str = Field(min_length=1)
    provenance: Provenance
    source_url: str = Field(min_length=1)


class MedicationGuidance(BaseModel):
    """Everything shown for one verified medication."""

    model_config = _STRICT

    display_name: str = Field(min_length=1)
    rxcui: str = Field(min_length=1)
    match_tier: MatchTier
    prescription_text: str = Field(min_length=1)
    sentences: tuple[GuideSentence, ...] = Field(min_length=1)
    quotes: tuple[LabelQuote, ...]
    label_source_url: str | None = None


class RenderedGuide(BaseModel):
    """A guide that passed provenance validation and may be displayed."""

    model_config = _STRICT

    document_id: str = Field(min_length=1)
    medications: tuple[MedicationGuidance, ...]
    unverified: tuple[UnverifiedItem, ...]
    attempts: int = Field(ge=1, le=MAX_COMPOSE_ATTEMPTS)

    @property
    def all_sentences(self) -> tuple[GuideSentence, ...]:
        """Every composed sentence across every medication."""
        return tuple(
            sentence for medication in self.medications for sentence in medication.sentences
        )

    @property
    def has_content(self) -> bool:
        """True when there is anything to show."""
        return bool(self.medications) or bool(self.unverified)


class GuideRejected(BaseModel):
    """A guide that could not be shown, and why."""

    model_config = _STRICT

    document_id: str = Field(min_length=1)
    reason: Literal["provenance_validation_failed"]
    detail: str = Field(min_length=1)
    attempts: int = Field(ge=1)


class ProvenanceViolation(BaseModel):
    """One line whose dose could not be justified by the source it cites."""

    model_config = _STRICT

    text: str = Field(min_length=1)
    source: ProvenanceSource
    ref: str = Field(min_length=1)
    problem: Literal["source_cannot_justify_dose", "dose_absent_from_source"]
    dose_token: str = Field(min_length=1)


def find_dose_tokens(text: str) -> tuple[str, ...]:
    """Return every dose-shaped token in ``text``."""
    return tuple(match.group(0) for match in _DOSE_PATTERN.finditer(text))


def _normalise_for_containment(text: str) -> str:
    """Collapse case and whitespace so containment is not defeated by layout."""
    return " ".join(text.casefold().split())


def _trim_label_heading(text: str) -> str:
    """Drop a leading section heading openFDA embeds in the section body."""
    return _LABEL_HEADING.sub("", text, count=1).strip()


def validate_provenance(guide: RenderedGuide) -> tuple[ProvenanceViolation, ...]:
    """Check every dose-shaped token against the source its line cites.

    Args:
        guide: The composed guide.

    Returns:
        Every violation found. Empty means the guide may be displayed.
    """
    violations: list[ProvenanceViolation] = []

    lines: list[tuple[str, Provenance, str]] = [
        (sentence.text, sentence.provenance, sentence.source_text)
        for medication in guide.medications
        for sentence in medication.sentences
    ]
    lines.extend(
        (quote.text, quote.provenance, quote.text)
        for medication in guide.medications
        for quote in medication.quotes
    )

    for text, provenance, source_text in lines:
        tokens = find_dose_tokens(text)
        if not tokens:
            continue
        haystack = _normalise_for_containment(source_text)
        for token in tokens:
            if provenance.source not in {"ocr_span", "openfda"}:
                violations.append(
                    ProvenanceViolation(
                        text=text,
                        source=provenance.source,
                        ref=provenance.ref,
                        problem="source_cannot_justify_dose",
                        dose_token=token,
                    )
                )
            elif _normalise_for_containment(token) not in haystack:
                violations.append(
                    ProvenanceViolation(
                        text=text,
                        source=provenance.source,
                        ref=provenance.ref,
                        problem="dose_absent_from_source",
                        dose_token=token,
                    )
                )
    return tuple(violations)


def _compose_medication(
    medication: VerifiedMedication, label: OpenFdaLabel | None
) -> MedicationGuidance:
    """Build the guidance for one verified medication."""
    sentences: list[GuideSentence] = [
        GuideSentence(
            text=f"This medicine is {medication.display_name}.",
            provenance=Provenance(source="rxnorm", ref=medication.rxcui),
            source_text=medication.resolution.matched_str,
        )
    ]
    if medication.dose is not None:
        sentences.append(
            GuideSentence(
                text=f"The strength on your prescription is {medication.dose.text}.",
                provenance=Provenance(source="ocr_span", ref=medication.dose.span.span_id),
                source_text=medication.dose.span.text,
            )
        )
    if medication.frequency is not None:
        sentences.append(
            GuideSentence(
                text=f"Your prescription says to take it {medication.frequency.text}.",
                provenance=Provenance(source="ocr_span", ref=medication.frequency.span.span_id),
                source_text=medication.frequency.span.text,
            )
        )

    quotes: list[LabelQuote] = []
    if label is not None:
        for section in label.sections:
            trimmed = _trim_label_heading(section.text)
            if not trimmed:
                continue
            quotes.append(
                LabelQuote(
                    title=section.title,
                    text=trimmed,
                    provenance=Provenance(source="openfda", ref=section.ref),
                    source_url=label.source_url,
                )
            )

    return MedicationGuidance(
        display_name=medication.display_name,
        rxcui=medication.rxcui,
        match_tier=medication.resolution.match_tier,
        prescription_text=medication.line.line_span.text,
        sentences=tuple(sentences),
        quotes=tuple(quotes),
        label_source_url=label.source_url if label is not None else None,
    )


async def compose(
    guide: PatientGuide,
    *,
    openfda: OpenFdaClient,
) -> RenderedGuide | GuideRejected:
    """Compose and validate the patient-facing guide.

    Args:
        guide: The verified/unverified split from the verifier.
        openfda: Supplies verbatim label text for resolved ingredients.

    Returns:
        A :class:`RenderedGuide` when provenance validation passes, otherwise a
        :class:`GuideRejected` describing the failure.
    """
    labels: dict[str, OpenFdaLabel | None] = {}
    for medication in guide.verified:
        if medication.rxcui not in labels:
            labels[medication.rxcui] = await openfda.fetch_label(
                medication.rxcui, medication.resolution.matched_str
            )

    violations: tuple[ProvenanceViolation, ...] = ()
    for attempt in range(1, MAX_COMPOSE_ATTEMPTS + 1):
        candidate = RenderedGuide(
            document_id=guide.document_id,
            medications=tuple(
                _compose_medication(medication, labels[medication.rxcui])
                for medication in guide.verified
            ),
            unverified=guide.unverified,
            attempts=attempt,
        )
        violations = validate_provenance(candidate)
        if not violations:
            if attempt > 1:
                _LOGGER.info("guide_recomposed", document_id=guide.document_id, attempt=attempt)
            return candidate
        _LOGGER.warning(
            "guide_provenance_violation",
            document_id=guide.document_id,
            attempt=attempt,
            violations=len(violations),
            problems=sorted({violation.problem for violation in violations}),
        )

    return GuideRejected(
        document_id=guide.document_id,
        reason="provenance_validation_failed",
        detail=(
            f"{len(violations)} line(s) contained a dose that its cited source does not support"
        ),
        attempts=MAX_COMPOSE_ATTEMPTS,
    )
