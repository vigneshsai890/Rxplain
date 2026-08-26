"""The verification gate. Nothing reaches a patient without passing it.

A candidate :class:`~app.domain.medication.MedicationLine` becomes a
:class:`VerifiedMedication` only when all of the following hold:

1. **No low-confidence tokens.** Every token backing the line scored at or above
   ``ocr_min_token_confidence``. One doubtful character in a strength is enough
   to disqualify the whole medication.
2. **RxNorm resolved.** The drug text matched a real RxNorm concept. The result
   type is :class:`~app.domain.medication.ResolvedConcept`, so an unresolved drug
   is not merely rejected at runtime - it will not type-check.
3. **Dose and frequency are verbatim-traceable.** Each populated field is
   compared against the OCR span it claims to come from, and must score at least
   ``verbatim_min_similarity``. Numbers and dose units are additionally compared
   exactly - see :func:`verbatim_similarity` for why a fuzzy ratio alone is not
   safe for a dose.

Anything that fails becomes an :class:`UnverifiedItem`, which the interface
renders as "We couldn't verify this - ask your pharmacist". A wrong answer is
worse than no answer, so there is no partial credit and no best guess.

How the type system carries the guarantee
----------------------------------------
:class:`VerifiedMedication` declares ``resolution: ResolvedConcept``, and
``ResolvedConcept`` is a distinct type from ``UnresolvedConcept``. Because
:class:`PatientGuide` accepts only ``tuple[VerifiedMedication, ...]``, there is
no assignment that puts an unresolved drug inside a guide which passes
``mypy --strict``. :class:`TracedText` carries the same idea for text: its
``similarity`` field has a hard floor of 0.90, so an untraceable string cannot be
constructed at all, whatever the caller intended.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Literal, Self

import structlog
from pydantic import BaseModel, ConfigDict, Field, model_validator
from rapidfuzz import fuzz

from app.adapters.rxnorm import RxNormResolver
from app.config import Settings
from app.domain.ingest import SourceSpan
from app.domain.medication import MedicationLine, ResolvedConcept
from app.safety.gates import GateReport

_LOGGER = structlog.stdlib.get_logger("app.safety.verifier")

_STRICT = ConfigDict(strict=True, frozen=True, extra="forbid")

# Hard floor for verbatim traceability, independent of configuration. Settings
# validation refuses anything below this, so the two can only agree or be
# stricter - never looser.
ABSOLUTE_MIN_SIMILARITY = 0.90

UnverifiedReason = Literal[
    "low_confidence_tokens",
    "rxnorm_unresolved",
    "dose_not_traceable",
    "frequency_not_traceable",
]

ASK_PHARMACIST_MESSAGE = "We couldn't verify this - ask your pharmacist"

# Numbers, and the unit words a number qualifies. These are the characters where
# a dose error lives, and a fuzzy ratio cannot see them: "50 MG" against
# "500 MG" scores 0.91, and "500 MG" against "500 MCG" scores 0.92, so both a
# ten-fold and a thousand-fold error would clear a 0.90 threshold. They are
# therefore compared exactly, before any fuzzy scoring happens.
_DIGIT_RUNS = re.compile(r"\d+(?:[.,]\d+)?")
_WORDS = re.compile(r"[a-z]+")
_CRITICAL_UNITS = frozenset(
    {"mg", "mcg", "ug", "g", "gm", "kg", "ml", "l", "unit", "units", "iu", "meq", "mmol"}
)


def _critical_signature(text: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return the ``(numbers, units)`` that must match character for character."""
    folded = text.casefold()
    numbers = tuple(_DIGIT_RUNS.findall(folded))
    units = tuple(word for word in _WORDS.findall(folded) if word in _CRITICAL_UNITS)
    return numbers, units


def verbatim_similarity(claimed: str, span: SourceSpan) -> float:
    """Score how closely ``claimed`` reproduces the text of ``span``.

    Two checks, in order:

    1. Every number and every dose unit must be identical, in the same order.
       Any difference returns 0.0 outright. This is deliberately stricter than
       the fuzzy threshold: an edit-distance ratio cannot tell a dose apart from
       ten times that dose.
    2. The remaining wording is scored with a fuzzy ratio, which tolerates case,
       spacing and punctuation while still catching paraphrase.

    Args:
        claimed: The text an extractor says it read.
        span: The OCR span it claims to have read it from.

    Returns:
        A ratio in ``[0, 1]``. 1.0 means identical after folding case and
        whitespace; 0.0 means a number or a unit did not match.
    """
    claimed_text = " ".join(claimed.casefold().split())
    span_text = " ".join(span.text.casefold().split())
    if _critical_signature(claimed_text) != _critical_signature(span_text):
        return 0.0
    return fuzz.ratio(claimed_text, span_text) / 100.0


class TracedText(BaseModel):
    """Text proven to be a near-verbatim reproduction of an OCR span.

    Cannot be constructed below :data:`ABSOLUTE_MIN_SIMILARITY`, so holding one
    of these is itself evidence that the text was traced.
    """

    model_config = _STRICT

    text: str = Field(min_length=1)
    span: SourceSpan
    similarity: float = Field(ge=ABSOLUTE_MIN_SIMILARITY, le=1.0)


class VerifiedMedication(BaseModel):
    """A medication that cleared every gate.

    ``display_name`` comes from RxNorm rather than from OCR, so the name shown to
    a patient is a curated term rather than a scan artefact. Its provenance is
    therefore ``rxnorm``, while dose and frequency remain ``ocr_span``.
    """

    model_config = _STRICT

    resolution: ResolvedConcept
    display_name: str = Field(min_length=1)
    line: MedicationLine
    dose: TracedText | None = None
    frequency: TracedText | None = None

    @model_validator(mode="after")
    def _check_display_name_comes_from_rxnorm(self) -> Self:
        """The displayed name must be the string RxNorm matched, not OCR text."""
        if self.display_name != self.resolution.matched_str:
            raise ValueError("display_name must be the RxNorm matched string")
        return self

    @property
    def rxcui(self) -> str:
        """The resolved RxCUI. Present by construction."""
        return self.resolution.rxcui


class UnverifiedItem(BaseModel):
    """Something the pipeline refused to confirm.

    ``raw_text`` is verbatim OCR so a patient can see what was read, but no
    interpretation of it is offered.
    """

    model_config = _STRICT

    raw_text: str = Field(min_length=1)
    line_span: SourceSpan
    reason: UnverifiedReason
    detail: str = Field(min_length=1)

    @property
    def message(self) -> str:
        """The fixed patient-facing wording for an unverified item."""
        return ASK_PHARMACIST_MESSAGE


class PatientGuide(BaseModel):
    """The verified/unverified split for one document.

    ``verified`` is typed ``tuple[VerifiedMedication, ...]``; there is no way to
    place an unresolved drug in it that survives ``mypy --strict``.
    """

    model_config = _STRICT

    document_id: str = Field(min_length=1)
    verified: tuple[VerifiedMedication, ...]
    unverified: tuple[UnverifiedItem, ...]

    @property
    def has_content(self) -> bool:
        """True when there is anything at all to show the patient."""
        return bool(self.verified) or bool(self.unverified)

    @property
    def rxcuis(self) -> tuple[str, ...]:
        """RxCUIs of every verified medication, in order."""
        return tuple(medication.rxcui for medication in self.verified)


def _trace(
    claimed: str | None, span: SourceSpan | None, threshold: float
) -> tuple[TracedText | None, float]:
    """Attempt to trace ``claimed`` to ``span``.

    Returns:
        ``(traced, similarity)``. ``traced`` is ``None`` when the field is absent
        or scores below ``threshold``.
    """
    if claimed is None or span is None:
        return None, 1.0
    similarity = verbatim_similarity(claimed, span)
    if similarity < threshold:
        return None, similarity
    return TracedText(text=claimed, span=span, similarity=similarity), similarity


async def verify_lines(
    lines: Sequence[MedicationLine],
    *,
    resolver: RxNormResolver,
    gate_report: GateReport,
    settings: Settings,
) -> PatientGuide:
    """Sort candidate lines into verified medications and unverified items.

    Args:
        lines: Candidates from an extractor.
        resolver: RxNorm resolver.
        gate_report: Per-token confidence outcome for the same document.
        settings: Supplies ``verbatim_min_similarity``.

    Returns:
        A guide containing only medications that cleared every gate, plus an
        unverified item for each that did not.
    """
    threshold = max(settings.verbatim_min_similarity, ABSOLUTE_MIN_SIMILARITY)
    verified: list[VerifiedMedication] = []
    unverified: list[UnverifiedItem] = []

    for line in lines:
        flagged = tuple(
            token_id for token_id in line.line_span.token_ids if gate_report.is_flagged(token_id)
        )
        if flagged:
            unverified.append(
                UnverifiedItem(
                    raw_text=line.line_span.text,
                    line_span=line.line_span,
                    reason="low_confidence_tokens",
                    detail=f"{len(flagged)} token(s) scanned below the confidence threshold",
                )
            )
            continue

        resolution = await resolver.resolve(line.drug_text)
        if not isinstance(resolution, ResolvedConcept):
            unverified.append(
                UnverifiedItem(
                    raw_text=line.line_span.text,
                    line_span=line.line_span,
                    reason="rxnorm_unresolved",
                    detail=f"no RxNorm match ({resolution.reason})",
                )
            )
            continue

        dose, dose_similarity = _trace(line.dose_text, line.dose_span, threshold)
        if line.dose_text is not None and dose is None:
            unverified.append(
                UnverifiedItem(
                    raw_text=line.line_span.text,
                    line_span=line.line_span,
                    reason="dose_not_traceable",
                    detail=f"dose text scored {dose_similarity:.2f} against its source span",
                )
            )
            continue

        frequency, frequency_similarity = _trace(
            line.frequency_text, line.frequency_span, threshold
        )
        if line.frequency_text is not None and frequency is None:
            unverified.append(
                UnverifiedItem(
                    raw_text=line.line_span.text,
                    line_span=line.line_span,
                    reason="frequency_not_traceable",
                    detail=(
                        f"frequency text scored {frequency_similarity:.2f} against its source span"
                    ),
                )
            )
            continue

        verified.append(
            VerifiedMedication(
                resolution=resolution,
                display_name=resolution.matched_str,
                line=line,
                dose=dose,
                frequency=frequency,
            )
        )

    _LOGGER.info(
        "verification_complete",
        document_id=gate_report.document_id,
        verified=len(verified),
        unverified=len(unverified),
    )
    return PatientGuide(
        document_id=gate_report.document_id,
        verified=tuple(verified),
        unverified=tuple(unverified),
    )
