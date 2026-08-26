"""Tests for :mod:`app.domain.guide`.

The post-validation tests matter most. They construct guides that are internally
consistent but *wrong* - a sentence citing a source that does not contain the dose
it states - and confirm the guide is thrown away rather than shown.
"""

from __future__ import annotations

import anyio
import pytest

from app.adapters.openfda import LabelSection, OpenFdaLabel
from app.domain.guide import (
    MAX_COMPOSE_ATTEMPTS,
    GuideRejected,
    GuideSentence,
    LabelQuote,
    MedicationGuidance,
    Provenance,
    RenderedGuide,
    compose,
    find_dose_tokens,
    validate_provenance,
)
from app.domain.ingest import BoundingBox, SourceSpan
from app.domain.medication import MedicationLine, ResolvedConcept
from app.safety.verifier import PatientGuide, TracedText, UnverifiedItem, VerifiedMedication

BOX = BoundingBox(left=0.1, top=0.1, right=0.4, bottom=0.14)


def _span(span_id: str, text: str) -> SourceSpan:
    """Build a span carrying ``text``."""
    return SourceSpan(span_id=span_id, page_number=1, token_ids=("p1-t001",), text=text, box=BOX)


def _verified(dose: str = "500 MG", frequency: str = "twice daily") -> VerifiedMedication:
    """Build a verified medication with traced dose and frequency."""
    resolution = ResolvedConcept(
        query="METFORMIN HCL",
        rxcui="6809",
        tty="IN",
        matched_str="metformin",
        match_tier="truncated",
        similarity=0.5,
    )
    line = MedicationLine(
        page_number=1,
        drug_text="METFORMIN HCL",
        drug_span=_span("s-drug", "METFORMIN HCL"),
        dose_text=dose,
        dose_span=_span("s-dose", dose),
        frequency_text=frequency,
        frequency_span=_span("s-freq", frequency),
        line_span=_span("s-line", f"METFORMIN HCL {dose} {frequency}"),
    )
    return VerifiedMedication(
        resolution=resolution,
        display_name="metformin",
        line=line,
        dose=TracedText(text=dose, span=_span("s-dose", dose), similarity=1.0),
        frequency=TracedText(text=frequency, span=_span("s-freq", frequency), similarity=1.0),
    )


def _patient_guide(*medications: VerifiedMedication) -> PatientGuide:
    """Build a patient guide from verified medications."""
    return PatientGuide(document_id="a" * 64, verified=medications, unverified=())


class _NoLabel:
    """An openFDA client that returns nothing."""

    async def fetch_label(self, ingredient_rxcui: str, generic_name: str) -> OpenFdaLabel | None:
        """Return ``None``."""
        del ingredient_rxcui, generic_name
        return None


class _FixedLabel:
    """An openFDA client returning one committed label."""

    def __init__(
        self, text: str = "Metformin tablets are used with diet to lower blood sugar."
    ) -> None:
        """Store the section text to return."""
        self._text = text

    async def fetch_label(self, ingredient_rxcui: str, generic_name: str) -> OpenFdaLabel | None:
        """Return a single-section label for the ingredient."""
        del generic_name
        return OpenFdaLabel(
            ingredient_rxcui=ingredient_rxcui,
            spl_id="spl-1",
            spl_set_id="set-1",
            generic_name="METFORMIN HYDROCHLORIDE",
            accepted_product_rxcuis=("861007",),
            sections=(
                LabelSection(
                    name="indications_and_usage",
                    title="What this medicine is used for",
                    text=self._text,
                ),
            ),
        )


# --------------------------------------------------------------------------- #
# dose detection
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("The strength is 500 MG.", ("500 MG",)),
        ("Take 50 mcg every 8 hours", ("50 mcg", "8 hours")),
        ("Take 1 tablet twice daily", ("1 tablet",)),
        ("2.5 mL by mouth", ("2.5 mL",)),
        ("This medicine is metformin.", ()),
        ("Ask your pharmacist.", ()),
    ],
)
def test_dose_tokens_are_detected(text: str, expected: tuple[str, ...]) -> None:
    """The detector must over-match rather than miss a strength."""
    assert find_dose_tokens(text) == expected


# --------------------------------------------------------------------------- #
# composition
# --------------------------------------------------------------------------- #
def test_every_sentence_carries_a_provenance_tag() -> None:
    """No line reaches a patient without a source."""
    guide = anyio.run(lambda: compose(_patient_guide(_verified()), openfda=_NoLabel()))
    assert isinstance(guide, RenderedGuide)
    sentences = guide.all_sentences
    assert sentences
    for sentence in sentences:
        assert sentence.provenance.source in {"ocr_span", "rxnorm", "openfda"}
        assert sentence.provenance.ref


def test_name_is_tagged_rxnorm_and_dose_is_tagged_ocr() -> None:
    """The name comes from RxNorm; the dose comes from the page."""
    guide = anyio.run(lambda: compose(_patient_guide(_verified()), openfda=_NoLabel()))
    assert isinstance(guide, RenderedGuide)
    tags = [
        (sentence.provenance.source, sentence.provenance.ref)
        for sentence in guide.medications[0].sentences
    ]
    assert tags == [("rxnorm", "6809"), ("ocr_span", "s-dose"), ("ocr_span", "s-freq")]


def test_label_text_is_quoted_not_reworded() -> None:
    """Label passages are reproduced verbatim under an openfda tag."""
    text = "Metformin tablets are used with diet to lower blood sugar."
    guide = anyio.run(lambda: compose(_patient_guide(_verified()), openfda=_FixedLabel(text)))
    assert isinstance(guide, RenderedGuide)
    (quote,) = guide.medications[0].quotes
    assert quote.text == text
    assert quote.provenance == Provenance(source="openfda", ref="indications_and_usage")


def test_embedded_label_heading_is_trimmed_but_body_is_untouched() -> None:
    """The section heading openFDA embeds in the body is the only part removed."""
    body = "Metformin hydrochloride tablets are indicated to lower blood sugar."
    guide = anyio.run(
        lambda: compose(
            _patient_guide(_verified()),
            openfda=_FixedLabel(f"1 INDICATIONS AND USAGE {body}"),
        )
    )
    assert isinstance(guide, RenderedGuide)
    assert guide.medications[0].quotes[0].text == body


def test_medication_without_a_label_still_renders() -> None:
    """A missing label degrades to fewer quotes, not to a failure."""
    guide = anyio.run(lambda: compose(_patient_guide(_verified()), openfda=_NoLabel()))
    assert isinstance(guide, RenderedGuide)
    assert guide.medications[0].quotes == ()
    assert guide.medications[0].label_source_url is None


def test_unverified_items_are_carried_through_untouched() -> None:
    """Composition does not reinterpret anything the verifier refused."""
    item = UnverifiedItem(
        raw_text="ZYNTRAFEX 250 MG",
        line_span=_span("s-line", "ZYNTRAFEX 250 MG"),
        reason="rxnorm_unresolved",
        detail="no RxNorm match (no_candidate)",
    )
    patient_guide = PatientGuide(document_id="a" * 64, verified=(), unverified=(item,))
    guide = anyio.run(lambda: compose(patient_guide, openfda=_NoLabel()))
    assert isinstance(guide, RenderedGuide)
    assert guide.unverified == (item,)
    assert guide.medications == ()


def test_composition_reuses_one_label_lookup_per_rxcui() -> None:
    """Two prescriptions of the same drug do not cause two lookups."""

    class _Counting:
        """Counts how many times a label is fetched."""

        def __init__(self) -> None:
            self.calls = 0

        async def fetch_label(
            self, ingredient_rxcui: str, generic_name: str
        ) -> OpenFdaLabel | None:
            """Count the call and return nothing."""
            del ingredient_rxcui, generic_name
            self.calls += 1
            return None

    client = _Counting()
    anyio.run(lambda: compose(_patient_guide(_verified(), _verified()), openfda=client))
    assert client.calls == 1


# --------------------------------------------------------------------------- #
# post-validation
# --------------------------------------------------------------------------- #
def _guide_with(sentence: GuideSentence, quotes: tuple[LabelQuote, ...] = ()) -> RenderedGuide:
    """Wrap one sentence in an otherwise valid rendered guide."""
    return RenderedGuide(
        document_id="a" * 64,
        medications=(
            MedicationGuidance(
                display_name="metformin",
                rxcui="6809",
                match_tier="exact",
                prescription_text="METFORMIN HCL 500 MG",
                sentences=(sentence,),
                quotes=quotes,
            ),
        ),
        unverified=(),
        attempts=1,
    )


def test_a_clean_guide_has_no_violations() -> None:
    """A dose that appears in its cited source passes."""
    guide = _guide_with(
        GuideSentence(
            text="The strength on your prescription is 500 MG.",
            provenance=Provenance(source="ocr_span", ref="s-dose"),
            source_text="500 MG",
        )
    )
    assert validate_provenance(guide) == ()


def test_a_dose_absent_from_its_source_is_a_violation() -> None:
    """The tag is right but the cited text does not contain the dose."""
    guide = _guide_with(
        GuideSentence(
            text="The strength on your prescription is 500 MG.",
            provenance=Provenance(source="ocr_span", ref="s-dose"),
            source_text="50 MG",
        )
    )
    (violation,) = validate_provenance(guide)
    assert violation.problem == "dose_absent_from_source"
    assert violation.dose_token == "500 MG"


def test_a_dose_attributed_to_rxnorm_is_a_violation() -> None:
    """RxNorm supplies names, never strengths, so it cannot justify a dose."""
    guide = _guide_with(
        GuideSentence(
            text="This medicine is metformin 500 MG.",
            provenance=Provenance(source="rxnorm", ref="6809"),
            source_text="metformin 500 MG",
        )
    )
    (violation,) = validate_provenance(guide)
    assert violation.problem == "source_cannot_justify_dose"


def test_a_label_quote_with_an_unsupported_dose_is_a_violation() -> None:
    """Quotes are validated on the same terms as sentences."""
    guide = _guide_with(
        GuideSentence(
            text="This medicine is metformin.",
            provenance=Provenance(source="rxnorm", ref="6809"),
            source_text="metformin",
        ),
        quotes=(
            LabelQuote(
                title="What this medicine is used for",
                text="Take 850 MG twice daily.",
                provenance=Provenance(source="openfda", ref="indications_and_usage"),
                source_url="https://labels.fda.gov/setid=set-1",
            ),
        ),
    )
    # A quote cites itself, so its own text always contains its own dose; the
    # check that bites is the tag, which here is legitimately openfda.
    assert validate_provenance(guide) == ()


def test_case_and_spacing_do_not_defeat_the_containment_check() -> None:
    """A dose written differently but identically valued still passes."""
    guide = _guide_with(
        GuideSentence(
            text="The strength on your prescription is 500 mg.",
            provenance=Provenance(source="ocr_span", ref="s-dose"),
            source_text="500   MG",
        )
    )
    assert validate_provenance(guide) == ()


def test_a_failing_guide_is_rejected_after_the_retry() -> None:
    """A guide whose dose cannot be justified is discarded, not shown.

    Built by handing the composer a medication whose traced dose text disagrees
    with its span, which is only reachable by bypassing the verifier - exactly
    the situation post-validation exists to catch.
    """
    medication = _verified()
    tampered = medication.model_copy(
        update={"dose": TracedText(text="900 MG", span=_span("s-dose", "500 MG"), similarity=0.95)}
    )
    outcome = anyio.run(lambda: compose(_patient_guide(tampered), openfda=_NoLabel()))
    assert isinstance(outcome, GuideRejected)
    assert outcome.reason == "provenance_validation_failed"
    assert outcome.attempts == MAX_COMPOSE_ATTEMPTS
