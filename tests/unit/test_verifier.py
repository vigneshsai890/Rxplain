"""Tests for :mod:`app.safety.verifier`, including the core safety property.

The property that matters is stated once, in
:func:`test_no_input_can_place_an_unresolved_drug_in_a_guide`: whatever the
extractor produces and whatever the resolver answers, a ``PatientGuide`` never
contains a drug RxNorm did not confirm.
"""

from __future__ import annotations

import json

import anyio
import pytest
from hypothesis import given
from hypothesis import settings as hypothesis_settings
from hypothesis import strategies as st
from pydantic import ValidationError

from app.config import Settings
from app.domain.ingest import BoundingBox, SourceSpan
from app.domain.medication import (
    MedicationLine,
    ResolvedConcept,
    RxNormResolution,
    UnresolvedConcept,
)
from app.safety.gates import GateReport
from app.safety.verifier import (
    ABSOLUTE_MIN_SIMILARITY,
    ASK_PHARMACIST_MESSAGE,
    PatientGuide,
    TracedText,
    UnverifiedItem,
    VerifiedMedication,
    _trace,
    verbatim_similarity,
    verify_lines,
)
from app.web.logging import configure_logging

BOX = BoundingBox(left=0.1, top=0.1, right=0.4, bottom=0.14)


def _span(span_id: str, text: str, token_ids: tuple[str, ...]) -> SourceSpan:
    """Build a span with the given text and token identifiers."""
    return SourceSpan(span_id=span_id, page_number=1, token_ids=token_ids, text=text, box=BOX)


def _line(
    drug: str = "METFORMIN HCL",
    dose: str | None = "500 MG",
    frequency: str | None = "twice daily",
    *,
    dose_span_text: str | None = None,
    frequency_span_text: str | None = None,
    token_ids: tuple[str, ...] = ("p1-t001", "p1-t002", "p1-t003"),
) -> MedicationLine:
    """Build a medication line whose spans reproduce its text by default."""
    return MedicationLine(
        page_number=1,
        drug_text=drug,
        drug_span=_span("s-drug", drug, ("p1-t001",)),
        dose_text=dose,
        dose_span=(
            _span("s-dose", dose_span_text if dose_span_text is not None else dose, ("p1-t002",))
            if dose is not None
            else None
        ),
        frequency_text=frequency,
        frequency_span=(
            _span(
                "s-freq",
                frequency_span_text if frequency_span_text is not None else frequency,
                ("p1-t003",),
            )
            if frequency is not None
            else None
        ),
        line_span=_span("s-line", f"{drug} {dose} {frequency}", token_ids),
    )


def _gate(flagged: frozenset[str] = frozenset()) -> GateReport:
    """Build a gate report with the given flagged token identifiers."""
    return GateReport(
        document_id="a" * 64,
        token_count=10,
        mean_confidence=0.95,
        mean_confidence_threshold=0.80,
        token_confidence_threshold=0.60,
        low_confidence_token_ids=flagged,
        rejection_reason=None,
    )


def _resolved(query: str = "METFORMIN HCL") -> ResolvedConcept:
    """Build a resolved concept for ``query``."""
    return ResolvedConcept(
        query=query,
        rxcui="6809",
        tty="IN",
        matched_str="metformin",
        match_tier="truncated",
        similarity=0.5,
    )


class _StubResolver:
    """Resolver returning a fixed answer, or per-query answers."""

    def __init__(self, answers: dict[str, RxNormResolution] | None = None) -> None:
        """Store the lookup table; unknown queries resolve as unresolved."""
        self._answers = answers or {}

    async def resolve(self, drug_text: str) -> RxNormResolution:
        """Return the configured answer for ``drug_text``."""
        return self._answers.get(
            drug_text,
            UnresolvedConcept(query=drug_text, reason="no_candidate", best_similarity=0.0),
        )


# --------------------------------------------------------------------------- #
# TracedText: the threshold lives in the type
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("similarity", [0.0, 0.5, 0.89, 0.8999])
def test_traced_text_cannot_be_constructed_below_the_floor(similarity: float) -> None:
    """An untraceable string cannot be represented, not merely rejected."""
    with pytest.raises(ValidationError):
        TracedText(text="500 MG", span=_span("s", "500 MG", ("p1-t002",)), similarity=similarity)


@given(similarity=st.floats(min_value=0.0, max_value=ABSOLUTE_MIN_SIMILARITY, exclude_max=True))
@hypothesis_settings(max_examples=100, deadline=None)
def test_no_similarity_below_the_floor_is_constructible(similarity: float) -> None:
    """Property: the floor holds for every float below it."""
    with pytest.raises(ValidationError):
        TracedText(text="x", span=_span("s", "x", ("p1-t001",)), similarity=similarity)


def test_traced_text_accepts_the_floor_exactly() -> None:
    """The bound is inclusive, so 0.90 is traceable."""
    traced = TracedText(
        text="500 MG", span=_span("s", "500 MG", ("p1-t002",)), similarity=ABSOLUTE_MIN_SIMILARITY
    )
    assert traced.similarity == ABSOLUTE_MIN_SIMILARITY


def test_verbatim_similarity_is_one_for_identical_text() -> None:
    """Case and whitespace differences do not count against a match."""
    assert verbatim_similarity("500  mg", _span("s", "500 MG", ("p1-t002",))) == 1.0


def test_formatting_only_difference_still_passes() -> None:
    """A missing space is formatting, not a dose change, so it stays traceable."""
    score = verbatim_similarity("500mg", _span("s", "500 MG", ("p1-t002",)))
    assert score >= ABSOLUTE_MIN_SIMILARITY


@pytest.mark.parametrize(
    ("claimed", "span_text"),
    [
        # A plain fuzzy ratio scores this pair 0.91 and would let it through.
        ("50 MG", "500 MG"),
        ("500 MG", "50 MG"),
        # A thousand-fold unit error. Fuzzy ratio scores 0.92.
        ("500 MG", "500 MCG"),
        ("500 MCG", "500 MG"),
        # Digit transposition and insertion.
        ("250 MG", "205 MG"),
        ("1000 MG", "100 MG"),
        ("every 8 hours", "every 6 hours"),
        ("2 tablets", "1 tablets"),
    ],
)
def test_number_or_unit_differences_score_zero(claimed: str, span_text: str) -> None:
    """A dose error can never be smoothed over by edit distance."""
    assert verbatim_similarity(claimed, _span("s", span_text, ("p1-t002",))) == 0.0


def test_paraphrase_without_numbers_is_caught_by_the_fuzzy_ratio() -> None:
    """Reworded frequency text falls below the floor."""
    score = verbatim_similarity("twice a day", _span("s", "twice daily", ("p1-t003",)))
    assert score < ABSOLUTE_MIN_SIMILARITY


# --------------------------------------------------------------------------- #
# VerifiedMedication: the resolution type carries the guarantee
# --------------------------------------------------------------------------- #
def test_verified_medication_rejects_an_unresolved_concept() -> None:
    """Static typing forbids this; validation refuses it at runtime too."""
    unresolved = UnresolvedConcept(query="ZYNTRAFEX", reason="no_candidate", best_similarity=0.0)
    with pytest.raises(ValidationError):
        VerifiedMedication(
            resolution=unresolved,  # type: ignore[arg-type]
            display_name="ZYNTRAFEX",
            line=_line(),
        )


def test_display_name_must_come_from_rxnorm() -> None:
    """The shown name is the curated RxNorm string, never raw OCR text."""
    with pytest.raises(ValidationError, match="RxNorm matched string"):
        VerifiedMedication(resolution=_resolved(), display_name="METFORMIN HCL 500MG", line=_line())


def test_unverified_item_message_is_fixed() -> None:
    """Every unverified item shows the same wording."""
    item = UnverifiedItem(
        raw_text="ZYNTRAFEX 250 MG",
        line_span=_span("s", "ZYNTRAFEX 250 MG", ("p1-t001",)),
        reason="rxnorm_unresolved",
        detail="no RxNorm match (no_candidate)",
    )
    assert item.message == ASK_PHARMACIST_MESSAGE


# --------------------------------------------------------------------------- #
# verify_lines: gate by gate
# --------------------------------------------------------------------------- #
@pytest.mark.anyio
async def test_resolved_and_traceable_line_is_verified(settings: Settings) -> None:
    """The happy path: RxCUI resolved and both fields traced."""
    guide = await verify_lines(
        [_line()],
        resolver=_StubResolver({"METFORMIN HCL": _resolved()}),
        gate_report=_gate(),
        settings=settings,
    )
    assert len(guide.verified) == 1
    assert guide.unverified == ()
    assert guide.rxcuis == ("6809",)
    assert guide.verified[0].display_name == "metformin"


@pytest.mark.anyio
async def test_unresolved_drug_becomes_unverified(settings: Settings) -> None:
    """A drug RxNorm cannot confirm never reaches the verified list."""
    guide = await verify_lines(
        [_line(drug="ZYNTRAFEX")],
        resolver=_StubResolver(),
        gate_report=_gate(),
        settings=settings,
    )
    assert guide.verified == ()
    assert [item.reason for item in guide.unverified] == ["rxnorm_unresolved"]


@pytest.mark.anyio
async def test_low_confidence_token_blocks_verification(settings: Settings) -> None:
    """One doubtful token disqualifies the whole medication."""
    guide = await verify_lines(
        [_line()],
        resolver=_StubResolver({"METFORMIN HCL": _resolved()}),
        gate_report=_gate(frozenset({"p1-t002"})),
        settings=settings,
    )
    assert guide.verified == ()
    assert [item.reason for item in guide.unverified] == ["low_confidence_tokens"]


@pytest.mark.anyio
async def test_untraceable_dose_blocks_verification(settings: Settings) -> None:
    """A dose that does not match its span is refused, not shown."""
    guide = await verify_lines(
        [_line(dose="900 MG", dose_span_text="500 MG")],
        resolver=_StubResolver({"METFORMIN HCL": _resolved()}),
        gate_report=_gate(),
        settings=settings,
    )
    assert guide.verified == ()
    assert [item.reason for item in guide.unverified] == ["dose_not_traceable"]


@pytest.mark.anyio
async def test_line_without_dose_or_frequency_can_still_verify(settings: Settings) -> None:
    """Absent fields are not failures; they are simply absent."""
    guide = await verify_lines(
        [_line(dose=None, frequency=None)],
        resolver=_StubResolver({"METFORMIN HCL": _resolved()}),
        gate_report=_gate(),
        settings=settings,
    )
    assert len(guide.verified) == 1
    assert guide.verified[0].dose is None
    assert guide.verified[0].frequency is None


@pytest.mark.anyio
async def test_every_line_is_accounted_for(settings: Settings) -> None:
    """Nothing is silently dropped: each input lands in exactly one bucket."""
    lines = [_line(), _line(drug="ZYNTRAFEX"), _line(drug="LISINOPRIL")]
    guide = await verify_lines(
        lines,
        resolver=_StubResolver({"METFORMIN HCL": _resolved()}),
        gate_report=_gate(),
        settings=settings,
    )
    assert len(guide.verified) + len(guide.unverified) == len(lines)


@pytest.mark.anyio
async def test_tightened_threshold_rejects_a_borderline_field(settings: Settings) -> None:
    """Configuration can only make verification stricter.

    The pair below has identical numbers and units, so it clears the exact
    precondition and is graded purely on the fuzzy ratio.
    """
    borderline = _line(frequency="twice daily", frequency_span_text="twice daily.")
    lenient = await verify_lines(
        [borderline],
        resolver=_StubResolver({"METFORMIN HCL": _resolved()}),
        gate_report=_gate(),
        settings=settings,
    )
    assert len(lenient.verified) == 1

    strict = Settings(app_env="test", verbatim_min_similarity=1.0)
    tightened = await verify_lines(
        [borderline],
        resolver=_StubResolver({"METFORMIN HCL": _resolved()}),
        gate_report=_gate(),
        settings=strict,
    )
    assert tightened.verified == ()
    assert [item.reason for item in tightened.unverified] == ["frequency_not_traceable"]


@pytest.mark.anyio
async def test_verified_medication_carries_its_traced_dose_and_frequency(
    settings: Settings,
) -> None:
    """The traced values are attached, not dropped on the way through.

    Mutation testing found that nothing asserted this: replacing ``dose=dose``
    with ``dose=None`` left every test passing while silently withholding the
    dose from the patient.
    """
    guide = await verify_lines(
        [_line(dose="500 MG", frequency="twice daily")],
        resolver=_StubResolver({"METFORMIN HCL": _resolved()}),
        gate_report=_gate(),
        settings=settings,
    )
    (medication,) = guide.verified
    assert medication.dose is not None
    assert medication.dose.text == "500 MG"
    assert medication.dose.span.span_id == "s-dose"
    assert medication.dose.similarity == 1.0
    assert medication.frequency is not None
    assert medication.frequency.text == "twice daily"
    assert medication.frequency.span.span_id == "s-freq"


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("failing_line", "flagged", "expected_reason"),
    [
        (_line(drug="ZYNTRAFEX"), frozenset(), "rxnorm_unresolved"),
        (_line(), frozenset({"p1-t001"}), "low_confidence_tokens"),
        (_line(dose="900 MG", dose_span_text="500 MG"), frozenset(), "dose_not_traceable"),
        (
            _line(frequency="three times daily", frequency_span_text="twice daily"),
            frozenset(),
            "frequency_not_traceable",
        ),
    ],
)
async def test_an_unverified_line_does_not_stop_the_ones_after_it(
    settings: Settings,
    failing_line: MedicationLine,
    flagged: frozenset[str],
    expected_reason: str,
) -> None:
    """A rejection skips one medication, never abandons the rest of the page.

    Mutation testing found that turning any gate's ``continue`` into a ``break``
    passed the suite: everything below the first problem would have vanished from
    the guide with no indication it existed.
    """
    good = _line(drug="LISINOPRIL", token_ids=("p1-t011", "p1-t012", "p1-t013"))
    answers: dict[str, RxNormResolution] = {
        "METFORMIN HCL": _resolved(),
        "LISINOPRIL": ResolvedConcept(
            query="LISINOPRIL",
            rxcui="29046",
            tty="IN",
            matched_str="lisinopril",
            match_tier="exact",
            similarity=1.0,
        ),
    }
    guide = await verify_lines(
        [failing_line, good],
        resolver=_StubResolver(answers),
        gate_report=_gate(flagged),
        settings=settings,
    )
    assert [item.reason for item in guide.unverified] == [expected_reason]
    assert [medication.display_name for medication in guide.verified] == ["lisinopril"]


@pytest.mark.anyio
async def test_verification_emits_an_audit_record(
    settings: Settings, capsys: pytest.CaptureFixture[str]
) -> None:
    """The outcome is logged with counts, so a decision can be reconstructed.

    Asserted because this record is the audit trail for what a patient was
    shown, and mutation testing showed the whole statement could be deleted or
    have its fields blanked with the suite still passing.
    """
    configure_logging("INFO")
    guide = await verify_lines(
        [_line(), _line(drug="ZYNTRAFEX")],
        resolver=_StubResolver({"METFORMIN HCL": _resolved()}),
        gate_report=_gate(),
        settings=settings,
    )
    records = [
        json.loads(line)
        for line in capsys.readouterr().out.splitlines()
        if line.strip().startswith("{")
    ]
    completions = [record for record in records if record.get("message") == "verification_complete"]
    assert len(completions) == 1
    assert completions[0]["document_id"] == "a" * 64
    assert completions[0]["verified"] == len(guide.verified) == 1
    assert completions[0]["unverified"] == len(guide.unverified) == 1


# --------------------------------------------------------------------------- #
# the tracing helper
# --------------------------------------------------------------------------- #
def test_trace_treats_an_absent_field_as_nothing_to_check() -> None:
    """A field with no text and no span is absent, not untraceable."""
    assert _trace(None, None, 0.9) == (None, 1.0)


def test_trace_refuses_text_with_no_span() -> None:
    """Text without a span is an unciteable claim, so it is never traced."""
    assert _trace("500 MG", None, 0.9) == (None, 1.0)


def test_trace_refuses_a_span_with_no_text() -> None:
    """A span with nothing claimed against it yields no traced text."""
    assert _trace(None, _span("s", "500 MG", ("p1-t002",)), 0.9) == (None, 1.0)


def test_trace_returns_traced_text_above_the_threshold() -> None:
    """A faithful reproduction is traced and scored."""
    traced, similarity = _trace("500 MG", _span("s", "500 MG", ("p1-t002",)), 0.9)
    assert traced is not None
    assert traced.text == "500 MG"
    assert similarity == 1.0


def test_trace_returns_the_failing_score_below_the_threshold() -> None:
    """A failed trace reports how far off it was, for the unverified detail."""
    traced, similarity = _trace("50 MG", _span("s", "500 MG", ("p1-t002",)), 0.9)
    assert traced is None
    assert similarity == 0.0


# --------------------------------------------------------------------------- #
# The safety property
# --------------------------------------------------------------------------- #
@given(
    drug_names=st.lists(
        st.text(
            alphabet=st.characters(
                min_codepoint=32, max_codepoint=126, blacklist_characters="\r\n"
            ),
            min_size=1,
            max_size=24,
        ).filter(lambda value: value.strip() != ""),
        min_size=0,
        max_size=6,
    ),
    resolvable=st.lists(st.booleans(), min_size=0, max_size=6),
    flag_tokens=st.booleans(),
)
@hypothesis_settings(max_examples=200, deadline=None)
def test_no_input_can_place_an_unresolved_drug_in_a_guide(
    drug_names: list[str], resolvable: list[bool], *, flag_tokens: bool
) -> None:
    """No combination of inputs yields a guide containing an unresolved drug."""
    lines = [_line(drug=name) for name in drug_names]
    answers: dict[str, RxNormResolution] = {}
    for name, is_resolvable in zip(drug_names, resolvable, strict=False):
        if is_resolvable:
            answers[name] = ResolvedConcept(
                query=name,
                rxcui="6809",
                tty="IN",
                matched_str="metformin",
                match_tier="approximate",
                similarity=0.9,
            )

    guide = anyio.run(
        lambda: verify_lines(
            lines,
            resolver=_StubResolver(answers),
            gate_report=_gate(frozenset({"p1-t001"}) if flag_tokens else frozenset()),
            settings=Settings(app_env="test"),
        )
    )

    assert isinstance(guide, PatientGuide)
    for medication in guide.verified:
        assert isinstance(medication.resolution, ResolvedConcept)
        assert medication.line.drug_text in answers
        assert medication.rxcui.isdigit()
    assert len(guide.verified) + len(guide.unverified) == len(lines)
    if flag_tokens:
        assert guide.verified == ()


@given(drug_names=st.lists(st.sampled_from(["A", "B", "C", "ZYNTRAFEX"]), max_size=8))
@hypothesis_settings(max_examples=100, deadline=None)
def test_a_resolver_that_never_resolves_yields_no_verified_medications(
    drug_names: list[str],
) -> None:
    """Property: with resolution always failing, the verified list stays empty."""
    guide = anyio.run(
        lambda: verify_lines(
            [_line(drug=name) for name in drug_names],
            resolver=_StubResolver(),
            gate_report=_gate(),
            settings=Settings(app_env="test"),
        )
    )
    assert guide.verified == ()
    assert len(guide.unverified) == len(drug_names)
