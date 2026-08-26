"""Golden tests: the five fixture prescriptions must resolve identically.

Each committed file is the serialised outcome of the whole pipeline for one
prescription. Comparing bytes means a change anywhere - extraction, the
resolution ladder, or a verification gate - surfaces as a reviewable diff rather
than a silent shift in what a patient is told.

Regenerate deliberately with ``scripts/build_resolution_golden.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.adapters.extract import RuleBasedExtractor
from app.adapters.ocr import DeterministicFakeOcr
from app.adapters.rxnorm import PostgresRxNormResolver
from app.config import Settings
from app.domain import pipeline
from app.safety.verifier import ASK_PHARMACIST_MESSAGE

pytestmark = pytest.mark.anyio

GOLDEN_DIR = Path(__file__).resolve().parents[1] / "golden" / "resolution"

# Expected shape of each outcome, kept next to the byte comparison so a reader
# can see the intent without opening the JSON.
EXPECTED = {
    "clean_three_drugs": {"verified": 3, "unverified": 0, "rejected": False},
    "smudged_two_drugs": {"verified": 1, "unverified": 1, "rejected": False},
    "two_drugs_pdf": {"verified": 2, "unverified": 0, "rejected": False},
    "illegible_scan": {"verified": 0, "unverified": 0, "rejected": True},
    "unknown_drug_and_blank_page": {"verified": 1, "unverified": 1, "rejected": False},
}

EXPECTED_RXCUIS = {
    "clean_three_drugs": ("6809", "29046", "83366"),
    "smudged_two_drugs": ("5640",),
    "two_drugs_pdf": ("40144", "7646"),
    "unknown_drug_and_blank_page": ("104416",),
}


async def _run(stem: str, resolver: PostgresRxNormResolver, settings: Settings) -> object:
    """Run the pipeline for one committed prescription."""
    path = next(settings.sample_prescription_dir.glob(f"{stem}.*"))
    return await pipeline.run(
        path.read_bytes(),
        ocr=DeterministicFakeOcr(settings.ocr_fixture_dir),
        extractor=RuleBasedExtractor(),
        resolver=resolver,
        settings=settings,
    )


def test_a_golden_file_exists_for_every_prescription(settings: Settings) -> None:
    """Five prescriptions, five committed outcomes."""
    stems = {path.stem for path in settings.sample_prescription_dir.iterdir()}
    assert stems == set(EXPECTED)
    assert {path.stem for path in GOLDEN_DIR.glob("*.json")} == stems


@pytest.mark.parametrize("stem", sorted(EXPECTED))
async def test_resolution_is_byte_identical_to_the_golden_file(
    stem: str, resolver: PostgresRxNormResolver, settings: Settings
) -> None:
    """Re-running the pipeline reproduces the committed JSON exactly."""
    result = await _run(stem, resolver, settings)
    assert isinstance(result, pipeline.PipelineResult)
    body = (
        result.gate_report.model_dump_json()
        if result.guide is None
        else result.guide.model_dump_json()
    )
    assert body.encode("utf-8") == (GOLDEN_DIR / f"{stem}.json").read_bytes()


@pytest.mark.parametrize("stem", sorted(EXPECTED))
async def test_verified_and_unverified_counts_are_pinned(
    stem: str, resolver: PostgresRxNormResolver, settings: Settings
) -> None:
    """The accept/reject split for each fixture is locked down."""
    result = await _run(stem, resolver, settings)
    assert isinstance(result, pipeline.PipelineResult)
    expected = EXPECTED[stem]
    assert result.rejected is expected["rejected"]
    if result.guide is None:
        return
    assert len(result.guide.verified) == expected["verified"]
    assert len(result.guide.unverified) == expected["unverified"]


@pytest.mark.parametrize("stem", sorted(EXPECTED_RXCUIS))
async def test_verified_medications_carry_the_expected_real_rxcuis(
    stem: str, resolver: PostgresRxNormResolver, settings: Settings
) -> None:
    """Every RxCUI shown is the real NLM identifier for that ingredient."""
    result = await _run(stem, resolver, settings)
    assert isinstance(result, pipeline.PipelineResult)
    assert result.guide is not None
    assert result.guide.rxcuis == EXPECTED_RXCUIS[stem]


async def test_low_confidence_medication_is_withheld(
    resolver: PostgresRxNormResolver, settings: Settings
) -> None:
    """The smudged strength on the amoxicillin line blocks that medication."""
    result = await _run("smudged_two_drugs", resolver, settings)
    assert isinstance(result, pipeline.PipelineResult)
    assert result.guide is not None
    (item,) = result.guide.unverified
    assert item.reason == "low_confidence_tokens"
    assert "AMOXICILLIN" in item.raw_text
    assert item.message == ASK_PHARMACIST_MESSAGE
    assert all("AMOXICILLIN" not in med.line.drug_text for med in result.guide.verified)


async def test_unknown_drug_is_withheld(
    resolver: PostgresRxNormResolver, settings: Settings
) -> None:
    """A drug name RxNorm does not contain never appears as verified."""
    result = await _run("unknown_drug_and_blank_page", resolver, settings)
    assert isinstance(result, pipeline.PipelineResult)
    assert result.guide is not None
    (item,) = result.guide.unverified
    assert item.reason == "rxnorm_unresolved"
    assert "ZYNTRAFEX" in item.raw_text
    assert all("ZYNTRAFEX" not in med.display_name.upper() for med in result.guide.verified)


async def test_rejected_document_produces_no_guide_at_all(
    resolver: PostgresRxNormResolver, settings: Settings
) -> None:
    """An unreadable scan yields nothing, not a partial guide."""
    result = await _run("illegible_scan", resolver, settings)
    assert isinstance(result, pipeline.PipelineResult)
    assert result.guide is None
    assert result.gate_report.rejection_reason == "mean_confidence_below_threshold"
