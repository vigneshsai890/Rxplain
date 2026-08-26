"""End-to-end assertions on what a patient actually receives.

Two acceptance criteria live here:

* **100% provenance coverage.** Every sentence and every quote in the rendered
  guide carries a tag, and that tag is reproduced in the HTML so the disclosure
  a patient opens shows the same reference the model recorded.
* **Reading level.** The wording RxPlain chooses is measured with ``textstat``.
  Drug names are substituted out first, and the docstring on that test explains
  why.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import textstat
from fastapi.testclient import TestClient

from app.adapters.extract import RuleBasedExtractor
from app.adapters.ocr import DeterministicFakeOcr
from app.adapters.openfda import NullOpenFdaClient, PostgresCachedOpenFdaClient
from app.adapters.rxnorm import PostgresRxNormResolver
from app.config import Settings
from app.db import Database
from app.domain import guide as guide_module
from app.domain import pipeline
from app.domain.guide import RenderedGuide, find_dose_tokens, validate_provenance
from app.main import create_app
from app.safety.verifier import ASK_PHARMACIST_MESSAGE
from app.web.routes import Services

pytestmark = pytest.mark.anyio

TARGET_READING_GRADE = 6.0

# Prescriptions that produce at least one verified medication.
GUIDE_FIXTURES = (
    "clean_three_drugs",
    "smudged_two_drugs",
    "two_drugs_pdf",
    "unknown_drug_and_blank_page",
)


async def _render(stem: str, database: Database, settings: Settings) -> RenderedGuide:
    """Run the pipeline and compose the guide for one prescription.

    Raises:
        AssertionError: If the document is rejected or composition fails.
    """
    path = next(settings.sample_prescription_dir.glob(f"{stem}.*"))
    result = await pipeline.run(
        path.read_bytes(),
        ocr=DeterministicFakeOcr(settings.ocr_fixture_dir),
        extractor=RuleBasedExtractor(),
        resolver=PostgresRxNormResolver(database, settings),
        settings=settings,
    )
    assert result.guide is not None, f"{stem} was rejected at the gate"
    composed = await guide_module.compose(
        result.guide, openfda=PostgresCachedOpenFdaClient(database, settings)
    )
    assert isinstance(composed, RenderedGuide), f"{stem} failed provenance validation"
    return composed


# --------------------------------------------------------------------------- #
# provenance coverage
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("stem", GUIDE_FIXTURES)
async def test_every_line_of_output_has_provenance(
    stem: str, seeded_database: Database, settings: Settings
) -> None:
    """100% coverage: no sentence and no quote is untagged."""
    guide = await _render(stem, seeded_database, settings)
    lines = [
        (sentence.text, sentence.provenance)
        for medication in guide.medications
        for sentence in medication.sentences
    ]
    lines.extend(
        (quote.text, quote.provenance)
        for medication in guide.medications
        for quote in medication.quotes
    )
    assert lines, f"{stem} produced no patient-facing text"
    for text, provenance in lines:
        assert provenance.source in {"ocr_span", "rxnorm", "openfda"}, text
        assert provenance.ref.strip(), text


@pytest.mark.parametrize("stem", GUIDE_FIXTURES)
async def test_no_dose_survives_without_a_traceable_source(
    stem: str, seeded_database: Database, settings: Settings
) -> None:
    """Post-validation finds nothing to complain about on real fixtures."""
    guide = await _render(stem, seeded_database, settings)
    assert validate_provenance(guide) == ()


@pytest.mark.parametrize("stem", GUIDE_FIXTURES)
async def test_every_dose_shown_appears_in_the_prescription(
    stem: str, seeded_database: Database, settings: Settings
) -> None:
    """Independent check: each dose in a sentence is in the OCR line it cites."""
    guide = await _render(stem, seeded_database, settings)
    for medication in guide.medications:
        for sentence in medication.sentences:
            for token in find_dose_tokens(sentence.text):
                folded = " ".join(token.casefold().split())
                assert folded in " ".join(sentence.source_text.casefold().split())


@pytest.mark.parametrize("stem", GUIDE_FIXTURES)
async def test_rendered_html_shows_the_same_refs_the_model_recorded(
    stem: str, seeded_database: Database, settings: Settings
) -> None:
    """The disclosure a patient opens matches the recorded provenance."""
    guide = await _render(stem, seeded_database, settings)
    app = create_app(settings)
    app.state.services = Services(
        settings=settings,
        ocr=DeterministicFakeOcr(settings.ocr_fixture_dir),
        extractor=RuleBasedExtractor(),
        resolver=PostgresRxNormResolver(seeded_database, settings),
        openfda=PostgresCachedOpenFdaClient(seeded_database, settings),
    )
    path = next(settings.sample_prescription_dir.glob(f"{stem}.*"))
    with TestClient(app) as client:
        markup = client.post(
            "/guide", files={"prescription": (path.name, path.read_bytes(), "image/png")}
        ).text

    for medication in guide.medications:
        for sentence in medication.sentences:
            expected = f"{sentence.provenance.source}:{sentence.provenance.ref}"
            assert expected in markup, f"{expected} missing from rendered page"


# --------------------------------------------------------------------------- #
# reading level
# --------------------------------------------------------------------------- #
def _neutralise_drug_names(guide: RenderedGuide) -> str:
    """Replace drug names with a short placeholder before measuring.

    Drug names are the one part of a sentence RxPlain must not simplify: they come
    from RxNorm verbatim, and "levothyroxine sodium" alone scores above grade 12
    on its own. Measuring with names substituted tests the wording RxPlain chose,
    which is the only part it controls.
    """
    text = " ".join(
        sentence.text for medication in guide.medications for sentence in medication.sentences
    )
    for medication in guide.medications:
        text = text.replace(medication.display_name, "this drug")
    return text


@pytest.mark.parametrize("stem", GUIDE_FIXTURES)
async def test_composed_wording_is_at_or_below_sixth_grade(
    stem: str, seeded_database: Database, settings: Settings
) -> None:
    """RxPlain's own wording reads at a sixth-grade level or simpler."""
    guide = await _render(stem, seeded_database, settings)
    grade = textstat.flesch_kincaid_grade(_neutralise_drug_names(guide))
    assert grade <= TARGET_READING_GRADE, f"{stem} reads at grade {grade:.1f}"


async def test_the_disclaimer_is_at_or_below_sixth_grade(settings: Settings) -> None:
    """The safety notice is the text most important to understand."""
    base = Path(settings.sample_prescription_dir).parents[2] / "app" / "web" / "templates"
    markup = (base / "base.html").read_text(encoding="utf-8")
    footer = markup.split("<footer")[1]
    prose = re.sub(r"<[^>]+>", " ", footer).replace("&mdash;", "-")
    prose = " ".join(prose.split())
    grade = textstat.flesch_kincaid_grade(prose)
    assert grade <= TARGET_READING_GRADE, f"disclaimer reads at grade {grade:.1f}"


async def test_unverified_wording_is_the_mandated_string() -> None:
    """The unverified message is fixed by requirement, so it is asserted verbatim.

    It is deliberately not held to :data:`TARGET_READING_GRADE`. The phrase scores
    grade 7.4, driven almost entirely by "pharmacist" (four syllables) and
    "verify" (three). Rewording a mandated safety string to move a readability
    score would be the wrong trade, so the ceiling here is looser and the reason
    is recorded.
    """
    assert ASK_PHARMACIST_MESSAGE == "We couldn't verify this - ask your pharmacist"
    assert textstat.flesch_kincaid_grade(ASK_PHARMACIST_MESSAGE) <= 8.0


# --------------------------------------------------------------------------- #
# openFDA disabled
# --------------------------------------------------------------------------- #
async def test_guide_still_renders_with_openfda_disabled(
    seeded_database: Database, settings: Settings
) -> None:
    """Losing openFDA removes quotes; it never blocks the verified fields."""
    path = next(settings.sample_prescription_dir.glob("clean_three_drugs.*"))
    result = await pipeline.run(
        path.read_bytes(),
        ocr=DeterministicFakeOcr(settings.ocr_fixture_dir),
        extractor=RuleBasedExtractor(),
        resolver=PostgresRxNormResolver(seeded_database, settings),
        settings=settings,
    )
    assert result.guide is not None
    composed = await guide_module.compose(result.guide, openfda=NullOpenFdaClient())
    assert isinstance(composed, RenderedGuide)
    assert len(composed.medications) == 3
    assert all(medication.quotes == () for medication in composed.medications)
    assert all(medication.sentences for medication in composed.medications)
