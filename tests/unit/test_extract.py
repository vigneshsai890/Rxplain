"""Tests for :mod:`app.adapters.extract`."""

from __future__ import annotations

import pytest

from app.adapters.extract import (
    MedicationExtractor,
    RuleBasedExtractor,
    group_lines,
)
from app.domain.ingest import BoundingBox, OcrDocument, OcrPage, OcrToken

LINE_HEIGHT = 0.06


def _line_tokens(text: str, row: int, start_index: int, confidence: float = 0.99) -> list[OcrToken]:
    """Lay a line of text out horizontally on visual row ``row``."""
    tokens: list[OcrToken] = []
    left = 0.05
    for offset, word in enumerate(text.split(" ")):
        width = 0.012 * len(word)
        tokens.append(
            OcrToken(
                token_id=f"p1-t{start_index + offset:03d}",
                text=word,
                confidence=confidence,
                box=BoundingBox(
                    left=left,
                    top=0.05 + row * LINE_HEIGHT,
                    right=left + width,
                    bottom=0.05 + row * LINE_HEIGHT + 0.03,
                ),
            )
        )
        left += width + 0.006
    return tokens


def _document(*lines: str) -> OcrDocument:
    """Build a one-page document from text lines, one visual row each."""
    tokens: list[OcrToken] = []
    for row, text in enumerate(lines):
        tokens.extend(_line_tokens(text, row, len(tokens) + 1))
    return OcrDocument(
        document_id="e" * 64,
        media_type="image/png",
        pages=(OcrPage(page_number=1, width_px=1000, height_px=620, tokens=tuple(tokens)),),
    )


def test_rule_based_extractor_satisfies_the_protocol() -> None:
    """The shipped implementation is usable wherever the seam is declared."""
    assert isinstance(RuleBasedExtractor(), MedicationExtractor)


def test_group_lines_recovers_visual_rows() -> None:
    """Tokens are grouped by vertical band, then ordered left to right."""
    document = _document("METFORMIN HCL 500 MG", "Take 1 tablet once daily")
    lines = group_lines(document.pages[0])
    assert [" ".join(token.text for token in line) for line in lines] == [
        "METFORMIN HCL 500 MG",
        "Take 1 tablet once daily",
    ]


def test_group_lines_on_empty_page_is_empty() -> None:
    """A blank page has no lines rather than one empty line."""
    page = OcrPage(page_number=1, width_px=10, height_px=10, tokens=())
    assert group_lines(page) == ()


@pytest.mark.anyio
async def test_extracts_drug_dose_and_frequency() -> None:
    """The three fields come out as verbatim slices of the input."""
    document = _document("METFORMIN HCL 500 MG TABLET", "Take 1 tablet by mouth twice daily")
    (line,) = await RuleBasedExtractor().extract(document)
    assert line.drug_text == "METFORMIN HCL"
    assert line.dose_text == "500 MG"
    assert line.frequency_text == "twice daily"


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("sig", "expected"),
    [
        ("Take 1 tablet by mouth once daily", "once daily"),
        ("Take 1 tablet by mouth twice daily with meals", "twice daily with meals"),
        ("Take 1 capsule by mouth three times daily", "three times daily"),
        ("Take 1 tablet by mouth every 6 hours as needed", "every 6 hours as needed"),
        ("Take 1 tablet by mouth at bedtime", "at bedtime"),
        ("Take 1 tablet by mouth every morning", "every morning"),
        ("Take 1 tablet by mouth daily", "daily"),
    ],
)
async def test_frequency_phrases_are_matched_verbatim(sig: str, expected: str) -> None:
    """Each recognised phrase is returned exactly as written, not normalised."""
    document = _document("LISINOPRIL 10 MG TABLET", sig)
    (line,) = await RuleBasedExtractor().extract(document)
    assert line.frequency_text == expected


@pytest.mark.anyio
async def test_every_field_is_reproducible_from_its_span() -> None:
    """Extraction selects token ranges, so text always equals its span text."""
    document = _document("ATORVASTATIN CALCIUM 20 MG TABLET", "Take 1 tablet at bedtime")
    (line,) = await RuleBasedExtractor().extract(document)
    assert line.drug_text == line.drug_span.text
    assert line.dose_span is not None
    assert line.dose_text == line.dose_span.text
    assert line.frequency_span is not None
    assert line.frequency_text == line.frequency_span.text


@pytest.mark.anyio
async def test_line_without_a_strength_yields_nothing() -> None:
    """A dose is never guessed; the line is simply not extracted."""
    document = _document("VALLEY FAMILY PHARMACY", "Take 1 tablet once daily")
    assert await RuleBasedExtractor().extract(document) == ()


@pytest.mark.anyio
async def test_missing_sig_line_leaves_frequency_unset() -> None:
    """No instruction line means no frequency, not a default one."""
    document = _document("GABAPENTIN 300 MG CAPSULE")
    (line,) = await RuleBasedExtractor().extract(document)
    assert line.frequency_text is None
    assert line.frequency_span is None


@pytest.mark.anyio
async def test_unrecognised_frequency_leaves_it_unset() -> None:
    """An instruction the lexicon does not cover is not paraphrased."""
    document = _document("LOSARTAN 50 MG TABLET", "Take 1 tablet when your doctor says")
    (line,) = await RuleBasedExtractor().extract(document)
    assert line.frequency_text is None


@pytest.mark.anyio
async def test_multiple_medications_are_kept_in_reading_order() -> None:
    """Each drug line pairs with the sig line that follows it."""
    document = _document(
        "METFORMIN HCL 500 MG TABLET",
        "Take 1 tablet by mouth twice daily",
        "LISINOPRIL 10 MG TABLET",
        "Take 1 tablet by mouth once daily",
    )
    lines = await RuleBasedExtractor().extract(document)
    assert [line.drug_text for line in lines] == ["METFORMIN HCL", "LISINOPRIL"]
    assert [line.frequency_text for line in lines] == ["twice daily", "once daily"]


@pytest.mark.anyio
async def test_sig_line_is_never_read_as_a_drug() -> None:
    """A sig line has no strength pair and opens with an administration verb."""
    document = _document("Take 2 tablets by mouth once daily")
    assert await RuleBasedExtractor().extract(document) == ()


@pytest.mark.anyio
async def test_blank_page_produces_no_medications() -> None:
    """An empty page contributes nothing."""
    document = OcrDocument(
        document_id="f" * 64,
        media_type="image/png",
        pages=(OcrPage(page_number=1, width_px=10, height_px=10, tokens=()),),
    )
    assert await RuleBasedExtractor().extract(document) == ()
