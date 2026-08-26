"""Golden tests: the committed fixtures must round-trip byte-for-byte.

Each fixture file *is* the canonical serialisation of an :class:`OcrDocument`.
Re-serialising a parsed fixture and comparing bytes therefore catches any change
to the model's field set, field order, or number formatting - all of which would
silently invalidate stored provenance refs.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from app.adapters.ocr import DeterministicFakeOcr
from app.config import Settings
from app.domain.ingest import OcrDocument, ingest
from app.safety.gates import evaluate_document

FIXTURE_DIR = Path(__file__).resolve().parent / "ocr"
PRESCRIPTION_DIR = Path(__file__).resolve().parent / "prescriptions"

FIXTURE_FILES = sorted(FIXTURE_DIR.glob("*.json"))
PRESCRIPTION_FILES = sorted(PRESCRIPTION_DIR.iterdir())

# Expected gate outcome per committed prescription. Locked down so a threshold
# change cannot quietly turn a rejected scan into an accepted one.
EXPECTED_GATE_OUTCOMES = {
    "clean_three_drugs": (True, 0, None),
    "smudged_two_drugs": (True, 1, None),
    "two_drugs_pdf": (True, 0, None),
    "illegible_scan": (False, 3, "mean_confidence_below_threshold"),
    "unknown_drug_and_blank_page": (True, 0, None),
}


def test_fixture_set_is_complete() -> None:
    """One OCR fixture per committed prescription, and at least five of each."""
    assert len(PRESCRIPTION_FILES) >= 5
    assert len(FIXTURE_FILES) == len(PRESCRIPTION_FILES)


@pytest.mark.parametrize("path", FIXTURE_FILES, ids=lambda p: p.stem[:12])
def test_fixture_round_trips_byte_for_byte(path: Path) -> None:
    """Parsing and re-serialising a fixture reproduces the exact same bytes."""
    raw = path.read_bytes()
    document = OcrDocument.model_validate_json(raw)
    assert document.model_dump_json().encode("utf-8") == raw


@pytest.mark.parametrize("path", FIXTURE_FILES, ids=lambda p: p.stem[:12])
def test_fixture_name_matches_its_document_id(path: Path) -> None:
    """The filename is the lookup key, so it must equal the document ID."""
    assert OcrDocument.model_validate_json(path.read_bytes()).document_id == path.stem


@pytest.mark.parametrize("path", PRESCRIPTION_FILES, ids=lambda p: p.stem)
def test_prescription_digest_has_a_fixture(path: Path) -> None:
    """Every committed prescription image can be replayed by digest."""
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert (FIXTURE_DIR / f"{digest}.json").is_file()


@pytest.mark.anyio
@pytest.mark.parametrize("path", PRESCRIPTION_FILES, ids=lambda p: p.stem)
async def test_pipeline_output_is_byte_identical_to_the_fixture(
    path: Path, fake_ocr: DeterministicFakeOcr
) -> None:
    """Running the pipeline on the image reproduces the committed JSON exactly."""
    payload = path.read_bytes()
    document = await ingest(payload, ocr=fake_ocr, max_bytes=10 * 1024 * 1024)
    expected = (FIXTURE_DIR / f"{document.document_id}.json").read_bytes()
    assert document.model_dump_json().encode("utf-8") == expected


@pytest.mark.anyio
@pytest.mark.parametrize("path", PRESCRIPTION_FILES, ids=lambda p: p.stem)
async def test_pipeline_is_deterministic_across_runs(
    path: Path, fake_ocr: DeterministicFakeOcr
) -> None:
    """Two runs over the same bytes produce identical output."""
    payload = path.read_bytes()
    first = await ingest(payload, ocr=fake_ocr, max_bytes=10 * 1024 * 1024)
    second = await ingest(payload, ocr=fake_ocr, max_bytes=10 * 1024 * 1024)
    assert first.model_dump_json() == second.model_dump_json()


@pytest.mark.anyio
@pytest.mark.parametrize("path", PRESCRIPTION_FILES, ids=lambda p: p.stem)
async def test_gate_outcome_is_locked_down(
    path: Path, fake_ocr: DeterministicFakeOcr, settings: Settings
) -> None:
    """Each fixture's accept/reject verdict and flag count are pinned."""
    document = await ingest(path.read_bytes(), ocr=fake_ocr, max_bytes=10 * 1024 * 1024)
    report = evaluate_document(document, settings)
    expected_accepted, expected_flags, expected_reason = EXPECTED_GATE_OUTCOMES[path.stem]
    assert report.accepted is expected_accepted
    assert len(report.low_confidence_token_ids) == expected_flags
    assert report.rejection_reason == expected_reason
