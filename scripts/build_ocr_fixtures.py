"""Render the committed prescription fixtures and their OCR documents.

Run from the repository root::

    .venv/bin/python scripts/build_ocr_fixtures.py

Both the rendered image/PDF and the matching ``OcrDocument`` JSON are committed.
CI never re-renders: the JSON fixture is keyed by the SHA-256 of the committed
file bytes, so a Pillow or zlib upgrade cannot invalidate the golden tests.

Bounding boxes come from the renderer's own text metrics, so a box really does
frame the text it claims to. That matters because the result page crops these
boxes to show a patient the snippet behind each fact.

Every image carries a visible SAMPLE banner. These are test fixtures, not
prescriptions, and must not be mistaken for one.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from app.domain.ingest import BoundingBox, OcrDocument, OcrPage, OcrToken

REPO_ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = REPO_ROOT / "tests" / "golden" / "prescriptions"
FIXTURE_DIR = REPO_ROOT / "tests" / "golden" / "ocr"

_FONT_CANDIDATES = (
    Path("/System/Library/Fonts/Supplemental/Courier New.ttf"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"),
    Path("/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf"),
)

PAGE_WIDTH = 1000
PAGE_HEIGHT = 620
MARGIN_X = 48
LINE_HEIGHT = 44
BODY_TOP = 150
FONT_SIZE = 26
BANNER_FONT_SIZE = 20


@dataclass(frozen=True)
class Line:
    """One rendered line of text and the OCR confidence of its words.

    ``confidences`` is indexed per whitespace-separated word. A single value is
    applied to every word on the line.
    """

    text: str
    confidences: tuple[float, ...]

    def confidence_for(self, index: int) -> float:
        """Return the confidence for the word at ``index``."""
        if len(self.confidences) == 1:
            return self.confidences[0]
        return self.confidences[index]


@dataclass(frozen=True)
class Fixture:
    """A prescription document to render."""

    name: str
    suffix: str
    header: tuple[str, ...]
    lines: tuple[Line, ...]
    extra_blank_page: bool = False


HIGH = (0.97,)
GOOD = (0.93,)

FIXTURES: tuple[Fixture, ...] = (
    Fixture(
        name="clean_three_drugs",
        suffix=".png",
        header=("VALLEY FAMILY PHARMACY", "Rx #4471902   Date: 2026-03-14"),
        lines=(
            Line("METFORMIN HCL 500 MG TABLET", HIGH),
            Line("Take 1 tablet by mouth twice daily with meals", HIGH),
            Line("LISINOPRIL 10 MG TABLET", HIGH),
            Line("Take 1 tablet by mouth once daily", (0.96,)),
            Line("ATORVASTATIN CALCIUM 20 MG TABLET", (0.95,)),
            Line("Take 1 tablet by mouth at bedtime", HIGH),
        ),
    ),
    Fixture(
        name="smudged_two_drugs",
        suffix=".png",
        header=("RIVERSIDE CLINIC PHARMACY", "Rx #8830155   Date: 2026-02-02"),
        lines=(
            # Mixed per-word confidence: the strength tokens scan poorly, which
            # exercises per-token flagging without failing the document gate.
            Line(
                "AMOXICILLIN 500 MG CAPSULE",
                (0.94, 0.58, 0.91, 0.92),
            ),
            Line("Take 1 capsule by mouth three times daily", (0.88,)),
            Line("IBUPROFEN 400 MG TABLET", (0.92, 0.91, 0.90, 0.93)),
            Line("Take 1 tablet by mouth every 6 hours as needed", (0.86,)),
        ),
    ),
    Fixture(
        name="two_drugs_pdf",
        suffix=".pdf",
        header=("NORTHGATE HOSPITAL OUTPATIENT", "Rx #2210488   Date: 2026-01-19"),
        lines=(
            Line("LEVOTHYROXINE SODIUM 50 MCG TABLET", GOOD),
            Line("Take 1 tablet by mouth every morning", GOOD),
            Line("OMEPRAZOLE 20 MG CAPSULE", (0.95,)),
            Line("Take 1 capsule by mouth once daily", (0.94,)),
        ),
    ),
    Fixture(
        name="illegible_scan",
        suffix=".png",
        header=("COUNTY HEALTH PHARMACY", "Rx #5590021   Date: 2026-04-08"),
        lines=(
            # Mean confidence lands below 0.80, so the document gate rejects it.
            Line("SERTRALINE 50 MG TABLET", (0.55, 0.61, 0.48, 0.72)),
            Line("Take 1 tablet by mouth once daily", (0.66,)),
            Line("GABAPENTIN 300 MG CAPSULE", (0.71, 0.58, 0.63, 0.69)),
        ),
    ),
    Fixture(
        name="unknown_drug_and_blank_page",
        suffix=".png",
        header=("HARBOUR STREET PHARMACY", "Rx #7712340   Date: 2026-05-21"),
        lines=(
            Line("AMLODIPINE BESYLATE 5 MG TABLET", HIGH),
            Line("Take 1 tablet by mouth once daily", HIGH),
            # Not a drug RxNorm can resolve: exercises the unverified path.
            Line("ZYNTRAFEX 250 MG TABLET", (0.95, 0.94, 0.93, 0.95)),
            Line("Take 1 tablet by mouth twice daily", (0.96,)),
        ),
        extra_blank_page=True,
    ),
)


def _load_font(size: int) -> ImageFont.FreeTypeFont:
    """Load the first available monospace font.

    Raises:
        FileNotFoundError: If no candidate font exists on this machine.
    """
    for candidate in _FONT_CANDIDATES:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size)
    raise FileNotFoundError(f"none of the candidate fonts exist: {_FONT_CANDIDATES}")


def _render(fixture: Fixture) -> tuple[bytes, tuple[OcrToken, ...]]:
    """Render one fixture, returning its file bytes and its OCR tokens."""
    font = _load_font(FONT_SIZE)
    banner_font = _load_font(BANNER_FONT_SIZE)
    image = Image.new("RGB", (PAGE_WIDTH, PAGE_HEIGHT), "white")
    draw = ImageDraw.Draw(image)

    draw.rectangle((0, 0, PAGE_WIDTH - 1, 40), fill="#7a1d1d")
    draw.text((MARGIN_X, 10), "SAMPLE - NOT A REAL PRESCRIPTION", font=banner_font, fill="white")

    for index, heading in enumerate(fixture.header):
        draw.text((MARGIN_X, 62 + index * 30), heading, font=font, fill="#1a1a1a")
    draw.line((MARGIN_X, BODY_TOP - 18, PAGE_WIDTH - MARGIN_X, BODY_TOP - 18), fill="#999999")

    tokens: list[OcrToken] = []
    counter = 0
    for line_index, line in enumerate(fixture.lines):
        y = BODY_TOP + line_index * LINE_HEIGHT
        x = float(MARGIN_X)
        for word_index, word in enumerate(line.text.split(" ")):
            left, top, right, bottom = draw.textbbox((x, y), word, font=font)
            draw.text((x, y), word, font=font, fill="#111111")
            counter += 1
            tokens.append(
                OcrToken(
                    token_id=f"p1-t{counter:03d}",
                    text=word,
                    confidence=line.confidence_for(word_index),
                    box=BoundingBox(
                        left=left / PAGE_WIDTH,
                        top=top / PAGE_HEIGHT,
                        right=right / PAGE_WIDTH,
                        bottom=bottom / PAGE_HEIGHT,
                    ),
                )
            )
            x = right + font.getlength(" ")

    target = IMAGE_DIR / f"{fixture.name}{fixture.suffix}"
    if fixture.suffix == ".pdf":
        image.save(target, format="PDF", resolution=150.0)
    else:
        image.save(target, format="PNG", optimize=True)
    return target.read_bytes(), tuple(tokens)


def build() -> list[tuple[str, str, int]]:
    """Render every fixture and write its OCR document.

    Returns:
        One ``(name, document_id, token_count)`` row per fixture.
    """
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    summary: list[tuple[str, str, int]] = []

    for fixture in FIXTURES:
        payload, tokens = _render(fixture)
        document_id = hashlib.sha256(payload).hexdigest()
        pages = [
            OcrPage(
                page_number=1,
                width_px=PAGE_WIDTH,
                height_px=PAGE_HEIGHT,
                tokens=tokens,
            )
        ]
        if fixture.extra_blank_page:
            pages.append(
                OcrPage(page_number=2, width_px=PAGE_WIDTH, height_px=PAGE_HEIGHT, tokens=())
            )

        document = OcrDocument(
            document_id=document_id,
            media_type="application/pdf" if fixture.suffix == ".pdf" else "image/png",
            pages=tuple(pages),
        )
        (FIXTURE_DIR / f"{document_id}.json").write_text(
            document.model_dump_json(), encoding="utf-8"
        )
        summary.append((fixture.name, document_id, len(tokens)))
    return summary


if __name__ == "__main__":
    for name, document_id, token_count in build():
        print(f"{name:34s} {document_id[:16]}...  {token_count:3d} tokens")
