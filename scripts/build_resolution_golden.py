"""Regenerate the committed golden resolution files.

Run from the repository root, with Postgres up and the RxNorm seed loaded::

    docker compose up -d db
    .venv/bin/python scripts/load_rxnorm.py load-seed
    .venv/bin/python scripts/build_resolution_golden.py

Writes one file per committed prescription into ``tests/golden/resolution/``,
holding the serialised :class:`~app.safety.verifier.PatientGuide`. The golden
test compares byte-for-byte, so any change to resolution, extraction or
verification shows up as a reviewable diff instead of a silent behaviour change.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from app.adapters.extract import RuleBasedExtractor
from app.adapters.ocr import DeterministicFakeOcr
from app.adapters.rxnorm import PostgresRxNormResolver
from app.config import Settings
from app.db import Database
from app.domain import pipeline

REPO_ROOT = Path(__file__).resolve().parents[1]
GOLDEN_DIR = REPO_ROOT / "tests" / "golden" / "resolution"


async def build() -> list[tuple[str, int, int]]:
    """Write a golden file per prescription. Returns per-file counts."""
    settings = Settings()
    database = Database(settings)
    await database.open()
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    summary: list[tuple[str, int, int]] = []
    try:
        ocr = DeterministicFakeOcr(settings.ocr_fixture_dir)
        extractor = RuleBasedExtractor()
        resolver = PostgresRxNormResolver(database, settings)

        for path in sorted(settings.sample_prescription_dir.iterdir()):
            result = await pipeline.run(
                path.read_bytes(),
                ocr=ocr,
                extractor=extractor,
                resolver=resolver,
                settings=settings,
            )
            if result.guide is None:
                body = result.gate_report.model_dump_json()
                counts = (0, 0)
            else:
                body = result.guide.model_dump_json()
                counts = (len(result.guide.verified), len(result.guide.unverified))
            (GOLDEN_DIR / f"{path.stem}.json").write_text(body, encoding="utf-8")
            summary.append((path.stem, *counts))
    finally:
        await database.close()
    return summary


if __name__ == "__main__":
    for stem, verified, unverified in asyncio.run(build()):
        print(f"{stem:34s} verified={verified} unverified={unverified}")
