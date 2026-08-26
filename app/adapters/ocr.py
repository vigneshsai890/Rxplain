"""OCR access behind a Protocol, with a fixture-replaying implementation.

``OcrClient`` is the seam a hosted OCR provider plugs into. The implementation
shipped here, :class:`DeterministicFakeOcr`, replays committed fixtures keyed by
the SHA-256 of the uploaded bytes. That makes the whole pipeline reproducible
offline and byte-identical from run to run, which is what the golden tests
assert.

No cloud OCR implementation is included: this deployment has no Google Cloud
credentials, and shipping an adapter that cannot be executed would leave
untested code on the safety path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import anyio.to_thread
import structlog

from app.domain.ingest import OcrDocument, UploadedFile

_LOGGER = structlog.stdlib.get_logger("app.adapters.ocr")


def _read_if_present(path: Path) -> bytes | None:
    """Return the file's bytes, or ``None`` when it does not exist."""
    try:
        return path.read_bytes()
    except FileNotFoundError:
        return None


class OcrError(Exception):
    """Base class for OCR adapter failures."""


class FixtureNotFoundError(OcrError):
    """No committed fixture matches the digest of the uploaded bytes."""


@runtime_checkable
class OcrClient(Protocol):
    """Turns uploaded document bytes into a normalised :class:`OcrDocument`."""

    async def analyze(self, uploaded: UploadedFile, payload: bytes) -> OcrDocument:
        """Recognise text in ``payload``.

        Args:
            uploaded: Validated metadata for the payload.
            payload: The raw document bytes.

        Returns:
            The recognised document, carrying ``uploaded.document_id``.
        """
        ...


class DeterministicFakeOcr:
    """Replays committed OCR fixtures, selected by upload digest.

    Fixtures live one-per-document as ``<sha256>.json`` and are the canonical
    serialisation of :class:`OcrDocument`, so a change to the model's field set
    or order breaks the golden tests immediately.
    """

    def __init__(self, fixture_dir: Path) -> None:
        """Bind the replayer to a directory of fixtures."""
        self._fixture_dir = fixture_dir

    def fixture_path(self, document_id: str) -> Path:
        """Return the fixture path for a document digest."""
        return self._fixture_dir / f"{document_id}.json"

    def known_document_ids(self) -> tuple[str, ...]:
        """Return the digests this replayer can serve, sorted."""
        return tuple(sorted(path.stem for path in self._fixture_dir.glob("*.json")))

    async def analyze(self, uploaded: UploadedFile, payload: bytes) -> OcrDocument:
        """Load the fixture whose name matches ``uploaded.document_id``.

        Args:
            uploaded: Validated metadata for the payload.
            payload: Unused; the digest in ``uploaded`` selects the fixture.

        Returns:
            The recognised document.

        Raises:
            FixtureNotFoundError: If no fixture matches the digest.
        """
        del payload  # the digest is the lookup key
        path = self.fixture_path(uploaded.document_id)
        # Disk reads block, so they run in a worker thread rather than stalling
        # the event loop while other requests are in flight.
        raw = await anyio.to_thread.run_sync(_read_if_present, path)
        if raw is None:
            _LOGGER.warning("ocr_fixture_missing", document_id=uploaded.document_id)
            raise FixtureNotFoundError(f"no OCR fixture for document {uploaded.document_id}")
        return OcrDocument.model_validate_json(raw)
