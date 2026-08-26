"""Shared pytest fixtures."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.adapters.ocr import DeterministicFakeOcr
from app.config import Settings, get_settings
from app.main import create_app
from app.web import ratelimit

REPO_ROOT = Path(__file__).resolve().parents[1]
PRESCRIPTION_DIR = REPO_ROOT / "tests" / "golden" / "prescriptions"
OCR_FIXTURE_DIR = REPO_ROOT / "tests" / "golden" / "ocr"


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Strip every settings env var so a developer's shell cannot alter results.

    Derived from the model's own field list, so a new setting is isolated
    automatically instead of leaking into tests until someone notices.
    """
    for name in Settings.model_fields:
        monkeypatch.delenv(name.upper(), raising=False)
    get_settings.cache_clear()
    # The rate limiter is a process-wide singleton, so its counters would
    # otherwise accumulate across tests and throttle unrelated ones.
    ratelimit.reset()
    yield
    get_settings.cache_clear()
    ratelimit.reset()


@pytest.fixture(scope="session")
def anyio_backend() -> str:
    """Run async tests on asyncio only; RxPlain does not target trio."""
    return "asyncio"


@pytest.fixture
def settings() -> Settings:
    """Test settings: defaults everywhere except the environment marker."""
    return Settings(app_env="test", log_level="INFO")


@pytest.fixture
def fake_ocr() -> DeterministicFakeOcr:
    """OCR client replaying the committed golden fixtures."""
    return DeterministicFakeOcr(OCR_FIXTURE_DIR)


@pytest.fixture
def prescription_bytes() -> dict[str, bytes]:
    """Committed prescription files, keyed by stem (e.g. ``clean_three_drugs``)."""
    return {path.stem: path.read_bytes() for path in sorted(PRESCRIPTION_DIR.iterdir())}


@pytest.fixture
def prescription_digests(prescription_bytes: dict[str, bytes]) -> dict[str, str]:
    """SHA-256 of each committed prescription file, keyed by stem."""
    return {
        stem: hashlib.sha256(payload).hexdigest() for stem, payload in prescription_bytes.items()
    }


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    """A test client bound to a freshly built application."""
    with TestClient(create_app(settings)) as test_client:
        yield test_client
