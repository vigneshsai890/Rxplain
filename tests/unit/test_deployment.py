"""Tests for the deployment artefacts.

Vercel installs from ``requirements.txt`` and ignores ``pyproject.toml``, so the
deployed dependency set is described twice. This makes drift a failing test rather
than a difference that only shows up in production.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.config import Settings
from scripts.sync_requirements import REQUIREMENTS, render

REPO_ROOT = Path(__file__).resolve().parents[2]
VERCEL_JSON = REPO_ROOT / "vercel.json"
ENTRYPOINT = REPO_ROOT / "api" / "index.py"


def test_requirements_match_pyproject() -> None:
    """Regenerate with `python scripts/sync_requirements.py` if this fails."""
    assert REQUIREMENTS.read_text(encoding="utf-8") == render()


def test_the_vercel_entrypoint_exposes_the_asgi_app() -> None:
    """Vercel imports `app` from this module; a rename would break the deploy."""
    assert ENTRYPOINT.is_file()
    assert "from app.main import app" in ENTRYPOINT.read_text(encoding="utf-8")


def test_vercel_config_selects_the_database_free_backend() -> None:
    """Serverless has no database to connect to, so it must use the seed index."""
    config = json.loads(VERCEL_JSON.read_text(encoding="utf-8"))
    assert config["env"]["STORAGE_BACKEND"] == "memory"


def test_vercel_config_routes_everything_to_the_app() -> None:
    """A single ASGI handler serves every path, including static files."""
    config = json.loads(VERCEL_JSON.read_text(encoding="utf-8"))
    assert config["routes"] == [{"src": "/(.*)", "dest": "api/index.py"}]


def test_vercel_config_has_no_builds_functions_conflict() -> None:
    """Vercel rejects a config declaring both, so only one may be present."""
    config = json.loads(VERCEL_JSON.read_text(encoding="utf-8"))
    assert not ("builds" in config and "functions" in config)


def test_runtime_data_is_bundled() -> None:
    """The Python builder bundles imports, not data files read at runtime.

    Templates, static assets, OCR fixtures and the RxNorm seed are all opened at
    request time, so each must be declared or the deployment builds and then 500s.
    """
    config = json.loads(VERCEL_JSON.read_text(encoding="utf-8"))
    included = config["builds"][0]["config"]["includeFiles"]
    for required in (
        "app/web/templates/**",
        "app/web/static/**",
        "tests/golden/ocr/**",
        "tests/golden/prescriptions/**",
        "migrations/seed/**",
    ):
        assert required in included


def test_the_entrypoint_documents_the_duration_constraint() -> None:
    """A cold vision call measured 46.7s against a 60s ceiling; that is recorded."""
    body = ENTRYPOINT.read_text(encoding="utf-8")
    assert "46.7s" in body
    assert "60s" in body


def test_vercel_data_paths_are_relative_so_they_resolve_in_the_bundle() -> None:
    """Settings anchors relative paths to the install root, which Vercel needs."""
    config = json.loads(VERCEL_JSON.read_text(encoding="utf-8"))
    for key in ("OCR_FIXTURE_DIR", "SAMPLE_PRESCRIPTION_DIR", "RXNORM_SEED_PATH"):
        assert not config["env"][key].startswith("/")


def test_the_memory_backend_is_a_declared_option() -> None:
    """The config value Vercel sets must be one Settings accepts."""
    assert Settings(app_env="test", storage_backend="memory").storage_backend == "memory"
    assert Settings(app_env="test").storage_backend == "postgres"
