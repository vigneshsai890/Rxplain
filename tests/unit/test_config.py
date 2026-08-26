"""Tests for :mod:`app.config`."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.config import Settings, get_settings

ENV_EXAMPLE = Path(__file__).resolve().parents[2] / ".env.example"


def _example_keys() -> list[str]:
    """Return the variable names declared in the committed ``.env.example``."""
    keys: list[str] = []
    for line in ENV_EXAMPLE.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        keys.append(stripped.split("=", 1)[0])
    return keys


def test_defaults_are_safe() -> None:
    """An unconfigured process gets local defaults and hardened headers."""
    settings = Settings()
    assert settings.app_env == "local"
    assert settings.log_level == "INFO"
    assert settings.port == 8080
    assert settings.content_security_policy.startswith("default-src 'self'")
    assert settings.hsts_max_age_seconds == 63_072_000
    assert settings.referrer_policy == "no-referrer"


def test_default_csp_has_no_unsafe_directives() -> None:
    """Guards against a future edit loosening the policy."""
    policy = Settings().content_security_policy
    for forbidden in ("unsafe-inline", "unsafe-eval", "default-src *", "script-src *"):
        assert forbidden not in policy


def test_default_safety_thresholds_match_the_specification() -> None:
    """The gate and traceability thresholds ship at their documented values."""
    settings = Settings()
    assert settings.ocr_min_mean_confidence == 0.80
    assert settings.ocr_min_token_confidence == 0.60
    assert settings.verbatim_min_similarity == 0.90
    assert settings.max_upload_bytes == 10 * 1024 * 1024


@pytest.mark.parametrize("value", ["0.89", "0.5", "0.0"])
def test_verbatim_similarity_cannot_be_loosened(
    value: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Operator error may only tighten verbatim matching, never weaken it."""
    monkeypatch.setenv("VERBATIM_MIN_SIMILARITY", value)
    with pytest.raises(ValidationError):
        Settings()


def test_verbatim_similarity_can_be_tightened(monkeypatch: pytest.MonkeyPatch) -> None:
    """Values above the floor are accepted."""
    monkeypatch.setenv("VERBATIM_MIN_SIMILARITY", "0.97")
    assert Settings().verbatim_min_similarity == 0.97


def test_database_url_is_not_printable() -> None:
    """The connection string embeds a password, so it must stay redacted."""
    settings = Settings()
    assert "rxplain_local_dev" not in repr(settings)
    assert "rxplain_local_dev" not in str(settings)
    assert settings.database_url.get_secret_value().startswith("postgresql://")


def test_env_example_declares_only_real_fields() -> None:
    """Every key in ``.env.example`` maps to a declared Settings field."""
    declared = set(Settings.model_fields)
    assert _example_keys(), "committed .env.example must declare variables"
    assert {key.lower() for key in _example_keys()} <= declared


def _documented_keys() -> list[str]:
    """Return every variable named in ``.env.example``, commented out or not.

    An optional secret is documented as ``# NVIDIA_API_KEY=...`` rather than being
    assigned, because assigning it an empty value would switch a code path on with
    an unusable credential. It is still documented, so it counts here.
    """
    keys: list[str] = []
    for line in ENV_EXAMPLE.read_text(encoding="utf-8").splitlines():
        stripped = line.strip().lstrip("#").strip()
        if "=" not in stripped:
            continue
        name = stripped.split("=", 1)[0].strip()
        if name.isupper() and name.replace("_", "").isalnum():
            keys.append(name)
    return keys


def test_env_example_documents_every_field() -> None:
    """A new setting must be documented, not left for someone to discover."""
    documented = {key.lower() for key in _documented_keys()}
    assert set(Settings.model_fields) <= documented


def test_optional_secrets_are_not_assigned_a_value() -> None:
    """A commented secret cannot accidentally enable a path with a bad key."""
    assert "nvidia_api_key" not in {key.lower() for key in _example_keys()}
    assert Settings().nvidia_api_key is None


def test_env_example_is_loadable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The example file parses and is a faithful transcription of the defaults.

    Comparing against ``Settings()`` rather than a hardcoded list means adding a
    setting without documenting it in ``.env.example`` fails here.
    """
    monkeypatch.chdir(tmp_path)
    defaults = Settings()
    shutil.copyfile(ENV_EXAMPLE, tmp_path / ".env")
    assert Settings() == defaults


def test_environment_overrides_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Values come from the environment, including numeric coercion."""
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("LOG_LEVEL", "WARNING")
    monkeypatch.setenv("PORT", "9001")
    settings = Settings()
    assert (settings.app_env, settings.log_level, settings.port) == ("production", "WARNING", 9001)


def test_unknown_env_file_key_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A typo in ``.env`` fails startup instead of being silently ignored."""
    (tmp_path / ".env").write_text("APP_ENV=staging\nHSTS_MAX_AGE_SECOND=1\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValidationError, match="hsts_max_age_second"):
        Settings()


@pytest.mark.parametrize("value", ["TRACE", "info", ""])
def test_invalid_log_level_is_rejected(value: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Only the four supported level names are accepted."""
    monkeypatch.setenv("LOG_LEVEL", value)
    with pytest.raises(ValidationError):
        Settings()


@pytest.mark.parametrize("value", ["0", "65536", "-1"])
def test_out_of_range_port_is_rejected(value: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Ports outside the TCP range fail validation."""
    monkeypatch.setenv("PORT", value)
    with pytest.raises(ValidationError):
        Settings()


def test_empty_security_header_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """A blank CSP would silently disable the policy, so it is invalid."""
    monkeypatch.setenv("CONTENT_SECURITY_POLICY", "")
    with pytest.raises(ValidationError):
        Settings()


def test_settings_are_immutable() -> None:
    """Configuration cannot be mutated after validation."""
    settings = Settings()
    field_name = "port"  # indirect so the frozen-field type error is not the assertion
    with pytest.raises(ValidationError):
        setattr(settings, field_name, 1234)


def test_get_settings_is_cached() -> None:
    """Repeated lookups return the identical validated object."""
    assert get_settings() is get_settings()
