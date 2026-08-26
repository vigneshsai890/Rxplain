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
    assert settings.content_security_policy == "default-src 'self'"
    assert settings.hsts_max_age_seconds == 63_072_000
    assert settings.referrer_policy == "no-referrer"


def test_env_example_declares_only_real_fields() -> None:
    """Every key in ``.env.example`` maps to a declared Settings field."""
    declared = set(Settings.model_fields)
    assert _example_keys(), "committed .env.example must declare variables"
    assert {key.lower() for key in _example_keys()} <= declared


def test_env_example_is_loadable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The committed example file validates as-is when copied to ``.env``."""
    shutil.copyfile(ENV_EXAMPLE, tmp_path / ".env")
    monkeypatch.chdir(tmp_path)
    assert Settings() == Settings(
        app_env="local",
        log_level="INFO",
        port=8080,
        content_security_policy="default-src 'self'",
        hsts_max_age_seconds=63_072_000,
        referrer_policy="no-referrer",
    )


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
