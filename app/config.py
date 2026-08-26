"""Application configuration.

Every value is read from the process environment, optionally seeded by a local
``.env`` file that is gitignored. ``extra="forbid"`` means an unrecognised key
in ``.env`` aborts startup rather than being silently dropped, so a typo can
never leave a safety threshold sitting at its default.

Settings deliberately do **not** run in Pydantic strict mode: environment
variables arrive as strings and must be coerced. Correctness is enforced
instead by ``Literal`` domains and numeric bounds, which reject bad input just
as firmly. Strict mode is reserved for in-process model boundaries.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

AppEnv = Literal["local", "test", "staging", "production"]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR"]


class Settings(BaseSettings):
    """Runtime configuration for the RxPlain service."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="forbid",
        frozen=True,
        validate_default=True,
    )

    app_env: AppEnv = "local"
    log_level: LogLevel = "INFO"
    port: int = Field(default=8080, ge=1, le=65535)

    content_security_policy: str = Field(default="default-src 'self'", min_length=1)
    hsts_max_age_seconds: int = Field(default=63_072_000, ge=0)
    referrer_policy: str = Field(default="no-referrer", min_length=1)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton.

    Cached so that configuration is parsed and validated exactly once. Tests
    call ``get_settings.cache_clear()`` to observe a changed environment.
    """
    return Settings()
