"""Application configuration.

Every value is read from the process environment, optionally seeded by a local
``.env`` file that is gitignored. ``extra="forbid"`` means an unrecognised key
in ``.env`` aborts startup rather than being silently dropped, so a typo can
never leave a safety threshold sitting at its default.

Settings deliberately do **not** run in Pydantic strict mode: environment
variables arrive as strings and must be coerced. Correctness is enforced
instead by ``Literal`` domains and numeric bounds, which reject bad input just
as firmly. Strict mode is reserved for in-process model boundaries.

Safety thresholds live here rather than in code so they can be tightened
without a code change. Their bounds encode the floors the safety model relies
on: ``verbatim_min_similarity`` cannot be configured below 0.9, so operator
error can only ever make verification stricter.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

AppEnv = Literal["local", "test", "staging", "production"]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR"]

_REPO_ROOT = Path(__file__).resolve().parents[1]


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

    # --- response hardening -------------------------------------------------
    content_security_policy: str = Field(
        default="default-src 'self'; img-src 'self' data:; form-action 'self'",
        min_length=1,
    )
    hsts_max_age_seconds: int = Field(default=63_072_000, ge=0)
    referrer_policy: str = Field(default="no-referrer", min_length=1)

    # --- storage ------------------------------------------------------------
    # "postgres" uses the database for RxNorm resolution and the openFDA cache.
    # "memory" serves both from the committed seed in process. The seed is 310
    # concepts, so the index fits in memory and every tier stays a set operation;
    # app/adapters/rxnorm_local.py reproduces pg_trgm exactly, verified against a
    # live server. "memory" is what a serverless or database-less deployment uses.
    storage_backend: Literal["postgres", "memory"] = "postgres"

    database_url: SecretStr = SecretStr(
        "postgresql://rxplain:rxplain_local_dev@127.0.0.1:55432/rxplain"
    )
    database_pool_min_size: int = Field(default=1, ge=1, le=32)
    database_pool_max_size: int = Field(default=8, ge=1, le=64)

    # --- ingestion ----------------------------------------------------------
    max_upload_bytes: int = Field(default=10 * 1024 * 1024, ge=1, le=64 * 1024 * 1024)

    # Golden fixtures are the OCR source of truth for this deployment, so they
    # are runtime data, not just test data, and ship inside the image.
    ocr_fixture_dir: Path = Path("tests/golden/ocr")
    sample_prescription_dir: Path = Path("tests/golden/prescriptions")

    # --- data locations -----------------------------------------------------
    migrations_dir: Path = Path("migrations")
    rxnorm_seed_path: Path = Path("migrations/seed/rxnorm_seed.json")

    # --- safety gates -------------------------------------------------------
    ocr_min_mean_confidence: float = Field(default=0.80, ge=0.0, le=1.0)
    ocr_min_token_confidence: float = Field(default=0.60, ge=0.0, le=1.0)

    # Floor of 0.9 mirrors the invariant baked into safety.verifier.TracedText:
    # configuration can tighten verbatim matching but never loosen it.
    verbatim_min_similarity: float = Field(default=0.90, ge=0.90, le=1.0)

    # Trigram similarity required by the RxNorm approximate-match tier.
    rxnorm_min_similarity: float = Field(default=0.85, ge=0.50, le=1.0)

    # --- sign-in ------------------------------------------------------------
    # Both unset means sign-in is off and no session cookie is ever issued, which
    # keeps the anonymous, zero-retention path as the default.
    google_oauth_client_id: str | None = None
    google_oauth_client_secret: SecretStr | None = None

    # --- rate limiting ------------------------------------------------------
    # /guide triggers a transcription that can take ~45s, so an unauthenticated
    # caller can amplify cost heavily. These are per-client-IP.
    rate_limit_guide: str = Field(default="10/minute", min_length=3)
    rate_limit_default: str = Field(default="120/minute", min_length=3)
    rate_limit_enabled: bool = True

    # --- prescription history (additive storage) ----------------------------
    # Unset means nothing is persisted about a patient, which is the default.
    # Postgres keeps reference data; this holds only saved history.
    mongodb_uri: SecretStr | None = None
    mongodb_database: str = Field(default="rxplain", min_length=1)
    mongodb_timeout_seconds: float = Field(default=15.0, gt=0.0, le=120.0)

    # --- transcription ------------------------------------------------------
    # When no key is set the fixture replayer is used, so the app still runs
    # offline. Setting a key switches the OCR seam to the hosted vision model.
    nvidia_api_key: SecretStr | None = None
    nvidia_base_url: str = Field(default="https://integrate.api.nvidia.com/v1", min_length=1)
    nvidia_vision_model: str = Field(default="meta/llama-3.2-11b-vision-instruct", min_length=1)
    nvidia_text_model: str = Field(default="mistralai/mistral-nemotron", min_length=1)
    # Observed latency for an 11B vision call is ~45s, so the ceiling is generous.
    nvidia_timeout_seconds: float = Field(default=120.0, gt=0.0, le=300.0)

    # Confidence assigned to vision-transcribed words. These are not measured
    # values: the model marks words it is unsure of, and the unsure figure sits
    # below ocr_min_token_confidence so the existing per-token gate flags them.
    vision_confident_token_confidence: float = Field(default=0.90, ge=0.0, le=1.0)
    vision_unsure_token_confidence: float = Field(default=0.45, ge=0.0, le=1.0)

    # --- openFDA ------------------------------------------------------------
    openfda_base_url: str = Field(default="https://api.fda.gov", min_length=1)
    openfda_timeout_seconds: float = Field(default=20.0, gt=0.0, le=120.0)
    openfda_cache_ttl_hours: int = Field(default=168, ge=1, le=8760)
    openfda_enabled: bool = True

    @field_validator(
        "ocr_fixture_dir",
        "sample_prescription_dir",
        "migrations_dir",
        "rxnorm_seed_path",
        mode="after",
    )
    @classmethod
    def _resolve_against_repo_root(cls, value: Path) -> Path:
        """Anchor relative data paths so the working directory cannot matter.

        The container runs from ``/srv`` and Cloud Run may set any working
        directory, so a relative path in configuration is resolved against the
        installation root rather than the process CWD.
        """
        return value if value.is_absolute() else _REPO_ROOT / value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton.

    Cached so that configuration is parsed and validated exactly once. Tests
    call ``get_settings.cache_clear()`` to observe a changed environment.
    """
    return Settings()
