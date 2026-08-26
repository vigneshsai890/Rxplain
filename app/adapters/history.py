"""Prescription history, stored in MongoDB.

This is **additive** storage. Postgres keeps the reference data that resolution
depends on - RxNorm concepts behind a ``pg_trgm`` index, and the openFDA label
cache - because the approximate-match tier of the resolution ladder is a
similarity query with no document-store equivalent. MongoDB holds only saved
history, which is a document per prescription and a natural fit.

Privacy posture, stated plainly
-------------------------------
Everything else in RxPlain stores nothing about a patient: images are held in
memory for one request, and Postgres contains only public reference data. History
is the first feature that persists anything identifying, so it is:

* **Off unless configured.** With no ``mongodb_uri`` the app uses
  :class:`NullHistoryStore` and writes nothing, which keeps the zero-retention
  default intact.
* **Opt-in per request.** Nothing is saved unless the user supplies a profile
  name with the upload.
* **Deletable in one call.** :meth:`HistoryStore.delete_all` removes every entry
  for a profile.
* **Minimal.** No image, no full-page transcription. Only the verified fields
  already shown on screen, plus a count of what could not be verified.

A profile name is whatever the user types, so it is treated as identifying: it is
never logged. Log lines carry a salted hash prefix instead, which is enough to
correlate requests without recording who they belong to.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import Protocol, runtime_checkable

import structlog
from pydantic import BaseModel, ConfigDict, Field
from pymongo import DESCENDING, AsyncMongoClient
from pymongo.errors import PyMongoError

from app.config import Settings
from app.domain.guide import RenderedGuide

_LOGGER = structlog.stdlib.get_logger("app.adapters.history")

_STRICT = ConfigDict(strict=True, frozen=True, extra="forbid")

COLLECTION = "prescriptions"
MAX_PROFILE_LENGTH = 60


class HistoryError(RuntimeError):
    """The history store could not complete an operation."""


def profile_fingerprint(profile: str) -> str:
    """Return a short, non-reversing fingerprint of a profile name for logs."""
    return hashlib.sha256(profile.strip().casefold().encode("utf-8")).hexdigest()[:12]


class HistoryMedication(BaseModel):
    """One verified medication, as saved."""

    model_config = _STRICT

    display_name: str = Field(min_length=1)
    rxcui: str = Field(min_length=1)
    dose_text: str | None = None
    frequency_text: str | None = None


class HistoryEntry(BaseModel):
    """One saved prescription for one profile."""

    model_config = _STRICT

    profile: str = Field(min_length=1, max_length=MAX_PROFILE_LENGTH)
    document_id: str = Field(min_length=1)
    saved_at: datetime
    medications: tuple[HistoryMedication, ...]
    unverified_count: int = Field(ge=0)

    @property
    def medication_count(self) -> int:
        """How many medications were confirmed on this prescription."""
        return len(self.medications)


def entry_from_guide(profile: str, guide: RenderedGuide) -> HistoryEntry:
    """Build a history entry from a rendered guide.

    Only fields already displayed to the user are carried across, so saving adds
    no information the user has not already seen.
    """
    return HistoryEntry(
        profile=profile.strip(),
        document_id=guide.document_id,
        saved_at=datetime.now(tz=UTC),
        medications=tuple(
            HistoryMedication(
                display_name=medication.display_name,
                rxcui=medication.rxcui,
                dose_text=next(
                    (
                        sentence.source_text
                        for sentence in medication.sentences
                        if sentence.provenance.source == "ocr_span"
                    ),
                    None,
                ),
                frequency_text=next(
                    (
                        sentence.source_text
                        for sentence in reversed(medication.sentences)
                        if sentence.provenance.source == "ocr_span"
                    ),
                    None,
                ),
            )
            for medication in guide.medications
        ),
        unverified_count=len(guide.unverified),
    )


@runtime_checkable
class HistoryStore(Protocol):
    """Persists and retrieves saved prescriptions per profile."""

    @property
    def enabled(self) -> bool:
        """Whether this store actually persists anything."""
        ...

    async def save(self, entry: HistoryEntry) -> None:
        """Persist one entry."""
        ...

    async def list_for(self, profile: str, limit: int = 50) -> tuple[HistoryEntry, ...]:
        """Return a profile's entries, newest first."""
        ...

    async def delete_all(self, profile: str) -> int:
        """Delete every entry for a profile, returning how many were removed."""
        ...

    async def close(self) -> None:
        """Release any connection held."""
        ...


class NullHistoryStore:
    """Stores nothing. The default, so zero retention needs no configuration."""

    @property
    def enabled(self) -> bool:
        """Always ``False``."""
        return False

    async def save(self, entry: HistoryEntry) -> None:
        """Discard the entry."""
        del entry

    async def list_for(self, profile: str, limit: int = 50) -> tuple[HistoryEntry, ...]:
        """Return nothing."""
        del profile, limit
        return ()

    async def delete_all(self, profile: str) -> int:
        """Report that nothing was deleted."""
        del profile
        return 0

    async def close(self) -> None:
        """Nothing to release."""


class MongoHistoryStore:
    """History stored in MongoDB, one document per saved prescription.

    The entry is persisted as a JSON string alongside the queryable fields.
    Reading it back through ``model_validate_json`` keeps the strict models
    strict: a driver decodes JSON arrays to Python lists, which strict validation
    rejects for a tuple field.
    """

    def __init__(self, settings: Settings) -> None:
        """Open a client against the configured cluster.

        Raises:
            ValueError: If no MongoDB URI is configured.
        """
        if settings.mongodb_uri is None:
            raise ValueError("MongoHistoryStore requires mongodb_uri to be set")
        # The document type is spelled out because the driver's default is
        # parameterised on a dict of Any, which strict type checking rejects.
        self._client: AsyncMongoClient[dict[str, object]] = AsyncMongoClient(
            settings.mongodb_uri.get_secret_value(),
            serverSelectionTimeoutMS=int(settings.mongodb_timeout_seconds * 1000),
            tz_aware=True,
        )
        self._collection = self._client[settings.mongodb_database][COLLECTION]

    @property
    def enabled(self) -> bool:
        """Always ``True``."""
        return True

    async def ensure_indexes(self) -> None:
        """Create the index that the timeline query needs.

        Raises:
            HistoryError: If the cluster is unreachable.
        """
        try:
            await self._collection.create_index([("profile_key", 1), ("saved_at", DESCENDING)])
        except PyMongoError as exc:
            raise HistoryError("could not prepare the history collection") from exc

    async def save(self, entry: HistoryEntry) -> None:
        """Persist one entry.

        Raises:
            HistoryError: If the write fails.
        """
        try:
            await self._collection.insert_one(
                {
                    "profile_key": entry.profile.casefold(),
                    "document_id": entry.document_id,
                    "saved_at": entry.saved_at,
                    "payload": entry.model_dump_json(),
                }
            )
        except PyMongoError as exc:
            _LOGGER.warning("history_save_failed", error=type(exc).__name__)
            raise HistoryError("could not save this prescription") from exc
        _LOGGER.info(
            "history_saved",
            profile=profile_fingerprint(entry.profile),
            medications=entry.medication_count,
        )

    async def list_for(self, profile: str, limit: int = 50) -> tuple[HistoryEntry, ...]:
        """Return a profile's entries, newest first.

        Raises:
            HistoryError: If the read fails.
        """
        try:
            cursor = (
                self._collection.find({"profile_key": profile.strip().casefold()})
                .sort("saved_at", DESCENDING)
                .limit(max(limit, 1))
            )
            documents = await cursor.to_list(length=max(limit, 1))
        except PyMongoError as exc:
            raise HistoryError("could not read this profile's history") from exc

        entries: list[HistoryEntry] = []
        for document in documents:
            payload = document.get("payload")
            if isinstance(payload, str):
                entries.append(HistoryEntry.model_validate_json(payload))
        return tuple(entries)

    async def delete_all(self, profile: str) -> int:
        """Delete every entry for a profile.

        Raises:
            HistoryError: If the delete fails.
        """
        try:
            result = await self._collection.delete_many({"profile_key": profile.strip().casefold()})
        except PyMongoError as exc:
            raise HistoryError("could not delete this profile's history") from exc
        deleted = int(result.deleted_count)
        _LOGGER.info("history_deleted", profile=profile_fingerprint(profile), deleted=deleted)
        return deleted

    async def close(self) -> None:
        """Close the client."""
        await self._client.close()
