"""Integration tests for the MongoDB history store.

Skips when no cluster is configured, which is the default and what CI does, so the
unit suite stays runnable without network access.

Each test namespaces its own profile and deletes it afterwards, so a run leaves
the cluster as it found it and two concurrent runs cannot collide.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime

import pytest

from app.adapters.history import (
    HistoryEntry,
    HistoryMedication,
    HistoryStore,
    MongoHistoryStore,
)
from app.config import Settings

pytestmark = pytest.mark.anyio


@pytest.fixture
async def store(settings: Settings) -> AsyncIterator[MongoHistoryStore]:
    """An open store with indexes ready, or skip the test."""
    if settings.mongodb_uri is None:
        pytest.skip("MONGODB_URI is not configured")
    history = MongoHistoryStore(settings)
    try:
        await history.ensure_indexes()
    except Exception as exc:  # noqa: BLE001 - any failure here means unreachable
        await history.close()
        pytest.skip(f"MongoDB is not reachable: {type(exc).__name__}")
    try:
        yield history
    finally:
        await history.close()


@pytest.fixture
def profile() -> str:
    """A unique profile name so tests cannot see each other's rows."""
    return f"test-{uuid.uuid4().hex[:12]}"


def _entry(profile: str, *, medications: int = 2, unverified: int = 1) -> HistoryEntry:
    """Build a history entry for ``profile``."""
    return HistoryEntry(
        profile=profile,
        document_id="a" * 64,
        saved_at=datetime.now(tz=UTC),
        medications=tuple(
            HistoryMedication(
                display_name=f"drug-{index}",
                rxcui=str(6809 + index),
                dose_text="500 MG",
                frequency_text="twice daily",
            )
            for index in range(medications)
        ),
        unverified_count=unverified,
    )


async def test_the_store_satisfies_the_protocol(store: MongoHistoryStore) -> None:
    """It must be interchangeable with the null store."""
    assert isinstance(store, HistoryStore)
    assert store.enabled is True


async def test_indexes_can_be_created_repeatedly(store: MongoHistoryStore) -> None:
    """Startup runs this every time, so it has to be idempotent."""
    await store.ensure_indexes()
    await store.ensure_indexes()


async def test_a_saved_entry_reads_back_intact(store: MongoHistoryStore, profile: str) -> None:
    """The strict model survives the round trip through the driver."""
    original = _entry(profile)
    await store.save(original)
    try:
        (loaded,) = await store.list_for(profile)
        assert loaded.profile == profile
        assert loaded.document_id == original.document_id
        assert loaded.medication_count == 2
        assert loaded.unverified_count == 1
        assert [m.display_name for m in loaded.medications] == ["drug-0", "drug-1"]
        assert loaded.medications[0].dose_text == "500 MG"
    finally:
        await store.delete_all(profile)


async def test_lookup_is_case_insensitive(store: MongoHistoryStore, profile: str) -> None:
    """ "Grandma" and "grandma" are one person, not two profiles."""
    await store.save(_entry(profile))
    try:
        assert len(await store.list_for(profile.upper())) == 1
        assert len(await store.list_for(f"  {profile}  ")) == 1
    finally:
        await store.delete_all(profile)


async def test_entries_come_back_newest_first(store: MongoHistoryStore, profile: str) -> None:
    """The timeline view depends on this ordering."""
    older = _entry(profile, medications=1)
    newer = HistoryEntry(
        profile=profile,
        document_id="b" * 64,
        saved_at=datetime.now(tz=UTC),
        medications=(HistoryMedication(display_name="newer", rxcui="1"),),
        unverified_count=0,
    )
    await store.save(older)
    await store.save(newer)
    try:
        entries = await store.list_for(profile)
        assert len(entries) == 2
        assert entries[0].saved_at >= entries[1].saved_at
    finally:
        await store.delete_all(profile)


async def test_the_limit_is_respected(store: MongoHistoryStore, profile: str) -> None:
    """A long history must not be returned in full by accident."""
    for _ in range(3):
        await store.save(_entry(profile, medications=1))
    try:
        assert len(await store.list_for(profile, limit=2)) == 2
    finally:
        await store.delete_all(profile)


async def test_one_profile_cannot_see_another(store: MongoHistoryStore) -> None:
    """Profiles are isolated by key, which is what the delete relies on too."""
    first = f"test-{uuid.uuid4().hex[:12]}"
    second = f"test-{uuid.uuid4().hex[:12]}"
    await store.save(_entry(first))
    await store.save(_entry(second))
    try:
        assert len(await store.list_for(first)) == 1
        assert len(await store.list_for(second)) == 1
    finally:
        await store.delete_all(first)
        await store.delete_all(second)


async def test_delete_all_removes_every_entry_for_a_profile(
    store: MongoHistoryStore, profile: str
) -> None:
    """This is the "delete all my data" path, so it must be complete."""
    for _ in range(3):
        await store.save(_entry(profile, medications=1))
    assert await store.delete_all(profile) == 3
    assert await store.list_for(profile) == ()


async def test_delete_all_is_safe_on_an_unknown_profile(
    store: MongoHistoryStore, profile: str
) -> None:
    """Deleting nothing reports zero rather than failing."""
    assert await store.delete_all(profile) == 0


async def test_an_unknown_profile_has_no_history(store: MongoHistoryStore, profile: str) -> None:
    """A fresh profile reads as empty, not as an error."""
    assert await store.list_for(profile) == ()
