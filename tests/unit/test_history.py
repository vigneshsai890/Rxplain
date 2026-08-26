"""Tests for :mod:`app.adapters.history`.

No database. The pure parts - the models, the guide-to-entry mapping, the log
fingerprint and the null store - are covered here; the MongoDB paths are covered
in ``tests/integration/test_history.py``, which skips when no cluster is
configured.

The privacy assertions matter most: history is the only feature that persists
anything identifying, so what it saves and what it logs are pinned.
"""

from __future__ import annotations

from datetime import UTC, datetime

import anyio
import pytest
from pydantic import ValidationError

from app.adapters.history import (
    COLLECTION,
    MAX_PROFILE_LENGTH,
    HistoryEntry,
    HistoryMedication,
    HistoryStore,
    MongoHistoryStore,
    NullHistoryStore,
    entry_from_guide,
    profile_fingerprint,
)
from app.config import Settings
from app.domain.guide import (
    GuideSentence,
    LabelQuote,
    MedicationGuidance,
    Provenance,
    RenderedGuide,
)
from app.domain.ingest import BoundingBox, SourceSpan
from app.safety.verifier import UnverifiedItem

BOX = BoundingBox(left=0.1, top=0.1, right=0.4, bottom=0.14)


def _span(span_id: str, text: str) -> SourceSpan:
    """Build a span carrying ``text``."""
    return SourceSpan(span_id=span_id, page_number=1, token_ids=("p1-t001",), text=text, box=BOX)


def _guidance(name: str = "metformin", rxcui: str = "6809") -> MedicationGuidance:
    """Build guidance with one rxnorm sentence and two ocr_span sentences."""
    return MedicationGuidance(
        display_name=name,
        rxcui=rxcui,
        match_tier="exact",
        prescription_text=f"{name.upper()} 500 MG twice daily",
        sentences=(
            GuideSentence(
                text=f"This medicine is {name}.",
                provenance=Provenance(source="rxnorm", ref=rxcui),
                source_text=name,
            ),
            GuideSentence(
                text="The strength on your prescription is 500 MG.",
                provenance=Provenance(source="ocr_span", ref="s-dose"),
                source_text="500 MG",
            ),
            GuideSentence(
                text="Your prescription says to take it twice daily.",
                provenance=Provenance(source="ocr_span", ref="s-freq"),
                source_text="twice daily",
            ),
        ),
        quotes=(
            LabelQuote(
                title="What this medicine is used for",
                text="Used with diet to lower blood sugar.",
                provenance=Provenance(source="openfda", ref="indications_and_usage"),
                source_url="https://labels.fda.gov/setid=set-1",
            ),
        ),
    )


def _rendered(*guidance: MedicationGuidance, unverified: int = 0) -> RenderedGuide:
    """Build a rendered guide with the given medications and refusal count."""
    items = tuple(
        UnverifiedItem(
            raw_text=f"UNKNOWN{index} 250 MG",
            line_span=_span(f"s-line-{index}", f"UNKNOWN{index} 250 MG"),
            reason="rxnorm_unresolved",
            detail="no RxNorm match (no_candidate)",
        )
        for index in range(unverified)
    )
    return RenderedGuide(
        document_id="a" * 64,
        medications=guidance or (_guidance(),),
        unverified=items,
        attempts=1,
    )


# --------------------------------------------------------------------------- #
# privacy: what gets saved, and what gets logged
# --------------------------------------------------------------------------- #
def test_only_already_displayed_fields_are_saved() -> None:
    """Saving adds no information the user has not already seen on screen."""
    entry = entry_from_guide("Grandma", _rendered())
    assert entry.profile == "Grandma"
    assert entry.document_id == "a" * 64
    (medication,) = entry.medications
    assert medication.display_name == "metformin"
    assert medication.rxcui == "6809"
    assert medication.dose_text == "500 MG"
    assert medication.frequency_text == "twice daily"
    # No image, no transcription, no label text, no provenance internals.
    assert set(HistoryMedication.model_fields) == {
        "display_name",
        "rxcui",
        "dose_text",
        "frequency_text",
    }


def test_the_saved_shape_carries_no_label_or_span_detail() -> None:
    """The entry model is deliberately narrow."""
    assert set(HistoryEntry.model_fields) == {
        "profile",
        "document_id",
        "saved_at",
        "medications",
        "unverified_count",
    }


def test_unverified_items_are_counted_not_stored() -> None:
    """A refusal is worth remembering; its text is not worth persisting."""
    entry = entry_from_guide("Dad", _rendered(unverified=2))
    assert entry.unverified_count == 2


def test_the_profile_name_is_never_logged_in_the_clear() -> None:
    """A profile name is whatever the user typed, so it is identifying."""
    fingerprint = profile_fingerprint("Grandma")
    assert "grandma" not in fingerprint.casefold()
    assert len(fingerprint) == 12
    assert fingerprint.isalnum()


def test_the_fingerprint_is_stable_and_case_insensitive() -> None:
    """The same person typed two ways correlates to one fingerprint."""
    assert profile_fingerprint("Grandma") == profile_fingerprint("  grandma  ")
    assert profile_fingerprint("Grandma") != profile_fingerprint("Grandpa")


def test_profile_names_are_trimmed() -> None:
    """Stray whitespace must not create a second profile."""
    assert entry_from_guide("  Mum  ", _rendered()).profile == "Mum"


# --------------------------------------------------------------------------- #
# models
# --------------------------------------------------------------------------- #
def test_a_medication_with_no_dose_or_frequency_is_valid() -> None:
    """A prescription may omit either, so the entry must allow it."""
    medication = HistoryMedication(display_name="metformin", rxcui="6809")
    assert medication.dose_text is None
    assert medication.frequency_text is None


def test_an_empty_profile_is_rejected() -> None:
    """An unnamed profile could not be retrieved or deleted."""
    with pytest.raises(ValidationError):
        HistoryEntry(
            profile="",
            document_id="a" * 64,
            saved_at=datetime.now(tz=UTC),
            medications=(),
            unverified_count=0,
        )


def test_an_over_long_profile_is_rejected() -> None:
    """An unbounded name is a storage and display hazard."""
    with pytest.raises(ValidationError):
        HistoryEntry(
            profile="x" * (MAX_PROFILE_LENGTH + 1),
            document_id="a" * 64,
            saved_at=datetime.now(tz=UTC),
            medications=(),
            unverified_count=0,
        )


def test_a_negative_unverified_count_is_rejected() -> None:
    """A count below zero would mean the mapping is broken."""
    with pytest.raises(ValidationError):
        HistoryEntry(
            profile="Mum",
            document_id="a" * 64,
            saved_at=datetime.now(tz=UTC),
            medications=(),
            unverified_count=-1,
        )


def test_medication_count_reports_the_number_saved() -> None:
    """Used by the timeline view and by the audit log line."""
    entry = entry_from_guide("Mum", _rendered(_guidance(), _guidance("lisinopril", "29046")))
    assert entry.medication_count == 2


def test_an_entry_round_trips_through_json() -> None:
    """Entries are persisted as JSON, so strict models must survive the trip."""
    entry = entry_from_guide("Mum", _rendered())
    assert HistoryEntry.model_validate_json(entry.model_dump_json()) == entry


# --------------------------------------------------------------------------- #
# the null store: the zero-retention default
# --------------------------------------------------------------------------- #
def test_history_is_off_unless_a_uri_is_configured() -> None:
    """The default posture stores nothing about a patient."""
    assert Settings(app_env="test").mongodb_uri is None


def test_the_null_store_satisfies_the_protocol() -> None:
    """It must be usable wherever the real store is."""
    assert isinstance(NullHistoryStore(), HistoryStore)
    assert isinstance(NullHistoryStore().enabled, bool)
    assert NullHistoryStore().enabled is False


def test_the_null_store_persists_nothing() -> None:
    """Saving then reading returns nothing, and deleting reports nothing."""

    async def exercise() -> tuple[tuple[HistoryEntry, ...], int]:
        store = NullHistoryStore()
        await store.save(entry_from_guide("Mum", _rendered()))
        listed = await store.list_for("Mum")
        deleted = await store.delete_all("Mum")
        await store.close()
        return listed, deleted

    listed, deleted = anyio.run(exercise)
    assert listed == ()
    assert deleted == 0


# --------------------------------------------------------------------------- #
# the Mongo store, without connecting
# --------------------------------------------------------------------------- #
def test_the_mongo_store_requires_a_uri() -> None:
    """Without a URI the app must fall back, not build a broken client."""
    with pytest.raises(ValueError, match="mongodb_uri"):
        MongoHistoryStore(Settings(app_env="test"))


def test_the_collection_name_is_fixed() -> None:
    """Renaming it would orphan existing history."""
    assert COLLECTION == "prescriptions"
