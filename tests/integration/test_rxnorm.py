"""Integration tests for the RxNorm fallback ladder.

Every expected RxCUI below is real NLM data, seeded from the public RxNav API.
The values are asserted literally so a change to the ladder that starts
resolving a drug to a different concept fails loudly.
"""

from __future__ import annotations

import pytest

from app.adapters.rxnorm import PostgresRxNormResolver
from app.config import Settings
from app.db import Database
from app.domain.medication import ResolvedConcept, UnresolvedConcept

pytestmark = pytest.mark.anyio


@pytest.mark.parametrize(
    ("drug_text", "rxcui", "tier"),
    [
        # Tier 1: the OCR text is literally an RxNorm string.
        ("metformin", "6809", "exact"),
        ("Glucophage", "151827", "exact"),
        # Tier 2: strength and dosage form discarded, then exact.
        ("LISINOPRIL 10 MG TABLET", "29046", "normalized"),
        ("AMOXICILLIN 500 MG CAPSULE", "723", "normalized"),
        ("IBUPROFEN 400 MG TABLET", "5640", "normalized"),
        ("OMEPRAZOLE 20 MG CAPSULE", "7646", "normalized"),
        ("ATORVASTATIN CALCIUM 20 MG TABLET", "83366", "normalized"),
        ("LEVOTHYROXINE SODIUM 50 MCG TABLET", "40144", "normalized"),
        ("AMLODIPINE BESYLATE 5 MG TABLET", "104416", "normalized"),
        # Tier 4: "hcl" is dropped so the base ingredient resolves.
        ("METFORMIN HCL 500 MG TABLET", "6809", "truncated"),
    ],
)
async def test_resolves_to_the_expected_concept(
    resolver: PostgresRxNormResolver, drug_text: str, rxcui: str, tier: str
) -> None:
    """Each fixture drug string lands on its real RxCUI at the expected tier."""
    resolution = await resolver.resolve(drug_text)
    assert isinstance(resolution, ResolvedConcept)
    assert resolution.rxcui == rxcui
    assert resolution.match_tier == tier


async def test_combination_product_keeps_both_ingredients(
    resolver: PostgresRxNormResolver,
) -> None:
    """A two-drug string matches its combination concept, never one half of it.

    This is the safeguard that makes tier 4 safe: truncation would resolve
    "lisinopril hydrochlorothiazide" to lisinopril alone and silently drop a
    medication, so tier 3 must claim it first.
    """
    resolution = await resolver.resolve("LISINOPRIL HYDROCHLOROTHIAZIDE 20-25 MG TABLET")
    assert isinstance(resolution, ResolvedConcept)
    assert resolution.match_tier == "approximate"
    assert resolution.tty == "MIN"
    assert "lisinopril" in resolution.matched_str.casefold()
    assert "hydrochlorothiazide" in resolution.matched_str.casefold()


@pytest.mark.parametrize(
    ("drug_text", "reason"),
    [
        ("ZYNTRAFEX 250 MG TABLET", "no_candidate"),
        ("QQQQZZZZ", "no_candidate"),
        ("500 MG TABLET", "empty_query"),
        ("", "empty_query"),
    ],
)
async def test_unresolvable_text_is_reported_not_guessed(
    resolver: PostgresRxNormResolver, drug_text: str, reason: str
) -> None:
    """Nothing plausible is returned for a drug RxNorm does not contain."""
    resolution = await resolver.resolve(drug_text)
    assert isinstance(resolution, UnresolvedConcept)
    assert resolution.reason == reason


async def test_near_miss_is_refused_rather_than_corrected(
    resolver: PostgresRxNormResolver,
) -> None:
    """A misspelling close to a real drug still fails: no silent correction."""
    resolution = await resolver.resolve("METFORMEN 500 MG")
    assert isinstance(resolution, UnresolvedConcept)
    assert resolution.reason == "below_similarity_threshold"
    assert 0.0 < resolution.best_similarity < 0.85


async def test_lowering_the_threshold_is_what_would_admit_a_near_miss(
    seeded_database: Database,
) -> None:
    """Confirms the refusal above is the threshold at work, not absent data."""
    lenient = PostgresRxNormResolver(
        seeded_database, Settings(app_env="test", rxnorm_min_similarity=0.5)
    )
    resolution = await lenient.resolve("METFORMEN 500 MG")
    assert isinstance(resolution, ResolvedConcept)
    assert resolution.match_tier == "approximate"


async def test_ingredient_product_relations_are_loaded(seeded_database: Database) -> None:
    """The openFDA join depends on these, so the seed must populate them."""
    async with seeded_database.connection() as conn:
        cur = await conn.execute(
            "SELECT count(*) FROM rxnorm_ingredient_product WHERE ingredient_rxcui = %s",
            ("6809",),
        )
        row = await cur.fetchone()
    assert row is not None
    assert int(row[0]) > 0
