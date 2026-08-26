"""Tests for the in-process RxNorm resolver.

The similarity figures below were taken from a live PostgreSQL server running
``pg_trgm``. They are asserted literally, so a drift in the Python port shows up
as a failing test rather than as a quietly different resolution.

No database is needed here, which is the point: these run in an environment where
the memory backend is the only option.
"""

from __future__ import annotations

import pytest

from app.adapters.rxnorm_local import (
    InProcessRxNormResolver,
    similarity,
    trigrams,
)
from app.config import Settings
from app.domain.medication import ResolvedConcept, UnresolvedConcept

pytestmark = pytest.mark.anyio

# Verified against `SELECT similarity(a, b)` on PostgreSQL 17 with pg_trgm.
PG_SIMILARITY = [
    ("metformin hcl", "metformin", 0.714286),
    ("metformin hcl", "metformin hydrochloride", 0.407407),
    ("lisinopril hydrochlorothiazide", "hydrochlorothiazide lisinopril", 1.0),
    ("metformen", "metformin", 0.538462),
    ("atorvastatin calcium", "atorvastatin calcium", 1.0),
    ("zyntrafex", "metformin", 0.0),
    ("amlodipine besylate", "amlodipine", 0.550000),
]

# Verified identical to the Postgres-backed resolver across these queries.
EXPECTED_RESOLUTION = [
    ("metformin", "6809", "exact"),
    ("Glucophage", "151827", "exact"),
    ("METFORMIN HCL 500 MG TABLET", "6809", "truncated"),
    ("LISINOPRIL 10 MG TABLET", "29046", "normalized"),
    ("ATORVASTATIN CALCIUM 20 MG TABLET", "83366", "normalized"),
    ("AMOXICILLIN 500 MG CAPSULE", "723", "normalized"),
    ("IBUPROFEN 400 MG TABLET", "5640", "normalized"),
    ("OMEPRAZOLE 20 MG CAPSULE", "7646", "normalized"),
    ("LEVOTHYROXINE SODIUM 50 MCG TABLET", "40144", "normalized"),
    ("AMLODIPINE BESYLATE 5 MG TABLET", "104416", "normalized"),
    ("LISINOPRIL HYDROCHLOROTHIAZIDE 20-25 MG TABLET", "214618", "approximate"),
]


@pytest.fixture(scope="module")
def resolver() -> InProcessRxNormResolver:
    """Build the in-process index once for the module."""
    return InProcessRxNormResolver(Settings(app_env="test"))


# --------------------------------------------------------------------------- #
# the pg_trgm port
# --------------------------------------------------------------------------- #
def test_trigrams_match_show_trgm() -> None:
    """The trigram set matches ``show_trgm('metformin hcl')`` exactly."""
    assert sorted(trigrams("metformin hcl")) == [
        "  h",
        "  m",
        " hc",
        " me",
        "cl ",
        "etf",
        "for",
        "hcl",
        "in ",
        "met",
        "min",
        "orm",
        "rmi",
        "tfo",
    ]


def test_trigrams_of_empty_text_is_empty() -> None:
    """Text with no alphanumerics yields no trigrams."""
    assert trigrams("") == frozenset()
    assert trigrams("   ///  ") == frozenset()


@pytest.mark.parametrize(("left", "right", "expected"), PG_SIMILARITY)
def test_similarity_matches_postgres(left: str, right: str, expected: float) -> None:
    """Each figure came from a live pg_trgm query and must be reproduced."""
    assert similarity(left, right) == pytest.approx(expected, abs=1e-6)


def test_similarity_is_symmetric() -> None:
    """Jaccard is symmetric, and so is pg_trgm."""
    for left, right, _ in PG_SIMILARITY:
        assert similarity(left, right) == pytest.approx(similarity(right, left))


def test_similarity_of_identical_text_is_one() -> None:
    """A string is a perfect match for itself."""
    assert similarity("metformin", "metformin") == 1.0


def test_similarity_ignores_case_and_punctuation() -> None:
    """Normalisation happens before trigrams, as in Postgres."""
    assert similarity("Metformin HCl", "metformin hcl") == 1.0


# --------------------------------------------------------------------------- #
# the index
# --------------------------------------------------------------------------- #
def test_the_committed_seed_is_indexed(resolver: InProcessRxNormResolver) -> None:
    """All 310 committed concepts load."""
    assert resolver.concept_count == 310


def test_product_relations_are_indexed(resolver: InProcessRxNormResolver) -> None:
    """The openFDA join needs these, including for salt forms."""
    assert len(resolver.product_rxcuis("6809")) > 0
    assert len(resolver.product_rxcuis("104416")) > 0
    assert resolver.product_rxcuis("does-not-exist") == frozenset()


# --------------------------------------------------------------------------- #
# the ladder
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(("drug_text", "rxcui", "tier"), EXPECTED_RESOLUTION)
async def test_resolution_matches_the_database_backend(
    resolver: InProcessRxNormResolver, drug_text: str, rxcui: str, tier: str
) -> None:
    """Each query lands on the same RxCUI and tier as the Postgres resolver."""
    resolution = await resolver.resolve(drug_text)
    assert isinstance(resolution, ResolvedConcept)
    assert resolution.rxcui == rxcui
    assert resolution.match_tier == tier


async def test_a_combination_product_is_not_truncated(
    resolver: InProcessRxNormResolver,
) -> None:
    """The safeguard that keeps tier 4 safe holds in this backend too."""
    resolution = await resolver.resolve("LISINOPRIL HYDROCHLOROTHIAZIDE 20-25 MG TABLET")
    assert isinstance(resolution, ResolvedConcept)
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
        ("METFORMEN 500 MG", "below_similarity_threshold"),
    ],
)
async def test_unresolvable_text_reports_the_same_reason(
    resolver: InProcessRxNormResolver, drug_text: str, reason: str
) -> None:
    """The reason is shown to the patient, so it must match the other backend.

    ``no_candidate`` rather than ``below_similarity_threshold`` for a nonsense drug
    depends on reproducing pg_trgm's pre-filter, not just its score.
    """
    resolution = await resolver.resolve(drug_text)
    assert isinstance(resolution, UnresolvedConcept)
    assert resolution.reason == reason


async def test_a_near_miss_is_refused_not_corrected(
    resolver: InProcessRxNormResolver,
) -> None:
    """A misspelling close to a real drug still fails."""
    resolution = await resolver.resolve("METFORMEN 500 MG")
    assert isinstance(resolution, UnresolvedConcept)
    assert 0.0 < resolution.best_similarity < 0.85


async def test_lowering_the_threshold_is_what_would_admit_a_near_miss() -> None:
    """Confirms the refusal is the threshold at work, not missing data."""
    lenient = InProcessRxNormResolver(Settings(app_env="test", rxnorm_min_similarity=0.5))
    resolution = await lenient.resolve("METFORMEN 500 MG")
    assert isinstance(resolution, ResolvedConcept)
    assert resolution.match_tier == "approximate"
