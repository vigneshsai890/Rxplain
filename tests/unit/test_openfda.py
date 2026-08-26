"""Tests for the openFDA label join.

The selection logic is where a wrong label would come from, so it is tested
against response bodies shaped exactly like openFDA's - including the real case
that caused a combination product's label to be offered for a single-ingredient
prescription.
"""

from __future__ import annotations

import pytest

from app.adapters.openfda import (
    PATIENT_SECTIONS,
    NullOpenFdaClient,
    OpenFdaLabel,
    PostgresCachedOpenFdaClient,
    describes_only,
)

METFORMIN_PRODUCTS = frozenset({"861007", "861769", "861819"})


def _result(
    *,
    rxcui: tuple[str, ...] = ("861007",),
    substances: tuple[str, ...] = ("METFORMIN HYDROCHLORIDE",),
    generic: str = "METFORMIN HYDROCHLORIDE",
    indications: str | None = "Metformin tablets lower blood sugar.",
    spl_id: str = "spl-1",
    spl_set_id: str = "set-1",
) -> dict[str, object]:
    """Build one openFDA label result."""
    result: dict[str, object] = {
        "openfda": {
            "rxcui": list(rxcui),
            "substance_name": list(substances),
            "generic_name": [generic],
            "spl_id": [spl_id],
            "spl_set_id": [spl_set_id],
        }
    }
    if indications is not None:
        result["indications_and_usage"] = [indications]
    return result


def _select(
    *results: dict[str, object],
    ingredient_name: str = "metformin",
    products: frozenset[str] = METFORMIN_PRODUCTS,
) -> OpenFdaLabel | None:
    """Run label selection over the given results."""
    return PostgresCachedOpenFdaClient._select(
        {"results": list(results)}, "6809", ingredient_name, products
    )


# --------------------------------------------------------------------------- #
# describes_only
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("substances", "ingredient"),
    [
        (("METFORMIN HYDROCHLORIDE",), "metformin"),
        (("metformin",), "metformin"),
        (("AMLODIPINE BESYLATE",), "amlodipine besylate"),
        (("ATORVASTATIN CALCIUM",), "atorvastatin calcium"),
        (("LEVOTHYROXINE SODIUM",), "levothyroxine"),
    ],
)
def test_a_single_matching_substance_is_accepted(
    substances: tuple[str, ...], ingredient: str
) -> None:
    """A salt form spelled out by the label still names the same ingredient."""
    assert describes_only(substances, ingredient)


@pytest.mark.parametrize(
    ("substances", "ingredient"),
    [
        # The real failure: RxNorm links this product to metformin, so the RxCUI
        # check passes, but the label opens by describing a different drug.
        (("SITAGLIPTIN", "METFORMIN HYDROCHLORIDE"), "metformin"),
        (("LISINOPRIL", "HYDROCHLOROTHIAZIDE"), "lisinopril"),
        (("IBUPROFEN", "PHENYLEPHRINE HYDROCHLORIDE"), "ibuprofen"),
        # Wrong drug entirely.
        (("WARFARIN SODIUM",), "metformin"),
        # No substances listed at all.
        ((), "metformin"),
    ],
)
def test_anything_but_one_matching_substance_is_refused(
    substances: tuple[str, ...], ingredient: str
) -> None:
    """A second active substance disqualifies the label."""
    assert not describes_only(substances, ingredient)


def test_an_empty_ingredient_name_is_refused() -> None:
    """Nothing to compare against means nothing can be confirmed."""
    assert not describes_only(("METFORMIN",), "")


# --------------------------------------------------------------------------- #
# selection
# --------------------------------------------------------------------------- #
def test_a_matching_single_ingredient_label_is_selected() -> None:
    """The happy path: RxCUI overlap and one matching substance."""
    label = _select(_result())
    assert label is not None
    assert label.ingredient_rxcui == "6809"
    assert label.accepted_product_rxcuis == ("861007",)
    assert label.sections[0].name == "indications_and_usage"


def test_a_combination_product_label_is_rejected() -> None:
    """The RxCUI overlap is genuine; the substance check is what saves us."""
    combination = _result(substances=("SITAGLIPTIN", "METFORMIN HYDROCHLORIDE"))
    assert _select(combination) is None


def test_a_label_with_no_rxcui_overlap_is_rejected() -> None:
    """Without an overlap there is no evidence the label is for this drug."""
    assert _select(_result(rxcui=("999999",))) is None


def test_selection_skips_a_bad_candidate_and_takes_a_good_one() -> None:
    """One unusable result does not abandon the rest of the response."""
    label = _select(
        _result(rxcui=("999999",)),
        _result(substances=("SITAGLIPTIN", "METFORMIN HYDROCHLORIDE")),
        _result(spl_set_id="set-good"),
    )
    assert label is not None
    assert label.spl_set_id == "set-good"


def test_a_label_with_no_usable_sections_is_rejected() -> None:
    """A label with nothing to quote is of no use to a patient."""
    assert _select(_result(indications=None)) is None


def test_a_label_missing_its_identifiers_is_rejected() -> None:
    """Without SPL identifiers the quote could not be attributed."""
    broken = _result()
    openfda = broken["openfda"]
    assert isinstance(openfda, dict)
    del openfda["spl_set_id"]
    assert _select(broken) is None


@pytest.mark.parametrize("body", [None, [], {}, {"results": "nope"}, {"results": [1, 2]}])
def test_a_malformed_response_yields_no_label(body: object) -> None:
    """A response that is not shaped as expected is discarded, not parsed loosely."""
    assert (
        PostgresCachedOpenFdaClient._select(body, "6809", "metformin", METFORMIN_PRODUCTS) is None
    )


def test_patient_sections_are_ordered_most_severe_first() -> None:
    """A boxed warning must never be buried below softer text."""
    assert PATIENT_SECTIONS[0] == "boxed_warning"


def test_label_source_url_points_at_the_spl_set() -> None:
    """Attribution must be checkable by a person, not just a machine."""
    label = _select(_result(spl_set_id="abc123"))
    assert label is not None
    assert label.source_url.endswith("setid=abc123")


@pytest.mark.anyio
async def test_the_null_client_never_returns_a_label() -> None:
    """Disabling openFDA removes label text without failing the request."""
    assert await NullOpenFdaClient().fetch_label("6809", "metformin") is None
