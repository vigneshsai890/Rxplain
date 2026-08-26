"""Tests for RxNorm text normalisation and RXNCONSO.RRF parsing.

Neither needs a database. The RXNCONSO parser is exercised against lines in the
real 18-column release format, documented at
https://www.nlm.nih.gov/research/umls/rxnorm/docs/techdoc.html - so the parser
that would ingest a licensed release is covered even though the release itself
cannot be committed.
"""

from __future__ import annotations

import pytest

from app.adapters.rxnorm import normalize, search_key
from app.adapters.rxnorm_seed import RXNCONSO_COLUMNS, parse_rxnconso


def _rrf_line(
    rxcui: str = "6809",
    lat: str = "ENG",
    tty: str = "IN",
    text: str = "metformin",
    suppress: str = "N",
) -> str:
    """Build one RXNCONSO.RRF line with the given values in the right columns."""
    fields = [""] * len(RXNCONSO_COLUMNS)
    index = {name: position for position, name in enumerate(RXNCONSO_COLUMNS)}
    fields[index["RXCUI"]] = rxcui
    fields[index["LAT"]] = lat
    fields[index["SAB"]] = "RXNORM"
    fields[index["TTY"]] = tty
    fields[index["STR"]] = text
    fields[index["SUPPRESS"]] = suppress
    return "|".join(fields) + "|"


# --------------------------------------------------------------------------- #
# normalisation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("METFORMIN", "metformin"),
        ("Atorvastatin Calcium", "atorvastatin calcium"),
        ("hydrochlorothiazide / lisinopril", "hydrochlorothiazide lisinopril"),
        ("METFORMIN  HCL", "metformin hcl"),
        ("Amoxicillin (as trihydrate)", "amoxicillin as trihydrate"),
    ],
)
def test_normalize_folds_case_and_punctuation(raw: str, expected: str) -> None:
    """Normalisation removes formatting only; no word is added or renamed."""
    assert normalize(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("METFORMIN HCL 500 MG TABLET", "metformin hcl"),
        ("LISINOPRIL 10 MG TABLET", "lisinopril"),
        ("ATORVASTATIN CALCIUM 20 MG TABLET", "atorvastatin calcium"),
        ("LEVOTHYROXINE SODIUM 50 MCG TABLET", "levothyroxine sodium"),
        ("OMEPRAZOLE 20 MG CAPSULE", "omeprazole"),
        ("ALBUTEROL SULFATE HFA 90 MCG INHALER", "albuterol sulfate"),
        ("ZYNTRAFEX 250 MG TABLET", "zyntrafex"),
    ],
)
def test_search_key_drops_strength_and_dosage_form(raw: str, expected: str) -> None:
    """The search key keeps identity words and discards strength and form."""
    assert search_key(raw) == expected


def test_search_key_of_strength_only_text_is_empty() -> None:
    """Text with no identity words yields no key, which the resolver rejects."""
    assert search_key("500 MG TABLET") == ""


def test_search_key_never_invents_words() -> None:
    """Every word in the key is present in the input."""
    raw = "METFORMIN HCL 500 MG TABLET"
    assert set(search_key(raw).split()) <= set(normalize(raw).split())


# --------------------------------------------------------------------------- #
# RXNCONSO parsing
# --------------------------------------------------------------------------- #
def test_parses_a_wanted_english_row() -> None:
    """A normal ingredient row is loaded with its normalised form."""
    (row,) = list(parse_rxnconso(iter([_rrf_line()])))
    assert (row.rxcui, row.tty, row.text) == ("6809", "IN", "metformin")
    assert row.normalized == "metformin"
    assert row.source == "RXNCONSO"


def test_skips_non_english_rows() -> None:
    """Only English strings are shown to patients, so only those are loaded."""
    assert list(parse_rxnconso(iter([_rrf_line(lat="SPA")]))) == []


def test_skips_suppressed_rows() -> None:
    """Suppressed concepts are withdrawn by NLM and must not resolve."""
    assert list(parse_rxnconso(iter([_rrf_line(suppress="O")]))) == []


def test_skips_unwanted_term_types() -> None:
    """Dose-form-only term types are not drug identities."""
    assert list(parse_rxnconso(iter([_rrf_line(tty="DF", text="Oral Tablet")]))) == []


def test_skips_rows_with_a_non_numeric_rxcui() -> None:
    """An RxCUI is always numeric; anything else means a corrupt line."""
    assert list(parse_rxnconso(iter([_rrf_line(rxcui="ABC")]))) == []


def test_skips_rows_with_an_empty_string_field() -> None:
    """A concept with no name cannot be matched or displayed."""
    assert list(parse_rxnconso(iter([_rrf_line(text="   ")]))) == []


def test_skips_truncated_lines_rather_than_guessing() -> None:
    """A short line is dropped: a partly parsed drug name is worse than none."""
    assert list(parse_rxnconso(iter(["6809|ENG|P|"]))) == []


def test_parses_a_mixed_file() -> None:
    """Accepted and rejected rows can be interleaved."""
    lines = [
        _rrf_line(rxcui="6809", tty="IN", text="metformin"),
        _rrf_line(lat="FRE", text="metformine"),
        _rrf_line(rxcui="151827", tty="BN", text="Glucophage"),
        _rrf_line(tty="DF", text="Oral Tablet"),
        _rrf_line(rxcui="29046", tty="IN", text="lisinopril", suppress=""),
    ]
    rows = list(parse_rxnconso(iter(lines)))
    assert [row.rxcui for row in rows] == ["6809", "151827", "29046"]
