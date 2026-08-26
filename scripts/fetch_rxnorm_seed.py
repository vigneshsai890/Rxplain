"""Fetch a real RxNorm seed from the public NLM RxNav API.

Run from the repository root when the seed needs refreshing::

    .venv/bin/python scripts/fetch_rxnorm_seed.py

Writes ``migrations/seed/rxnorm_seed.json``, which is committed so that neither
CI nor a local run needs network access.

Why this exists alongside ``load_rxnconso.py``: the authoritative RXNCONSO.RRF
release is behind a UMLS licence, so it cannot be committed or downloaded
unattended. RxNav serves the same concepts over an open API, which keeps every
RxCUI in this repository real NLM data. Nothing here is invented - the script
records only what the API returned, plus the URL and timestamp it came from.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

REPO_ROOT = Path(__file__).resolve().parents[1]
SEED_PATH = REPO_ROOT / "migrations" / "seed" / "rxnorm_seed.json"

RXNAV_BASE: Final = "https://rxnav.nlm.nih.gov/REST"
USER_AGENT: Final = "RxPlain-seed-builder/0.1 (+https://github.com/vigneshsai890/Rxplain)"
REQUEST_PAUSE_SECONDS: Final = 0.1

# Ingredients covered by the committed prescription fixtures, plus common
# maintenance medicines so the approximate-match tier has neighbours to be
# confused by. Ingredient names only: RxNav resolves them to real RxCUIs.
INGREDIENTS: Final = (
    "albuterol",
    "amlodipine",
    "amoxicillin",
    "atorvastatin",
    "escitalopram",
    "gabapentin",
    "hydrochlorothiazide",
    "ibuprofen",
    "levothyroxine",
    "lisinopril",
    "losartan",
    "metformin",
    "metoprolol",
    "montelukast",
    "omeprazole",
    "pantoprazole",
    "prednisone",
    "sertraline",
    "simvastatin",
    "warfarin",
)

# Term types worth storing. IN/PIN/MIN are what a prescription's drug name maps
# onto; BN covers brand names a patient may see on the bottle.
CONCEPT_TTYS: Final = ("IN", "PIN", "MIN", "BN")

# Product-level term types, used only to build the ingredient -> product RxCUI
# map that validates an openFDA label really belongs to this ingredient.
PRODUCT_TTYS: Final = ("SCD", "SBD")


class RxNavError(RuntimeError):
    """RxNav returned something the script cannot interpret."""


def _get(path: str, params: dict[str, str]) -> dict[str, object]:
    """GET a JSON document from RxNav.

    Args:
        path: Path below ``RXNAV_BASE``, e.g. ``"/rxcui.json"``.
        params: Query parameters.

    Returns:
        The decoded JSON object.

    Raises:
        RxNavError: On a transport failure or a non-object response.
    """
    url = f"{RXNAV_BASE}{path}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(  # noqa: S310 - fixed https base, not user input
        url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310
            decoded = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RxNavError(f"RxNav request failed: {url}") from exc
    if not isinstance(decoded, dict):
        raise RxNavError(f"RxNav returned a non-object body: {url}")
    time.sleep(REQUEST_PAUSE_SECONDS)
    return decoded


def _as_str_list(value: object) -> list[str]:
    """Coerce an untyped JSON value into a list of strings."""
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def _properties(rxcui: str) -> dict[str, str]:
    """Return the ``{rxcui, name, tty}`` properties of a concept.

    Raises:
        RxNavError: If RxNav returns no properties for ``rxcui``.
    """
    body = _get(f"/rxcui/{rxcui}/properties.json", {})
    properties = body.get("properties")
    if not isinstance(properties, dict):
        raise RxNavError(f"no properties for rxcui {rxcui}")
    return {
        key: value
        for key, value in properties.items()
        if key in {"rxcui", "name", "tty"} and isinstance(value, str)
    }


def _lookup_ingredient(name: str) -> dict[str, str]:
    """Return the RxNorm ingredient (``tty=IN``) concept for a name.

    An exact lookup is used deliberately. Passing ``search=1`` enables RxNav's
    approximate matcher, which for "amlodipine" answers with the salt form
    "amlodipine besylate" (``tty=PIN``); traversing to products from a PIN
    yields nothing, so the openFDA join would silently have no RxCUIs to check.

    Raises:
        RxNavError: If no ingredient concept can be reached from ``name``.
    """
    body = _get("/rxcui.json", {"name": name})
    group = body.get("idGroup")
    ids = _as_str_list(group.get("rxnormId")) if isinstance(group, dict) else []
    if not ids:
        raise RxNavError(f"RxNav could not resolve {name!r} exactly")

    properties = _properties(ids[0])
    if properties.get("tty") == "IN":
        return properties

    for row in _related_concepts(ids[0], ("IN",)):
        return {"rxcui": row["rxcui"], "name": row["str"], "tty": "IN"}
    raise RxNavError(f"no IN concept reachable from {name!r}")


def _related_concepts(rxcui: str, ttys: tuple[str, ...]) -> list[dict[str, str]]:
    """Return ``{rxcui, tty, str}`` rows related to ``rxcui`` by term type."""
    body = _get(f"/rxcui/{rxcui}/related.json", {"tty": " ".join(ttys)})
    group = body.get("relatedGroup")
    if not isinstance(group, dict):
        return []
    concept_groups = group.get("conceptGroup")
    if not isinstance(concept_groups, list):
        return []

    rows: list[dict[str, str]] = []
    for concept_group in concept_groups:
        if not isinstance(concept_group, dict):
            continue
        tty = concept_group.get("tty")
        properties = concept_group.get("conceptProperties")
        if not isinstance(tty, str) or not isinstance(properties, list):
            continue
        for prop in properties:
            if not isinstance(prop, dict):
                continue
            concept_rxcui = prop.get("rxcui")
            name = prop.get("name")
            if isinstance(concept_rxcui, str) and isinstance(name, str):
                rows.append({"rxcui": concept_rxcui, "tty": tty, "str": name})
    return rows


def build_seed() -> dict[str, object]:
    """Fetch every ingredient and assemble the seed document."""
    concepts: dict[tuple[str, str, str], dict[str, str]] = {}
    products: set[tuple[str, str]] = set()
    anchors: list[dict[str, str]] = []

    for name in INGREDIENTS:
        ingredient = _lookup_ingredient(name)
        ingredient_rxcui = ingredient["rxcui"]
        anchors.append(ingredient)

        # Record the ingredient itself, then everything related to it.
        concepts[(ingredient_rxcui, "IN", ingredient["name"])] = {
            "rxcui": ingredient_rxcui,
            "tty": "IN",
            "str": ingredient["name"],
        }
        for row in _related_concepts(ingredient_rxcui, CONCEPT_TTYS):
            concepts[(row["rxcui"], row["tty"], row["str"])] = row

        product_rows = _related_concepts(ingredient_rxcui, PRODUCT_TTYS)
        if not product_rows:
            raise RxNavError(f"no products reachable from ingredient {name!r}")

        # Salt forms (PIN) are the same active ingredient, and a prescription
        # often names one ("AMLODIPINE BESYLATE"). RxNav cannot traverse from a
        # PIN to products, so each PIN inherits its ingredient's product set;
        # without this the openFDA join has no RxCUIs to check against.
        salt_rxcuis = [row["rxcui"] for row in _related_concepts(ingredient_rxcui, ("PIN",))]
        for row in product_rows:
            products.add((ingredient_rxcui, row["rxcui"]))
            for salt_rxcui in salt_rxcuis:
                products.add((salt_rxcui, row["rxcui"]))

        print(
            f"{name:22s} IN={ingredient_rxcui:>8s} "
            f"products={len(product_rows):4d} salts={len(salt_rxcuis)} concepts={len(concepts):5d}"
        )

    return {
        "source": "RXNAV_API",
        "source_url": RXNAV_BASE,
        "fetched_at": datetime.now(tz=UTC).isoformat(),
        "ingredients": anchors,
        "concepts": [concepts[key] for key in sorted(concepts)],
        "ingredient_products": [
            {"ingredient_rxcui": ingredient, "product_rxcui": product}
            for ingredient, product in sorted(products)
        ],
    }


if __name__ == "__main__":
    seed = build_seed()
    SEED_PATH.parent.mkdir(parents=True, exist_ok=True)
    SEED_PATH.write_text(json.dumps(seed, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\nwrote {SEED_PATH.relative_to(REPO_ROOT)}")
