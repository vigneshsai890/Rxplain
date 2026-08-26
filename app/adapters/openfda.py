"""openFDA drug-label access, cached in Postgres.

How a label is tied to a resolved drug
--------------------------------------
openFDA label records carry ``openfda.rxcui``, but those are *product* RxCUIs
(SCD/SBD level, e.g. 861007 for a specific metformin tablet). A prescription
resolves to an *ingredient* RxCUI (6809 for metformin), which never appears in
that list. Searching openFDA by ingredient RxCUI therefore returns nothing.

Matching on name alone would be worse: "generic_name" is free text from a
manufacturer, and accepting the first label whose name looks similar would let a
label for a different drug supply the text a patient reads.

So the join has three steps and all must succeed:

1. Search ``openfda.generic_name`` for the string **RxNorm** matched - not the
   raw OCR text - so the query itself is already verified data.
2. Require the returned label's ``openfda.rxcui`` list to intersect the product
   RxCUIs recorded for that ingredient in ``rxnorm_ingredient_product``. The
   intersection is stored alongside the cached label, so the link can be
   re-audited later.
3. Require the label to describe a **single** active substance, and that
   substance to be the resolved ingredient.

Step 3 is not redundant. RxNorm correctly relates a combination product to each
of its ingredients, so step 2 alone accepts a combination label for a
single-ingredient prescription: searching metformin returns ZITUVIMET
(sitagliptin and metformin) and searching lisinopril returns lisinopril with
hydrochlorothiazide, both with a genuine RxCUI overlap. Their label text opens by
describing a drug the patient is not taking. Step 3 rejects them.

If any step fails, the label is discarded. A medication then renders with its
verified dose and frequency but no label guidance, which is the safe
degradation: less information, never wrong information.

Label text is stored and later displayed **verbatim**. Nothing here rewrites,
summarises or simplifies it.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Final, Protocol, runtime_checkable

import httpx2
import structlog
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from pydantic import BaseModel, ConfigDict, Field

from app.adapters.rxnorm import normalize
from app.config import Settings
from app.db import Database

_LOGGER = structlog.stdlib.get_logger("app.adapters.openfda")

_STRICT = ConfigDict(strict=True, frozen=True, extra="forbid")

# Label sections quoted to patients, in the order they are shown. Chosen because
# they are patient-directed rather than prescriber-directed, and because openFDA
# populates them consistently.
PATIENT_SECTIONS: Final = (
    "boxed_warning",
    "indications_and_usage",
    "information_for_patients",
    "warnings_and_cautions",
)

_SECTION_TITLES: Final = {
    "boxed_warning": "Boxed warning",
    "indications_and_usage": "What this medicine is used for",
    "information_for_patients": "Information for patients",
    "warnings_and_cautions": "Warnings and cautions",
}


class LabelSection(BaseModel):
    """One verbatim section of an openFDA drug label."""

    model_config = _STRICT

    name: str = Field(min_length=1)
    title: str = Field(min_length=1)
    text: str = Field(min_length=1)

    @property
    def ref(self) -> str:
        """The provenance ref for text quoted from this section."""
        return self.name


class OpenFdaLabel(BaseModel):
    """A label accepted for one ingredient RxCUI."""

    model_config = _STRICT

    ingredient_rxcui: str = Field(min_length=1)
    spl_id: str = Field(min_length=1)
    spl_set_id: str = Field(min_length=1)
    generic_name: str = Field(min_length=1)
    accepted_product_rxcuis: tuple[str, ...] = Field(min_length=1)
    sections: tuple[LabelSection, ...]

    @property
    def source_url(self) -> str:
        """Human-checkable link to the label this text came from."""
        return f"https://labels.fda.gov/setid={self.spl_set_id}"

    def section(self, name: str) -> LabelSection | None:
        """Return the named section, if the label carries it."""
        for section in self.sections:
            if section.name == name:
                return section
        return None


@runtime_checkable
class OpenFdaClient(Protocol):
    """Supplies a verbatim drug label for a resolved ingredient."""

    async def fetch_label(self, ingredient_rxcui: str, generic_name: str) -> OpenFdaLabel | None:
        """Return a label whose RxCUIs confirm it belongs to this ingredient."""
        ...


class NullOpenFdaClient:
    """Returns no label at all, for runs with openFDA disabled."""

    async def fetch_label(self, ingredient_rxcui: str, generic_name: str) -> OpenFdaLabel | None:
        """Return ``None`` without performing any lookup."""
        del ingredient_rxcui, generic_name
        return None


def _first_string(value: object) -> str | None:
    """Return the first string in a JSON list, or the value if already a string."""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        for item in value:
            if isinstance(item, str):
                return item
    return None


def _string_list(value: object) -> tuple[str, ...]:
    """Return every string in a JSON list."""
    if not isinstance(value, list):
        return ()
    return tuple(item for item in value if isinstance(item, str))


def describes_only(substance_names: tuple[str, ...], ingredient_name: str) -> bool:
    """True when a label's substances are exactly the resolved ingredient.

    Requires one substance, whose words and the ingredient's words share a
    prefix in one direction or the other. That accepts a salt form the label
    spells out ("metformin hydrochloride" for "metformin") while rejecting any
    label listing a second active substance.

    Args:
        substance_names: ``openfda.substance_name`` from the label.
        ingredient_name: The string RxNorm matched.

    Returns:
        Whether the label may be quoted for this ingredient.
    """
    if len(substance_names) != 1:
        return False
    label_words = normalize(substance_names[0]).split()
    ingredient_words = normalize(ingredient_name).split()
    if not label_words or not ingredient_words:
        return False
    shorter, longer = sorted((label_words, ingredient_words), key=len)
    return longer[: len(shorter)] == shorter


def _sections_from_result(result: dict[str, object]) -> tuple[LabelSection, ...]:
    """Extract the patient-facing sections present on a label result."""
    sections: list[LabelSection] = []
    for name in PATIENT_SECTIONS:
        text = _first_string(result.get(name))
        if text is None or not text.strip():
            continue
        sections.append(LabelSection(name=name, title=_SECTION_TITLES[name], text=text.strip()))
    return tuple(sections)


class PostgresCachedOpenFdaClient:
    """openFDA client with a Postgres read-through cache."""

    def __init__(self, database: Database, settings: Settings) -> None:
        """Bind the client to a database and openFDA configuration."""
        self._database = database
        self._base_url = settings.openfda_base_url.rstrip("/")
        self._timeout = settings.openfda_timeout_seconds
        self._ttl = timedelta(hours=settings.openfda_cache_ttl_hours)

    async def fetch_label(self, ingredient_rxcui: str, generic_name: str) -> OpenFdaLabel | None:
        """Return a verified label for the ingredient, from cache or openFDA.

        Args:
            ingredient_rxcui: The RxCUI the drug text resolved to.
            generic_name: The string RxNorm matched, used as the search term.

        Returns:
            The label, or ``None`` when openFDA has nothing whose RxCUIs confirm
            it belongs to this ingredient.
        """
        cached = await self._read_cache(ingredient_rxcui)
        if cached is not None:
            return cached

        product_rxcuis = await self._product_rxcuis(ingredient_rxcui)
        if not product_rxcuis:
            _LOGGER.info("openfda_no_product_rxcuis", ingredient_rxcui=ingredient_rxcui)
            return None

        label = await self._search(ingredient_rxcui, generic_name, product_rxcuis)
        if label is None:
            return None
        await self._write_cache(label)
        return label

    async def _product_rxcuis(self, ingredient_rxcui: str) -> frozenset[str]:
        """Return the product RxCUIs recorded for an ingredient."""
        async with self._database.connection() as conn:
            cur = await conn.execute(
                "SELECT product_rxcui FROM rxnorm_ingredient_product WHERE ingredient_rxcui = %s",
                (ingredient_rxcui,),
            )
            rows = await cur.fetchall()
        return frozenset(str(row[0]) for row in rows)

    async def _read_cache(self, ingredient_rxcui: str) -> OpenFdaLabel | None:
        """Return a cached label if one is present and still fresh."""
        async with self._database.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                """
                SELECT payload::text AS payload_json, fetched_at
                FROM openfda_label_cache
                WHERE ingredient_rxcui = %s
                """,
                (ingredient_rxcui,),
            )
            row = await cur.fetchone()
        if row is None:
            return None
        fetched_at = row["fetched_at"]
        if not isinstance(fetched_at, datetime) or datetime.now(tz=UTC) - fetched_at > self._ttl:
            return None
        # Read back as JSON text: psycopg decodes jsonb arrays to Python lists,
        # which strict validation rejects for a tuple field.
        return OpenFdaLabel.model_validate_json(str(row["payload_json"]))

    async def _write_cache(self, label: OpenFdaLabel) -> None:
        """Store a label, replacing any previous entry for the ingredient."""
        async with self._database.connection() as conn, conn.cursor() as cur:
            await cur.execute(
                """
                INSERT INTO openfda_label_cache
                    (ingredient_rxcui, spl_id, spl_set_id, accepted_product_rxcuis,
                     payload, fetched_at)
                VALUES (%s, %s, %s, %s, %s, now())
                ON CONFLICT (ingredient_rxcui) DO UPDATE SET
                    spl_id = EXCLUDED.spl_id,
                    spl_set_id = EXCLUDED.spl_set_id,
                    accepted_product_rxcuis = EXCLUDED.accepted_product_rxcuis,
                    payload = EXCLUDED.payload,
                    fetched_at = now()
                """,
                (
                    label.ingredient_rxcui,
                    label.spl_id,
                    label.spl_set_id,
                    list(label.accepted_product_rxcuis),
                    Jsonb(json.loads(label.model_dump_json())),
                ),
            )
            await conn.commit()

    async def _search(
        self, ingredient_rxcui: str, generic_name: str, product_rxcuis: frozenset[str]
    ) -> OpenFdaLabel | None:
        """Query openFDA and accept the first label whose RxCUIs intersect."""
        params = {
            "search": f'openfda.generic_name:"{generic_name}"',
            "limit": "10",
        }
        try:
            async with httpx2.AsyncClient(timeout=self._timeout) as client:
                response = await client.get(f"{self._base_url}/drug/label.json", params=params)
        except httpx2.HTTPError as exc:
            _LOGGER.warning("openfda_request_failed", error=type(exc).__name__)
            return None

        if response.status_code == httpx2.codes.NOT_FOUND:
            _LOGGER.info("openfda_no_results", ingredient_rxcui=ingredient_rxcui)
            return None
        if response.status_code != httpx2.codes.OK:
            _LOGGER.warning("openfda_unexpected_status", status_code=response.status_code)
            return None

        return self._select(response.json(), ingredient_rxcui, generic_name, product_rxcuis)

    @staticmethod
    def _select(
        body: object,
        ingredient_rxcui: str,
        ingredient_name: str,
        product_rxcuis: frozenset[str],
    ) -> OpenFdaLabel | None:
        """Pick the first result whose product RxCUIs confirm the ingredient."""
        if not isinstance(body, dict):
            return None
        results = body.get("results")
        if not isinstance(results, list):
            return None

        for result in results:
            if not isinstance(result, dict):
                continue
            openfda = result.get("openfda")
            if not isinstance(openfda, dict):
                continue
            overlap = product_rxcuis & set(_string_list(openfda.get("rxcui")))
            if not overlap:
                continue
            substances = _string_list(openfda.get("substance_name"))
            if not describes_only(substances, ingredient_name):
                _LOGGER.info(
                    "openfda_label_rejected_multi_substance",
                    ingredient_rxcui=ingredient_rxcui,
                    substance_count=len(substances),
                )
                continue
            spl_id = _first_string(openfda.get("spl_id")) or _first_string(result.get("id"))
            spl_set_id = _first_string(openfda.get("spl_set_id")) or _first_string(
                result.get("set_id")
            )
            generic = _first_string(openfda.get("generic_name"))
            sections = _sections_from_result(result)
            if not (spl_id and spl_set_id and generic and sections):
                continue
            _LOGGER.info(
                "openfda_label_accepted",
                ingredient_rxcui=ingredient_rxcui,
                matched_product_rxcuis=len(overlap),
                sections=len(sections),
            )
            return OpenFdaLabel(
                ingredient_rxcui=ingredient_rxcui,
                spl_id=spl_id,
                spl_set_id=spl_set_id,
                generic_name=generic,
                accepted_product_rxcuis=tuple(sorted(overlap)),
                sections=sections,
            )

        _LOGGER.info("openfda_no_rxcui_overlap", ingredient_rxcui=ingredient_rxcui)
        return None
