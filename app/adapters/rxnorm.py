"""RxNorm resolution against Postgres.

The fallback ladder
-------------------
A drug string read off a prescription is rarely a literal RxNorm string. It is
resolved by descending these tiers and stopping at the first hit. The tier that
matched is recorded on the result, so a reviewer can always see how much
interpretation was involved.

1. ``exact`` - case-insensitive equality against ``rxnorm_concept.str``.
   Similarity is 1.0. This is the only tier that involves no interpretation.

2. ``normalized`` - equality against the ``normalized`` column, comparing
   :func:`search_key` of the query. Normalisation casefolds, strips punctuation,
   and drops dose numbers, dose units and dosage-form words, so
   "METFORMIN HCL 500 MG TABLET" is compared as "metformin hcl". Similarity 1.0:
   nothing was guessed, only formatting discarded.

3. ``approximate`` - trigram similarity against ``normalized``, requiring
   ``settings.rxnorm_min_similarity``. Ties are broken by term-type preference
   (IN before PIN before MIN before BN) and then by the shorter string, so a
   plain ingredient wins over a brand or a combination product. Below the
   threshold nothing is returned: a near miss is not a match.

4. ``truncated`` - only reached when tiers 1-3 all fail, which would otherwise
   mean an unresolved drug. The trailing word of the search key is dropped and
   tier 2 retried, repeating while at least one word remains. This recovers salt
   qualifiers OCR spells differently from RxNorm ("metformin hcl" against
   "metformin hydrochloride"). It runs last on purpose: a combination product
   such as "lisinopril hydrochlorothiazide" matches its multi-ingredient concept
   at tier 3, so truncation never gets the chance to silently discard an
   ingredient. Similarity is reported as the fraction of the key retained.

Anything that survives none of the four tiers becomes an
:class:`UnresolvedConcept`, which the verifier turns into "ask your pharmacist".
The resolver never invents an RxCUI and never returns a best guess.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import structlog
from psycopg import AsyncConnection
from psycopg.rows import TupleRow, dict_row

from app.config import Settings
from app.db import Database
from app.domain.medication import (
    ResolvedConcept,
    RxNormResolution,
    UnresolvedConcept,
    UnresolvedReason,
)

_LOGGER = structlog.stdlib.get_logger("app.adapters.rxnorm")

# Dropped when building a search key: these describe strength or presentation,
# never the identity of the drug.
_DOSE_UNITS = frozenset(
    {
        "mg",
        "mcg",
        "ug",
        "g",
        "gm",
        "kg",
        "ml",
        "l",
        "unit",
        "units",
        "iu",
        "meq",
        "mmol",
        "mg24hr",
        "percent",
    }
)
_FORM_WORDS = frozenset(
    {
        "tablet",
        "tablets",
        "tab",
        "tabs",
        "capsule",
        "capsules",
        "cap",
        "caps",
        "caplet",
        "caplets",
        "pill",
        "pills",
        "er",
        "xr",
        "sr",
        "dr",
        "cr",
        "la",
        "xl",
        "odt",
        "oral",
        "orally",
        "solution",
        "suspension",
        "syrup",
        "elixir",
        "injection",
        "injectable",
        "cream",
        "ointment",
        "gel",
        "lotion",
        "patch",
        "inhaler",
        "inhalation",
        "spray",
        "drop",
        "drops",
        "suppository",
        "po",
        "hfa",
    }
)

# Preference order when several concepts tie on similarity. A plain ingredient
# is the least specific claim, so it is the safest of the available answers.
_TTY_PRIORITY = {"IN": 0, "PIN": 1, "MIN": 2, "BN": 3}
_TTY_FALLBACK_PRIORITY = 9


def normalize(text: str) -> str:
    """Casefold, replace non-alphanumerics with spaces, collapse whitespace."""
    folded = text.casefold()
    cleaned = "".join(char if char.isalnum() else " " for char in folded)
    return " ".join(cleaned.split())


def search_key(text: str) -> str:
    """Return the normalised form with strength and dosage-form words removed."""
    words = [
        word
        for word in normalize(text).split()
        if not word.isdigit() and word not in _DOSE_UNITS and word not in _FORM_WORDS
    ]
    return " ".join(words)


@runtime_checkable
class RxNormResolver(Protocol):
    """Resolves a drug string to an RxNorm concept, or reports failure."""

    async def resolve(self, drug_text: str) -> RxNormResolution:
        """Resolve ``drug_text``, never raising for an unrecognised drug."""
        ...


class PostgresRxNormResolver:
    """Resolver backed by the ``rxnorm_concept`` table."""

    def __init__(self, database: Database, settings: Settings) -> None:
        """Bind the resolver to a database and its similarity threshold."""
        self._database = database
        self._min_similarity = settings.rxnorm_min_similarity

    async def resolve(self, drug_text: str) -> RxNormResolution:
        """Descend the fallback ladder for ``drug_text``.

        Args:
            drug_text: Verbatim drug text as read by OCR.

        Returns:
            A :class:`ResolvedConcept` on a hit at any tier, otherwise an
            :class:`UnresolvedConcept` explaining which way it failed.
        """
        key = search_key(drug_text)
        if not key:
            return UnresolvedConcept(query=drug_text, reason="empty_query", best_similarity=0.0)

        async with self._database.connection() as conn:
            exact = await self._match_exact(conn, drug_text)
            if exact is not None:
                return exact

            normalized = await self._match_normalized(conn, drug_text, key, "normalized", 1.0)
            if normalized is not None:
                return normalized

            approximate, best_similarity = await self._match_approximate(conn, drug_text, key)
            if approximate is not None:
                return approximate

            truncated = await self._match_truncated(conn, drug_text, key)
            if truncated is not None:
                return truncated

        reason: UnresolvedReason = (
            "below_similarity_threshold" if best_similarity > 0.0 else "no_candidate"
        )
        _LOGGER.info("rxnorm_unresolved", reason=reason, best_similarity=round(best_similarity, 4))
        return UnresolvedConcept(
            query=drug_text, reason=reason, best_similarity=min(best_similarity, 1.0)
        )

    async def _match_exact(
        self, conn: AsyncConnection[TupleRow], drug_text: str
    ) -> ResolvedConcept | None:
        """Tier 1: case-insensitive equality on the stored string."""
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                """
                SELECT rxcui, tty, str
                FROM rxnorm_concept
                WHERE lower(str) = lower(%s)
                """,
                (drug_text,),
            )
            rows = await cur.fetchall()
        best = self._preferred(rows)
        if best is None:
            return None
        return ResolvedConcept(
            query=drug_text,
            rxcui=str(best["rxcui"]),
            tty=str(best["tty"]),
            matched_str=str(best["str"]),
            match_tier="exact",
            similarity=1.0,
        )

    async def _match_normalized(
        self,
        conn: AsyncConnection[TupleRow],
        drug_text: str,
        key: str,
        tier: str,
        similarity: float,
    ) -> ResolvedConcept | None:
        """Tier 2 (and tier 4's inner step): equality on the normalized column."""
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                """
                SELECT rxcui, tty, str
                FROM rxnorm_concept
                WHERE normalized = %s
                """,
                (key,),
            )
            rows = await cur.fetchall()
        best = self._preferred(rows)
        if best is None:
            return None
        return ResolvedConcept(
            query=drug_text,
            rxcui=str(best["rxcui"]),
            tty=str(best["tty"]),
            matched_str=str(best["str"]),
            match_tier="normalized" if tier == "normalized" else "truncated",
            similarity=similarity,
        )

    async def _match_approximate(
        self, conn: AsyncConnection[TupleRow], drug_text: str, key: str
    ) -> tuple[ResolvedConcept | None, float]:
        """Tier 3: trigram similarity, returning the best score either way."""
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                """
                SELECT rxcui, tty, str, similarity(normalized, %s) AS score
                FROM rxnorm_concept
                WHERE normalized %% %s
                ORDER BY score DESC
                LIMIT 25
                """,
                (key, key),
            )
            rows = await cur.fetchall()
        if not rows:
            return None, 0.0

        best_score = float(rows[0]["score"])
        if best_score < self._min_similarity:
            return None, best_score

        tied = [row for row in rows if float(row["score"]) == best_score]
        best = self._preferred(tied)
        if best is None:
            return None, best_score
        return (
            ResolvedConcept(
                query=drug_text,
                rxcui=str(best["rxcui"]),
                tty=str(best["tty"]),
                matched_str=str(best["str"]),
                match_tier="approximate",
                similarity=min(best_score, 1.0),
            ),
            best_score,
        )

    async def _match_truncated(
        self, conn: AsyncConnection[TupleRow], drug_text: str, key: str
    ) -> ResolvedConcept | None:
        """Tier 4: drop trailing words and retry the normalized tier."""
        words = key.split()
        for length in range(len(words) - 1, 0, -1):
            shortened = " ".join(words[:length])
            retained = length / len(words)
            match = await self._match_normalized(conn, drug_text, shortened, "truncated", retained)
            if match is not None:
                _LOGGER.info(
                    "rxnorm_truncated_match",
                    dropped_words=len(words) - length,
                    retained=round(retained, 3),
                )
                return match
        return None

    @staticmethod
    def _preferred(rows: list[dict[str, object]]) -> dict[str, object] | None:
        """Pick the safest row: preferred term type, then the shortest string."""
        if not rows:
            return None
        return min(
            rows,
            key=lambda row: (
                _TTY_PRIORITY.get(str(row["tty"]), _TTY_FALLBACK_PRIORITY),
                len(str(row["str"])),
                str(row["str"]),
            ),
        )
