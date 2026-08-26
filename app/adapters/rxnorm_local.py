"""In-process RxNorm resolution, for deployments without a database.

The committed seed is 310 concepts. At that size a database buys nothing: the
whole index fits in memory, every lookup is a dict hit, and the similarity tier is
a set operation over a few hundred candidates. Removing the dependency means the
application deploys to a platform with no provisioned Postgres and no connection
pool to exhaust, which is what serverless hosting requires.

This is not a different algorithm. :func:`similarity` is a port of PostgreSQL's
``pg_trgm``, verified against a live server:

===============================  =============================  ==========
a                                b                              similarity
===============================  =============================  ==========
metformin hcl                    metformin                        0.714286
metformin hcl                    metformin hydrochloride          0.407407
lisinopril hydrochlorothiazide   hydrochlorothiazide lisinopril   1.000000
metformen                        metformin                        0.538462
amlodipine besylate              amlodipine                       0.550000
zyntrafex                        metformin                        0.000000
===============================  =============================  ==========

``tests/unit/test_rxnorm_local.py`` asserts each of those figures, so a drift in
the port is a failing test rather than a silently different resolution.

The fallback ladder is the same four tiers in the same order, documented in
:mod:`app.adapters.rxnorm`. Behaviour is identical; only the storage differs.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Final

import structlog

from app.adapters.rxnorm import _TTY_PRIORITY, normalize, search_key
from app.config import Settings
from app.domain.medication import (
    ResolvedConcept,
    RxNormResolution,
    UnresolvedConcept,
    UnresolvedReason,
)

_LOGGER = structlog.stdlib.get_logger("app.adapters.rxnorm_local")

_TTY_FALLBACK_PRIORITY: Final = 9

# pg_trgm pads each word with two leading spaces and one trailing space before
# taking 3-character substrings. Reproduced exactly; see the table above.
_PREFIX: Final = "  "
_SUFFIX: Final = " "
_TRIGRAM_SIZE: Final = 3

# PostgreSQL's `%` operator pre-filters candidates at pg_trgm.similarity_threshold
# before any score is compared, so a query with no candidate above it returns zero
# rows and reports no best score at all. Applying the same floor keeps the
# *reason* an unresolved drug carries identical between the two backends, which
# matters because that reason is what the interface tells the patient.
_PG_TRGM_PREFILTER: Final = 0.3


def trigrams(text: str) -> frozenset[str]:
    """Return the ``pg_trgm`` trigram set of ``text``.

    Args:
        text: Any string. Non-alphanumerics act as word separators.

    Returns:
        The distinct trigrams, matching ``show_trgm()``.
    """
    result: set[str] = set()
    for word in normalize(text).split():
        padded = f"{_PREFIX}{word}{_SUFFIX}"
        for index in range(len(padded) - _TRIGRAM_SIZE + 1):
            result.add(padded[index : index + _TRIGRAM_SIZE])
    return frozenset(result)


def similarity(left: str, right: str) -> float:
    """Return ``pg_trgm`` similarity: the Jaccard index of the trigram sets."""
    first, second = trigrams(left), trigrams(right)
    union = first | second
    if not union:
        return 0.0
    return len(first & second) / len(union)


class Concept:
    """One RxNorm concept held in memory."""

    __slots__ = ("normalized", "rxcui", "trigrams", "tty", "value")

    def __init__(self, rxcui: str, tty: str, value: str) -> None:
        """Precompute the normalised form and trigram set for this concept."""
        self.rxcui = rxcui
        self.tty = tty
        self.value = value
        self.normalized = normalize(value)
        self.trigrams = trigrams(value)

    @property
    def priority(self) -> tuple[int, int, str]:
        """Tie-break key: preferred term type, then the shorter string."""
        return (
            _TTY_PRIORITY.get(self.tty, _TTY_FALLBACK_PRIORITY),
            len(self.value),
            self.value,
        )


class InProcessRxNormResolver:
    """The four-tier resolution ladder, served from memory."""

    def __init__(self, settings: Settings) -> None:
        """Load and index the committed seed.

        Raises:
            FileNotFoundError: If the seed file is missing.
            TypeError: If the seed is not shaped as expected.
        """
        self._min_similarity = settings.rxnorm_min_similarity
        self._concepts: list[Concept] = []
        self._by_lower_value: dict[str, list[Concept]] = defaultdict(list)
        self._by_normalized: dict[str, list[Concept]] = defaultdict(list)
        self._products: dict[str, frozenset[str]] = {}
        self._load(settings.rxnorm_seed_path)

    def _load(self, seed_path: Path) -> None:
        """Read the seed and build the lookup indexes."""
        payload = json.loads(seed_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError("seed file must contain a JSON object")
        concepts = payload.get("concepts")
        relations = payload.get("ingredient_products")
        if not isinstance(concepts, list) or not isinstance(relations, list):
            raise TypeError("seed must contain 'concepts' and 'ingredient_products' lists")

        for item in concepts:
            if not isinstance(item, dict):
                continue
            rxcui, tty, value = item.get("rxcui"), item.get("tty"), item.get("str")
            if not (isinstance(rxcui, str) and isinstance(tty, str) and isinstance(value, str)):
                continue
            concept = Concept(rxcui, tty, value)
            self._concepts.append(concept)
            self._by_lower_value[value.casefold()].append(concept)
            self._by_normalized[concept.normalized].append(concept)

        products: dict[str, set[str]] = defaultdict(set)
        for item in relations:
            if not isinstance(item, dict):
                continue
            ingredient, product = item.get("ingredient_rxcui"), item.get("product_rxcui")
            if isinstance(ingredient, str) and isinstance(product, str):
                products[ingredient].add(product)
        self._products = {key: frozenset(value) for key, value in products.items()}

        _LOGGER.info(
            "rxnorm_index_built",
            concepts=len(self._concepts),
            ingredients_with_products=len(self._products),
        )

    def product_rxcuis(self, ingredient_rxcui: str) -> frozenset[str]:
        """Return the product RxCUIs recorded for an ingredient."""
        return self._products.get(ingredient_rxcui, frozenset())

    @property
    def concept_count(self) -> int:
        """How many concepts are indexed."""
        return len(self._concepts)

    async def resolve(self, drug_text: str) -> RxNormResolution:
        """Descend the same four tiers as the database-backed resolver."""
        key = search_key(drug_text)
        if not key:
            return UnresolvedConcept(query=drug_text, reason="empty_query", best_similarity=0.0)

        exact = self._best(self._by_lower_value.get(drug_text.casefold(), []))
        if exact is not None:
            return self._resolved(drug_text, exact, "exact", 1.0)

        normalized = self._best(self._by_normalized.get(key, []))
        if normalized is not None:
            return self._resolved(drug_text, normalized, "normalized", 1.0)

        approximate, best_score = self._approximate(key)
        if approximate is not None:
            return self._resolved(drug_text, approximate, "approximate", best_score)

        truncated = self._truncated(key)
        if truncated is not None:
            concept, retained = truncated
            return self._resolved(drug_text, concept, "truncated", retained)

        reason: UnresolvedReason = (
            "below_similarity_threshold" if best_score > 0.0 else "no_candidate"
        )
        return UnresolvedConcept(
            query=drug_text, reason=reason, best_similarity=min(best_score, 1.0)
        )

    def _approximate(self, key: str) -> tuple[Concept | None, float]:
        """Tier 3: best trigram match, returning the top score either way."""
        query = trigrams(key)
        if not query:
            return None, 0.0
        best_score = 0.0
        tied: list[Concept] = []
        for concept in self._concepts:
            union = query | concept.trigrams
            score = len(query & concept.trigrams) / len(union) if union else 0.0
            if score < _PG_TRGM_PREFILTER:
                continue
            if score > best_score:
                best_score, tied = score, [concept]
            elif score == best_score and score > 0.0:
                tied.append(concept)
        if best_score < self._min_similarity:
            return None, best_score
        return self._best(tied), best_score

    def _truncated(self, key: str) -> tuple[Concept, float] | None:
        """Tier 4: drop trailing words and retry the normalised tier."""
        words = key.split()
        for length in range(len(words) - 1, 0, -1):
            candidate = self._best(self._by_normalized.get(" ".join(words[:length]), []))
            if candidate is not None:
                return candidate, length / len(words)
        return None

    @staticmethod
    def _best(candidates: list[Concept]) -> Concept | None:
        """Pick the safest candidate: preferred term type, then shortest string."""
        if not candidates:
            return None
        return min(candidates, key=lambda concept: concept.priority)

    @staticmethod
    def _resolved(drug_text: str, concept: Concept, tier: str, score: float) -> ResolvedConcept:
        """Build a resolution result for a matched concept."""
        return ResolvedConcept(
            query=drug_text,
            rxcui=concept.rxcui,
            tty=concept.tty,
            matched_str=concept.value,
            match_tier="exact"
            if tier == "exact"
            else "normalized"
            if tier == "normalized"
            else "approximate"
            if tier == "approximate"
            else "truncated",
            similarity=min(score, 1.0),
        )
