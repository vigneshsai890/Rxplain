"""Loading RxNorm concepts into Postgres.

Two sources feed one table, and each row records which it came from:

``RXNAV_API``
    The committed seed at ``migrations/seed/rxnorm_seed.json``, fetched from the
    public NLM RxNav API by ``scripts/fetch_rxnorm_seed.py``. This needs no
    licence and no network at load time, so it is what a local run, CI, and the
    container all use.

``RXNCONSO``
    A full ``RXNCONSO.RRF`` from a UMLS RxNorm release. That file is licensed and
    cannot be committed or fetched unattended, so :func:`load_rxnconso` is pointed
    at an operator's own copy by ``scripts/load_rxnorm.py``.

This module lives under ``app`` rather than ``scripts`` because the application
seeds itself on first start: a container with an empty database would otherwise
resolve nothing and report every drug as unverified.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Final, NamedTuple

import anyio.to_thread
import structlog

from app.adapters.rxnorm import normalize
from app.db import Database

_LOGGER = structlog.stdlib.get_logger("app.adapters.rxnorm_seed")

# RXNCONSO.RRF is pipe-delimited with these columns, in this order. Documented at
# https://www.nlm.nih.gov/research/umls/rxnorm/docs/techdoc.html
RXNCONSO_COLUMNS: Final = (
    "RXCUI",
    "LAT",
    "TS",
    "LUI",
    "STT",
    "SUI",
    "ISPREF",
    "RXAUI",
    "SAUI",
    "SCUI",
    "SDUI",
    "SAB",
    "TTY",
    "CODE",
    "STR",
    "SRL",
    "SUPPRESS",
    "CVF",
)
_IDX: Final = {name: index for index, name in enumerate(RXNCONSO_COLUMNS)}

# Term types a prescription's drug name can resolve onto. Product-level types are
# excluded: those are loaded as ingredient/product relations instead.
WANTED_TTYS: Final = frozenset({"IN", "PIN", "MIN", "BN", "SBDC", "SCDC"})


class ConceptRow(NamedTuple):
    """A row destined for ``rxnorm_concept``."""

    rxcui: str
    tty: str
    text: str
    normalized: str
    source: str


def parse_rxnconso(lines: Iterator[str]) -> Iterator[ConceptRow]:
    """Yield loadable concept rows from RXNCONSO.RRF lines.

    Keeps English, non-suppressed rows whose term type is in
    :data:`WANTED_TTYS`. Malformed short lines are skipped rather than guessed at,
    because a partially parsed drug name is worse than a missing one.

    Args:
        lines: Raw lines from an RXNCONSO.RRF file.

    Yields:
        One :class:`ConceptRow` per accepted line.
    """
    for line in lines:
        fields = line.rstrip("\n").split("|")
        if len(fields) < len(RXNCONSO_COLUMNS):
            continue
        if fields[_IDX["LAT"]] != "ENG":
            continue
        if fields[_IDX["SUPPRESS"]] not in {"N", ""}:
            continue
        tty = fields[_IDX["TTY"]]
        if tty not in WANTED_TTYS:
            continue
        text = fields[_IDX["STR"]].strip()
        rxcui = fields[_IDX["RXCUI"]].strip()
        if not text or not rxcui.isdigit():
            continue
        yield ConceptRow(rxcui, tty, text, normalize(text), "RXNCONSO")


def seed_rows(payload: dict[str, object]) -> tuple[list[ConceptRow], list[tuple[str, str]]]:
    """Split a seed document into concept rows and ingredient/product pairs.

    Raises:
        TypeError: If the seed is not shaped as ``fetch_rxnorm_seed.py`` writes.
    """
    concepts = payload.get("concepts")
    relations = payload.get("ingredient_products")
    if not isinstance(concepts, list) or not isinstance(relations, list):
        raise TypeError("seed must contain 'concepts' and 'ingredient_products' lists")

    concept_rows: list[ConceptRow] = []
    for item in concepts:
        if not isinstance(item, dict):
            raise TypeError("every concept must be an object")
        rxcui, tty, text = item.get("rxcui"), item.get("tty"), item.get("str")
        if not (isinstance(rxcui, str) and isinstance(tty, str) and isinstance(text, str)):
            raise TypeError(f"malformed concept: {item!r}")
        concept_rows.append(ConceptRow(rxcui, tty, text, normalize(text), "RXNAV_API"))

    pairs: list[tuple[str, str]] = []
    for item in relations:
        if not isinstance(item, dict):
            raise TypeError("every relation must be an object")
        ingredient, product = item.get("ingredient_rxcui"), item.get("product_rxcui")
        if not (isinstance(ingredient, str) and isinstance(product, str)):
            raise TypeError(f"malformed relation: {item!r}")
        pairs.append((ingredient, product))
    return concept_rows, pairs


async def insert_concepts(database: Database, rows: list[ConceptRow]) -> int:
    """Upsert concept rows, returning the number sent."""
    if not rows:
        return 0
    async with database.connection() as conn, conn.cursor() as cur:
        await cur.executemany(
            """
            INSERT INTO rxnorm_concept (rxcui, tty, str, normalized, source)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (rxcui, tty, str)
            DO UPDATE SET normalized = EXCLUDED.normalized, source = EXCLUDED.source
            """,
            rows,
        )
        await conn.commit()
    return len(rows)


async def insert_relations(database: Database, pairs: list[tuple[str, str]], source: str) -> int:
    """Upsert ingredient/product relations, returning the number sent."""
    if not pairs:
        return 0
    async with database.connection() as conn, conn.cursor() as cur:
        await cur.executemany(
            """
            INSERT INTO rxnorm_ingredient_product (ingredient_rxcui, product_rxcui, source)
            VALUES (%s, %s, %s)
            ON CONFLICT (ingredient_rxcui, product_rxcui) DO NOTHING
            """,
            [(ingredient, product, source) for ingredient, product in pairs],
        )
        await conn.commit()
    return len(pairs)


async def load_seed(database: Database, seed_path: Path) -> tuple[int, int]:
    """Load the committed RxNav seed. Returns ``(concepts, relations)``.

    Raises:
        TypeError: If the seed file does not contain a JSON object.
    """
    raw = await anyio.to_thread.run_sync(seed_path.read_bytes)
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise TypeError("seed file must contain a JSON object")
    concept_rows, pairs = seed_rows(payload)
    return (
        await insert_concepts(database, concept_rows),
        await insert_relations(database, pairs, "RXNAV_API"),
    )


async def load_rxnconso(database: Database, path: Path, batch_size: int = 5000) -> int:
    """Stream an RXNCONSO.RRF into ``rxnorm_concept``. Returns rows loaded."""
    loaded = 0
    batch: list[ConceptRow] = []
    with path.open(encoding="utf-8", errors="strict") as handle:
        for row in parse_rxnconso(handle):
            batch.append(row)
            if len(batch) >= batch_size:
                loaded += await insert_concepts(database, batch)
                batch.clear()
    loaded += await insert_concepts(database, batch)
    return loaded


async def concept_count(database: Database) -> int:
    """Return how many RxNorm concepts are loaded."""
    async with database.connection() as conn:
        cur = await conn.execute("SELECT count(*) FROM rxnorm_concept")
        row = await cur.fetchone()
    return 0 if row is None else int(row[0])


async def seed_if_empty(database: Database, seed_path: Path) -> bool:
    """Load the seed when the concept table is empty.

    Returns:
        Whether a load was performed.
    """
    if await concept_count(database) > 0:
        return False
    concepts, relations = await load_seed(database, seed_path)
    _LOGGER.info("rxnorm_seed_loaded", concepts=concepts, relations=relations)
    return True
