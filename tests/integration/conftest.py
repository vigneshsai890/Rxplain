"""Fixtures for tests that need a live Postgres.

These skip rather than fail when no database is reachable, so the unit suite
stays runnable on a machine without Docker. CI starts the service, so the skips
do not hide regressions there.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import psycopg
import pytest

from app.adapters.rxnorm import PostgresRxNormResolver
from app.adapters.rxnorm_seed import load_seed
from app.config import Settings
from app.db import Database


@pytest.fixture
async def database(settings: Settings) -> AsyncIterator[Database]:
    """An open database with migrations applied, or skip the test."""
    db = Database(settings)
    try:
        await db.open()
    except (psycopg.OperationalError, TimeoutError) as exc:
        pytest.skip(f"Postgres is not reachable: {exc}")
    try:
        await db.apply_migrations()
        yield db
    finally:
        await db.close()


@pytest.fixture
async def seeded_database(database: Database, settings: Settings) -> Database:
    """A database whose RxNorm tables are loaded from the committed seed."""
    async with database.connection() as conn:
        cur = await conn.execute("SELECT count(*) FROM rxnorm_concept")
        row = await cur.fetchone()
    if row is None or int(row[0]) == 0:
        await load_seed(database, settings.rxnorm_seed_path)
    return database


@pytest.fixture
async def resolver(seeded_database: Database, settings: Settings) -> PostgresRxNormResolver:
    """A resolver backed by the seeded database."""
    return PostgresRxNormResolver(seeded_database, settings)
