"""Postgres access: a connection pool and a checksum-verified migration runner.

Migrations are plain ``.sql`` files applied in filename order. Each applied file
is recorded with a SHA-256 checksum, so editing a migration that has already
run is an error rather than a silent divergence between environments.
"""

from __future__ import annotations

import hashlib
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import structlog
from psycopg import AsyncConnection
from psycopg.rows import TupleRow, dict_row
from psycopg_pool import AsyncConnectionPool

from app.config import Settings

_LOGGER = structlog.stdlib.get_logger("app.db")

_MIGRATION_TABLE = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    filename   text PRIMARY KEY,
    checksum   text NOT NULL,
    applied_at timestamptz NOT NULL DEFAULT now()
)
"""


class DatabaseNotOpenError(RuntimeError):
    """Raised when the pool is used before ``open()`` or after ``close()``."""


class MigrationChecksumError(RuntimeError):
    """Raised when an already-applied migration file has been modified."""


class Database:
    """Owns the connection pool for the process."""

    def __init__(self, settings: Settings) -> None:
        """Store configuration; no connection is made until ``open()``."""
        self._settings = settings
        self._pool: AsyncConnectionPool[AsyncConnection[TupleRow]] | None = None

    async def open(self) -> None:
        """Create the pool and wait until at least one connection is live."""
        pool: AsyncConnectionPool[AsyncConnection[TupleRow]] = AsyncConnectionPool(
            conninfo=self._settings.database_url.get_secret_value(),
            min_size=self._settings.database_pool_min_size,
            max_size=self._settings.database_pool_max_size,
            open=False,
        )
        await pool.open(wait=True, timeout=30.0)
        self._pool = pool

    async def close(self) -> None:
        """Close the pool if it is open. Safe to call more than once."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    @asynccontextmanager
    async def connection(self) -> AsyncIterator[AsyncConnection[TupleRow]]:
        """Check a connection out of the pool for the duration of the block."""
        if self._pool is None:
            raise DatabaseNotOpenError("database pool is not open")
        async with self._pool.connection() as conn:
            yield conn

    async def apply_migrations(self) -> tuple[str, ...]:
        """Apply pending migrations in filename order.

        Returns:
            The filenames applied by this call, in the order applied.

        Raises:
            MigrationChecksumError: If a previously applied file has changed.
        """
        files = sorted(self._settings.migrations_dir.glob("*.sql"))
        applied: list[str] = []
        async with self.connection() as conn:
            await conn.execute(_MIGRATION_TABLE)
            await conn.commit()

            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute("SELECT filename, checksum FROM schema_migrations")
                known = {str(row["filename"]): str(row["checksum"]) for row in await cur.fetchall()}

            for path in files:
                body = path.read_text(encoding="utf-8")
                checksum = hashlib.sha256(body.encode("utf-8")).hexdigest()
                previous = known.get(path.name)
                if previous == checksum:
                    continue
                if previous is not None:
                    raise MigrationChecksumError(
                        f"migration {path.name} changed after being applied"
                    )
                await conn.execute(body)
                await conn.execute(
                    "INSERT INTO schema_migrations (filename, checksum) VALUES (%s, %s)",
                    (path.name, checksum),
                )
                await conn.commit()
                applied.append(path.name)

        if applied:
            _LOGGER.info("migrations_applied", count=len(applied), files=applied)
        return tuple(applied)
