"""Command-line entry point for loading RxNorm data into Postgres.

The loading logic lives in :mod:`app.adapters.rxnorm_seed`, because the
application also seeds itself on first start. This module is only the CLI.

Usage::

    .venv/bin/python scripts/load_rxnorm.py load-seed
    .venv/bin/python scripts/load_rxnorm.py load-rxnconso /path/to/RXNCONSO.RRF

``RXNCONSO.RRF`` comes from a UMLS RxNorm release, which is licensed and cannot
be committed here, so point the second command at your own copy.
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from app.adapters.rxnorm_seed import load_rxnconso, load_seed
from app.config import Settings
from app.db import Database


async def _main(args: argparse.Namespace) -> None:
    """Open the database, run the requested command, then close."""
    settings = Settings()
    database = Database(settings)
    await database.open()
    try:
        await database.apply_migrations()
        if args.command == "load-seed":
            concepts, relations = await load_seed(database, settings.rxnorm_seed_path)
            print(f"loaded {concepts} concepts and {relations} relations from the seed")
        else:
            count = await load_rxnconso(database, Path(args.path))
            print(f"loaded {count} concepts from {args.path}")
    finally:
        await database.close()


def _parse_args() -> argparse.Namespace:
    """Build the command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("load-seed", help="load the committed RxNav seed")
    rrf = sub.add_parser("load-rxnconso", help="load a UMLS RXNCONSO.RRF file")
    rrf.add_argument("path", help="path to RXNCONSO.RRF")
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(_main(_parse_args()))
