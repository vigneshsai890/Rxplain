"""Regenerate ``requirements.txt`` from ``pyproject.toml``.

Vercel's Python runtime installs from ``requirements.txt`` and ignores
``pyproject.toml``, so the deployed dependency set is described in a second file.
Rather than maintain both by hand, this generates one from the other and a test
asserts they match, which turns drift into a failing build instead of a
production-only difference.

Run from the repository root::

    .venv/bin/python scripts/sync_requirements.py
"""

from __future__ import annotations

import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = REPO_ROOT / "pyproject.toml"
REQUIREMENTS = REPO_ROOT / "requirements.txt"

HEADER = (
    "# Generated from pyproject.toml [project.dependencies] by "
    "scripts/sync_requirements.py.\n"
    "# Vercel's Python runtime installs from requirements.txt, not from pyproject.\n"
    "# A test asserts the two stay identical, so this cannot drift silently.\n"
)


def runtime_dependencies() -> list[str]:
    """Return the pinned runtime dependencies declared in pyproject."""
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    project = data["project"]
    dependencies = project["dependencies"]
    if not isinstance(dependencies, list):
        raise TypeError("[project.dependencies] must be a list")
    return [item for item in dependencies if isinstance(item, str)]


def render() -> str:
    """Return the full requirements.txt content."""
    return HEADER + "\n".join(runtime_dependencies()) + "\n"


if __name__ == "__main__":
    REQUIREMENTS.write_text(render(), encoding="utf-8")
    print(f"wrote {REQUIREMENTS.relative_to(REPO_ROOT)}")
