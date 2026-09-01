"""Packaged, offline-first learning and diagnostic resources.

The GUI and command-line tools consume these files through :func:`load_json`
instead of depending on a source checkout.  Keeping the loader deliberately
small also makes the content usable by tests and third-party extensions.
"""

from __future__ import annotations

import json
from importlib.resources import files
from typing import Any

RESOURCE_FILES = frozenset(
    {
        "error_cards.json",
        "glossary.json",
        "learning_paths.json",
        "project_recipes.json",
        "sources.json",
    }
)


def load_json(name: str) -> dict[str, Any]:
    """Load one known Daedalus resource file from the installed package."""

    if name not in RESOURCE_FILES:
        raise ValueError(f"Unknown Daedalus resource: {name}")
    resource = files(__package__).joinpath(name)
    with resource.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Daedalus resource must contain a JSON object: {name}")
    return data


__all__ = ["RESOURCE_FILES", "load_json"]
