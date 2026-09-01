"""Native desktop control center."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORT_MODULE = {
    "MainWindow": ".main_window",
    "THEMES": ".theme",
    "build_stylesheet": ".theme",
    "Card": ".widgets",
    "InfoPanel": ".widgets",
    "PathField": ".widgets",
}

__all__ = [
    "Card",
    "InfoPanel",
    "MainWindow",
    "PathField",
    "THEMES",
    "build_stylesheet",
]


def __getattr__(name: str) -> Any:
    module_name = _EXPORT_MODULE.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
