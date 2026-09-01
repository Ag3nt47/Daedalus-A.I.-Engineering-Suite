"""Runtime font recovery for headless and minimal Qt platform plugins.

Normal desktop Qt plugins enumerate operating-system fonts.  Some offscreen
plugins expose an empty font database, however, which makes otherwise valid
QSS family fallbacks render as tofu squares.  Daedalus does not redistribute
system fonts; it registers a small set of known local files only when Qt has no
families and then installs substitutions for the theme's semantic families.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtWidgets import QApplication


@dataclass(frozen=True, slots=True)
class RuntimeFonts:
    ui_family: str
    code_family: str
    registered_files: tuple[str, ...]


def _known_font_files() -> tuple[tuple[str, Path], ...]:
    if sys.platform == "win32":
        windows = Path(os.environ.get("WINDIR", r"C:\Windows"))
        fonts = windows / "Fonts"
        return (
            ("ui", fonts / "segoeui.ttf"),
            ("ui", fonts / "segoeuib.ttf"),
            ("mono", fonts / "consola.ttf"),
            ("mono", fonts / "consolab.ttf"),
            ("ui", fonts / "arial.ttf"),
            ("mono", fonts / "cour.ttf"),
        )
    if sys.platform == "darwin":
        return (
            ("ui", Path("/System/Library/Fonts/Helvetica.ttc")),
            ("ui", Path("/System/Library/Fonts/Supplemental/Arial.ttf")),
            ("mono", Path("/System/Library/Fonts/Menlo.ttc")),
            ("mono", Path("/System/Library/Fonts/Monaco.ttf")),
        )
    return (
        ("ui", Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")),
        ("ui", Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")),
        ("mono", Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf")),
        ("mono", Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf")),
        ("ui", Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf")),
        ("mono", Path("/usr/share/fonts/truetype/liberation2/LiberationMono-Regular.ttf")),
    )


def _choose(families: list[str], preferences: tuple[str, ...], fallback: str) -> str:
    lookup = {family.casefold(): family for family in families}
    for preferred in preferences:
        if match := lookup.get(preferred.casefold()):
            return match
    return families[0] if families else fallback


@lru_cache(maxsize=1)
def ensure_runtime_fonts() -> RuntimeFonts:
    """Ensure Qt can resolve a UI and code face, returning their family names.

    This must run after ``QApplication`` construction.  On a healthy desktop
    font database it performs no file registration; substitutions still make
    the semantic theme families deterministic across operating systems.
    """

    application = QApplication.instance()
    if application is None:
        raise RuntimeError("ensure_runtime_fonts requires an active QApplication")

    families = list(QFontDatabase.families())
    registered_files: list[str] = []
    role_families: dict[str, list[str]] = {"ui": [], "mono": []}
    if not families:
        for role, path in _known_font_files():
            # One regular face per role is sufficient; Qt can synthesize bold
            # and italic for diagnostics, and font registration can be costly
            # on a platform plugin that has no native font database.
            if role_families[role]:
                continue
            if not path.is_file():
                continue
            font_id = QFontDatabase.addApplicationFont(str(path))
            if font_id < 0:
                continue
            registered_files.append(str(path))
            role_families[role].extend(QFontDatabase.applicationFontFamilies(font_id))
        families = list(QFontDatabase.families())

    ui_candidates = [*role_families["ui"], *families]
    code_candidates = [*role_families["mono"], *families]
    ui_family = _choose(
        ui_candidates,
        ("Segoe UI", "Inter", "Helvetica Neue", "DejaVu Sans", "Liberation Sans", "Arial"),
        application.font().family() or "Sans Serif",
    )
    code_family = _choose(
        code_candidates,
        ("Cascadia Code", "JetBrains Mono", "Fira Code", "Consolas", "DejaVu Sans Mono", "Menlo"),
        ui_family,
    )

    for semantic in ("Segoe UI", "Inter", "Helvetica Neue", "sans-serif"):
        if semantic.casefold() != ui_family.casefold():
            QFont.insertSubstitution(semantic, ui_family)
    for semantic in (
        "Cascadia Code",
        "JetBrains Mono",
        "Fira Code",
        "Consolas",
        "monospace",
    ):
        if semantic.casefold() != code_family.casefold():
            QFont.insertSubstitution(semantic, code_family)

    application.setFont(QFont(ui_family, application.font().pointSize()))
    return RuntimeFonts(ui_family, code_family, tuple(registered_files))


__all__ = ["RuntimeFonts", "ensure_runtime_fonts"]
