"""Theme tokens and responsive Qt style generation for the desktop workbench."""

from __future__ import annotations

import os
from dataclasses import dataclass, replace


@dataclass(frozen=True, slots=True)
class ThemeTokens:
    key: str
    label: str
    background: str
    surface: str
    raised: str
    border: str
    text: str
    muted: str
    accent: str
    accent_alt: str
    success: str
    warning: str
    danger: str
    code_background: str
    radius: int
    ui_font: str
    code_font: str


_UI_FONT = "'Segoe UI','Inter','Helvetica Neue',sans-serif"
_MONO_FONT = "'Cascadia Code','JetBrains Mono','Fira Code','Consolas',monospace"

THEMES: dict[str, ThemeTokens] = {
    "slate": ThemeTokens(
        "slate",
        "Dark Slate",
        "#020617",
        "#0f172a",
        "#1e293b",
        "#334155",
        "#f1f5f9",
        "#94a3b8",
        "#38bdf8",
        "#22d3a6",
        "#34d399",
        "#fbbf24",
        "#fb7185",
        "#070d19",
        9,
        _UI_FONT,
        _MONO_FONT,
    ),
    "cyber": ThemeTokens(
        "cyber",
        "Cyber Blueprint",
        "#070b12",
        "#0c1420",
        "#112033",
        "#1d3851",
        "#d8e8f5",
        "#7890a6",
        "#20a4f3",
        "#f97316",
        "#2dd4bf",
        "#f59e0b",
        "#f43f5e",
        "#050a10",
        7,
        _MONO_FONT,
        _MONO_FONT,
    ),
    "crt": ThemeTokens(
        "crt",
        "Phosphor CRT",
        "#000900",
        "#001500",
        "#002100",
        "#0a6931",
        "#56ff85",
        "#27954b",
        "#39ff6f",
        "#c7ff39",
        "#5cff8d",
        "#ffe45c",
        "#ff6b6b",
        "#000600",
        2,
        _MONO_FONT,
        _MONO_FONT,
    ),
}


def available_themes() -> tuple[tuple[str, str], ...]:
    return tuple((key, tokens.label) for key, tokens in THEMES.items())


def clamp_scale(value: float) -> float:
    try:
        scale = float(value)
    except (TypeError, ValueError):
        scale = 1.0
    return max(0.78, min(1.35, scale))


def reduced_motion(config: object | None = None) -> bool:
    """Return the effective reduced-motion preference.

    An environment override is useful for accessibility, automated tests, and
    remote desktops.  A config object may expose either ``reduced_motion`` or
    the inverse ``ui_motion_enabled`` convention.
    """

    raw = os.environ.get("DAEDALUS_REDUCE_MOTION", "").strip().casefold()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if config is not None:
        if hasattr(config, "reduced_motion"):
            return bool(getattr(config, "reduced_motion"))
        if hasattr(config, "ui_motion_enabled"):
            return not bool(getattr(config, "ui_motion_enabled"))
    return False


def motion_allowed(config: object | None = None) -> bool:
    return not reduced_motion(config)


def _mix(foreground: str, background: str, alpha: float) -> str:
    """Blend two ``#RRGGBB`` colors for QSS implementations without alpha."""

    try:
        fg = tuple(int(foreground[index : index + 2], 16) for index in (1, 3, 5))
        bg = tuple(int(background[index : index + 2], 16) for index in (1, 3, 5))
    except (TypeError, ValueError):
        return foreground
    values = tuple(round(bg[i] + (fg[i] - bg[i]) * alpha) for i in range(3))
    return "#" + "".join(f"{value:02x}" for value in values)


def tokens_for(name: str = "slate", accent: str | None = None) -> ThemeTokens:
    tokens = THEMES.get(str(name).casefold(), THEMES["slate"])
    if accent and len(accent) == 7 and accent.startswith("#"):
        tokens = replace(tokens, accent=accent)
    return tokens


def build_stylesheet(
    name: str = "slate",
    accent: str | None = None,
    *,
    scale: float = 1.0,
) -> str:
    """Build the application QSS from one palette and one scale factor."""

    t = tokens_for(name, accent)
    factor = clamp_scale(scale)

    def px(value: float) -> str:
        return f"{max(1, round(value * factor))}px"

    accent_soft = _mix(t.accent, t.surface, 0.16)
    accent_hover = _mix(t.accent, t.raised, 0.28)
    control_border = _mix(t.text, t.surface, 0.42)
    success_soft = _mix(t.success, t.surface, 0.14)
    warning_soft = _mix(t.warning, t.surface, 0.14)
    danger_soft = _mix(t.danger, t.surface, 0.18)
    return f"""
* {{
    color: {t.text};
    font-family: {t.ui_font};
    font-size: {px(13)};
}}
QMainWindow, QDialog, QWidget#AppRoot, QWidget[workspacePage="true"] {{
    background: {t.background};
}}
QLabel, QCheckBox, QRadioButton {{ background: transparent; }}
QWidget#qt_scrollarea_viewport, QScrollArea {{
    background: {t.background}; border: 0;
}}
QLabel#Brand {{
    color: {t.accent}; font-size: {px(24)}; font-weight: 800;
    letter-spacing: {px(4)};
}}
QLabel#BrandTagline {{ color: {t.muted}; font-size: {px(10)}; letter-spacing: {px(1)}; }}
QLabel#PageTitle {{ color: {t.text}; font-size: {px(23)}; font-weight: 800; }}
QLabel#PageSubtitle {{ color: {t.muted}; font-size: {px(12)}; }}
QLabel#SectionTitle {{ color: {t.accent}; font-size: {px(13)}; font-weight: 700; }}
QLabel#Muted {{ color: {t.muted}; }}
QLabel#Success {{ color: {t.success}; font-weight: 700; }}
QLabel#Warning {{ color: {t.warning}; font-weight: 700; }}
QLabel#Danger {{ color: {t.danger}; font-weight: 700; }}
QLabel#MetricValue {{
    color: {t.text}; font-size: {px(25)}; font-weight: 800;
}}

QFrame#AppHeader, QFrame#Sidebar, QFrame#StatusStrip {{
    background: {t.surface}; border: 1px solid {t.border};
}}
QFrame#AppHeader {{ border-width: 0 0 1px 0; }}
QFrame#Sidebar {{ border-width: 0 1px 0 0; }}
QFrame#StatusStrip {{ border-width: 1px 0 0 0; }}
QScrollArea#NavScroll, QWidget[navViewport="true"], QWidget#NavHost {{
    background: {t.surface}; border: 0;
}}
QFrame#Card, QFrame#InfoPanel, QFrame#PathField {{
    background: {t.surface}; border: 1px solid {t.border};
    border-radius: {px(t.radius)};
}}
QFrame#Card[accent="true"] {{
    border-left: {px(3)} solid {t.accent};
    background: {_mix(t.raised, t.surface, 0.22)};
}}
QFrame#InfoPanel {{ background: {accent_soft}; }}
QFrame#PathField[valid="false"] {{ border-color: {t.danger}; }}

QPushButton, QToolButton {{
    background: {t.raised}; color: {t.text}; border: 1px solid {control_border};
    border-bottom: {px(2)} solid {control_border}; border-radius: {px(t.radius)};
    padding: {px(7)} {px(12)};
}}
QPushButton:hover, QToolButton:hover {{ background: {accent_hover}; border-color: {t.accent}; }}
QPushButton:pressed, QToolButton:pressed {{
    background: {accent_soft}; border-color: {t.accent};
}}
QPushButton:disabled, QToolButton:disabled {{ color: {t.muted}; background: {t.surface}; }}
QPushButton#Primary {{ background: {t.accent}; color: {t.background}; font-weight: 800; border-color: {t.accent}; }}
QPushButton#Primary:hover {{ background: {t.accent_alt}; border-color: {t.accent_alt}; }}
QPushButton#Success {{ background: {success_soft}; color: {t.success}; border-color: {t.success}; }}
QPushButton#Warning {{ background: {warning_soft}; color: {t.warning}; border-color: {t.warning}; }}
QPushButton#Danger {{ background: {danger_soft}; color: {t.danger}; border-color: {t.danger}; }}
QToolButton#NavButton {{
    border: 0; border-radius: {px(t.radius)}; padding: {px(9)} {px(10)};
    text-align: left; background: transparent;
}}
QToolButton#NavButton:hover {{ background: {accent_soft}; }}
QToolButton#NavButton:checked {{
    color: {t.accent}; background: {accent_soft}; border-left: {px(3)} solid {t.accent};
    font-weight: 700;
}}

QLineEdit, QPlainTextEdit, QTextEdit, QTextBrowser, QSpinBox, QDoubleSpinBox,
QComboBox, QListWidget, QTreeWidget, QTableWidget {{
    background: {t.code_background}; color: {t.text}; border: 1px solid {control_border};
    border-radius: {px(t.radius)}; padding: {px(6)};
    selection-background-color: {accent_hover};
}}
QPlainTextEdit#CodeEditor {{ font-family: {t.code_font}; font-size: {px(12)}; }}
QLineEdit[valid="false"] {{ border-color: {t.danger}; }}
QLineEdit:disabled, QPlainTextEdit:disabled, QTextEdit:disabled, QTextBrowser:disabled,
QSpinBox:disabled, QDoubleSpinBox:disabled, QComboBox:disabled,
QListWidget:disabled, QTreeWidget:disabled, QTableWidget:disabled {{
    color: {t.muted}; background: {t.surface}; border-color: {t.border};
}}
QComboBox QAbstractItemView {{ background: {t.surface}; selection-background-color: {accent_hover}; }}

QTabWidget::pane {{
    border: 1px solid {t.border}; background: {_mix(t.surface, t.background, 0.78)};
    top: -1px; border-radius: {px(t.radius)};
}}
QTabBar::tab {{
    background: {t.background}; color: {t.muted}; border: 1px solid {t.border};
    padding: {px(8)} {px(14)}; margin-right: {px(2)};
}}
QTabBar::tab:hover {{ color: {t.text}; background: {accent_soft}; }}
QTabBar::tab:selected {{ color: {t.accent}; background: {t.surface}; border-bottom-color: {t.surface}; font-weight: 700; }}

QHeaderView::section {{
    background: {t.raised}; color: {t.muted}; border: 0;
    border-bottom: 1px solid {t.border}; padding: {px(7)}; font-weight: 700;
}}
QTableWidget {{ gridline-color: {t.border}; alternate-background-color: {_mix(t.text, t.surface, 0.035)}; }}
QTableWidget::item {{ padding: {px(5)}; }}
QTableWidget::item:hover {{ background: {accent_soft}; }}
QTableWidget::item:selected {{ background: {accent_hover}; }}

QProgressBar {{
    background: {t.code_background}; border: 1px solid {t.border};
    border-radius: {px(4)}; text-align: center;
}}
QProgressBar::chunk {{ background: {t.accent}; border-radius: {px(3)}; }}
QSplitter::handle {{ background: {control_border}; }}
QSplitter::handle:horizontal {{ width: {px(5)}; }}
QSplitter::handle:vertical {{ height: {px(5)}; }}
QScrollBar:vertical {{ background: {t.surface}; width: {px(10)}; }}
QScrollBar::handle:vertical {{ background: {control_border}; border-radius: {px(5)}; min-height: {px(24)}; }}
QScrollBar:horizontal {{ background: {t.surface}; height: {px(9)}; }}
QScrollBar::handle:horizontal {{ background: {control_border}; border-radius: {px(4)}; min-width: {px(24)}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QCheckBox {{ spacing: {px(7)}; }}
QCheckBox::indicator {{
    width: {px(15)}; height: {px(15)}; border: 1px solid {control_border};
    border-radius: {px(4)}; background: {t.code_background};
}}
QCheckBox::indicator:hover {{ border-color: {t.accent}; }}
QCheckBox::indicator:checked {{
    background: {t.accent}; border-color: {t.accent};
}}
QCheckBox:disabled {{ color: {t.muted}; }}
QCheckBox::indicator:disabled {{ background: {t.surface}; border-color: {t.border}; }}

QPushButton:focus, QToolButton:focus {{
    border: {px(2)} solid {t.accent};
}}
QToolButton#NavButton:focus {{
    border: {px(2)} solid {t.accent}; border-left-width: {px(3)};
}}
QLineEdit:focus, QPlainTextEdit:focus, QTextEdit:focus, QTextBrowser:focus,
QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QListWidget:focus,
QTreeWidget:focus, QTableWidget:focus {{
    border: {px(2)} solid {t.accent};
}}
QCheckBox:focus {{ color: {t.accent}; }}
QToolTip {{ background: {t.raised}; color: {t.text}; border: 1px solid {t.accent}; padding: {px(5)}; }}
"""


# Friendly aliases for callers migrating from older GUI prototypes.
build_qss = build_stylesheet
MONO_FONT_FAMILY = _MONO_FONT
