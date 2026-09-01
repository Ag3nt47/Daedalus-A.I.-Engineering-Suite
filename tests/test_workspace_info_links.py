"""Focused coverage for stage-specific external learning links."""

from __future__ import annotations

import os
from urllib.parse import parse_qs, urlsplit

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import Qt, QUrl
from PySide6.QtWidgets import QApplication

import daedalus.gui.pages as pages_module
from daedalus.gui.pages import (
    PROFESSIONAL_STAGE_GUIDANCE,
    YOUTUBE_SEARCH_QUERIES,
    WorkspacePage,
)

EXPECTED_QUERIES = {
    "mission": "build an AI model step by step for beginners",
    "developer": "define a machine learning problem and success metrics tutorial",
    "learn": "neural network fundamentals tensors gradients beginner tutorial",
    "architecture": "design neural network architecture tensor shapes tutorial",
    "calculator": "calculate neural network parameters memory and batch size tutorial",
    "training": "prepare tabular data train validation test split neural network tutorial",
    "workshop": "build a neural network from scratch Python NumPy tutorial",
    "evaluate": (
        "machine learning model evaluation confusion matrix precision recall F1 tutorial"
    ),
    "backup": "machine learning experiment checkpoints backup and reproducibility tutorial",
    "guard": "machine learning model release privacy security deployment checklist tutorial",
    "settings": "machine learning development environment project setup tutorial",
}


@pytest.fixture(scope="module")
def app() -> QApplication:
    application = QApplication.instance() or QApplication(["daedalus-info-link-tests"])
    application.setApplicationName("Daedalus Info Link Tests")
    return application


@pytest.mark.parametrize(("icon", "expected_query"), EXPECTED_QUERIES.items())
def test_every_workspace_stage_gets_an_accessible_encoded_youtube_search(
    app: QApplication,
    icon: str,
    expected_query: str,
) -> None:
    title = icon.replace("_", " ").title()
    page = WorkspacePage(None, title, "Test workspace", icon, ())
    try:
        assert YOUTUBE_SEARCH_QUERIES == EXPECTED_QUERIES
        assert page.tabs.tabText(page.tabs.count() - 1) == "Info"
        assert set(PROFESSIONAL_STAGE_GUIDANCE) == set(EXPECTED_QUERIES)
        assert page.info_widget.layout().indexOf(page.professional_guidance_panel) >= 0
        assert "Professional" in page.professional_guidance_panel.toggle.text()
        assert page.professional_guidance_panel.content.text().strip()
        assert page.info_widget.layout().indexOf(page.youtube_help_panel) >= 0
        assert page.youtube_search_query == expected_query

        parsed = urlsplit(page.youtube_search_url.toString())
        assert parsed.scheme == "https"
        assert parsed.hostname == "www.youtube.com"
        assert parsed.path == "/results"
        assert parse_qs(parsed.query) == {"search_query": [expected_query]}

        link = page.youtube_help_link
        assert link.textFormat() == Qt.TextFormat.RichText
        assert not link.openExternalLinks()
        assert link.textInteractionFlags() & Qt.TextInteractionFlag.LinksAccessibleByMouse
        assert link.textInteractionFlags() & Qt.TextInteractionFlag.LinksAccessibleByKeyboard
        assert link.focusPolicy() == Qt.FocusPolicy.StrongFocus
        assert title in link.accessibleName()
        assert "default browser" in link.accessibleDescription()
    finally:
        page.close()
        page.deleteLater()
        app.processEvents()


def test_youtube_link_requires_explicit_activation_and_rejects_spoofed_urls(
    app: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opened: list[QUrl] = []

    class FakeDesktopServices:
        @staticmethod
        def openUrl(url: QUrl) -> bool:
            opened.append(url)
            return True

    monkeypatch.setattr(pages_module, "QDesktopServices", FakeDesktopServices)
    page = WorkspacePage(None, "Data & Training Lab", "Test workspace", "training", ())
    try:
        assert opened == []
        expected_href = page.youtube_search_url.toString()
        page.youtube_help_link.linkActivated.emit(expected_href)
        assert [url.toString() for url in opened] == [expected_href]

        unsafe_hrefs = (
            "http://www.youtube.com/results?search_query=training",
            "https://www.youtube.com.evil.invalid/results?search_query=training",
            "https://www.youtube.com/watch?v=unrelated",
            "javascript:alert(1)",
        )
        for href in unsafe_hrefs:
            assert not page._open_youtube_search(href)
        assert [url.toString() for url in opened] == [expected_href]
    finally:
        page.close()
        page.deleteLater()
        app.processEvents()


def test_unmapped_future_workspace_still_gets_a_specific_search(app: QApplication) -> None:
    page = WorkspacePage(None, "Future Stage", "Test workspace", "future", ())
    try:
        parsed = urlsplit(page.youtube_search_url.toString())
        assert parse_qs(parsed.query) == {
            "search_query": ["Future Stage artificial intelligence tutorial"]
        }
    finally:
        page.deleteLater()
        app.processEvents()
