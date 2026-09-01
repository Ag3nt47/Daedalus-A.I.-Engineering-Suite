"""Offscreen coverage for the guided professional-project setup tab."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from daedalus.gui.pages import DeveloperBotPage
from daedalus.workspace.manager import WorkspaceManager


@pytest.fixture(scope="module")
def app() -> QApplication:
    application = QApplication.instance() or QApplication(["daedalus-standards-ui-test"])
    application.setApplicationName("Daedalus Project Standards UI Tests")
    return application


@pytest.fixture()
def manager(tmp_path: Path) -> WorkspaceManager:
    source = tmp_path / "source"
    source.mkdir()
    instance = WorkspaceManager(source, tmp_path / "private", tmp_path / "backup")
    instance.bootstrap()
    return instance


def _wait_until(app: QApplication, predicate, timeout: float = 6.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        # Yield the Python GIL so the pure-Python bounded source fingerprint can
        # advance in BackgroundTask while Qt callbacks continue on this thread.
        time.sleep(0.01)
        app.processEvents()
    assert predicate()


def _idle(panel: object) -> bool:
    return not getattr(panel, "_background_tasks", set()) and not getattr(
        panel, "_busy", False
    )


def test_developer_bot_exposes_ordered_accessible_setup_workflow(
    app: QApplication,
    manager: WorkspaceManager,
) -> None:
    project = manager.create_project("guided-setup")
    page = DeveloperBotPage(manager)
    page.show()
    app.processEvents()
    try:
        assert [page.tabs.tabText(index) for index in range(page.tabs.count())] == [
            "Tools",
            "Setup",
            "Info",
        ]
        panel = page.setup_panel
        assert panel.project == project.resolve()
        assert panel.audit_button.accessibleName() == (
            "Audit project setup and reproducibility"
        )
        assert "never replaced" in panel.initialize_button.toolTip()
        assert panel.capture_button.accessibleName() == (
            "Capture project environment evidence"
        )

        panel.audit_button.click()
        _wait_until(app, lambda: _idle(panel))
        assert panel.findings.rowCount() >= 1
        assert "guided-setup" in panel.summary.text()
        assert panel.tools.rowCount() > 0
        assert "Project source fingerprint" in panel.evidence.toPlainText()
        assert "no network request" in panel.evidence.toPlainText()
    finally:
        _wait_until(app, lambda: _idle(page.setup_panel))
        page.close()
        page.deleteLater()
        app.processEvents()


def test_setup_actions_preserve_legacy_files_and_capture_evidence(
    app: QApplication,
    manager: WorkspaceManager,
) -> None:
    legacy = manager.projects_dir / "legacy-ui"
    legacy.mkdir()
    (legacy / "main.py").write_text("print('owned by user')\n", encoding="utf-8")
    sentinel = "# user-owned package settings\n"
    (legacy / "pyproject.toml").write_text(sentinel, encoding="utf-8")

    page = DeveloperBotPage(manager)
    page.show()
    app.processEvents()
    panel = page.setup_panel
    try:
        assert page.set_project(legacy)
        panel.initialize_button.click()
        _wait_until(app, lambda: _idle(panel))
        assert (legacy / "project.json").is_file()
        assert (legacy / "cards" / "MODEL_CARD.template.md").is_file()
        assert (legacy / "observability" / "README.md").is_file()
        assert (legacy / "pyproject.toml").read_text(encoding="utf-8") == sentinel

        panel.capture_button.click()
        snapshot = legacy / "ENVIRONMENT_SNAPSHOT.json"
        _wait_until(app, lambda: snapshot.is_file() and _idle(panel))
        assert "Captured reviewable environment evidence" in panel.summary.text()
        assert str(manager.workspace_root) not in snapshot.read_text(encoding="utf-8")
    finally:
        _wait_until(app, lambda: _idle(panel))
        page.close()
        page.deleteLater()
        app.processEvents()
