"""Focused, deterministic smoke tests for the persistent project progress header."""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QInputDialog

from daedalus.developer.progress import ProjectProgressInspector
from daedalus.gui.main_window import MainWindow
from daedalus.workspace.manager import WorkspaceManager


@pytest.fixture(scope="module")
def app() -> QApplication:
    application = QApplication.instance() or QApplication(["daedalus-project-progress-test"])
    application.setApplicationName("Daedalus Project Progress Tests")
    return application


@pytest.fixture()
def manager(tmp_path: Path) -> WorkspaceManager:
    source = tmp_path / "source"
    workspace = tmp_path / "external-workspace"
    backup = tmp_path / "backup"
    source.mkdir()
    instance = WorkspaceManager(source, workspace, backup)
    instance.bootstrap()
    return instance


def _wait_until(app: QApplication, predicate, timeout: float = 4.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        time.sleep(0.01)
        app.processEvents()
    assert predicate()


def _snapshot(
    project: Path,
    *,
    percent: int,
    completed: int,
    next_gate: str | None,
):
    return SimpleNamespace(
        project_name=project.name,
        project_root=project,
        completed=completed,
        total=10,
        percent=percent,
        next_gate=next_gate,
        gates=(),
        findings=(),
    )


def _dispose(window: MainWindow, app: QApplication) -> None:
    _wait_until(app, lambda: not window._running_background_tasks())
    window.close()
    window.deleteLater()
    app.processEvents()


def test_project_progress_header_has_stable_empty_and_compact_states(
    app: QApplication,
    manager: WorkspaceManager,
) -> None:
    window = MainWindow(manager)
    window.show()
    app.processEvents()
    try:
        assert window.active_project.count() == 1
        assert window.active_project.currentData() is None
        assert window.active_project_path is None
        assert window.project_progress.minimum() == 0
        assert window.project_progress.maximum() == 100
        assert window.project_progress.value() == 0
        assert window.project_progress.format() == "No active project"
        assert window.project_progress.accessibleName() == "Active AI project completion"
        assert "No active project" in window.project_progress.accessibleDescription()
        assert window.project_progress_stage.text() == "Create or open a project to begin"

        window.resize(900, 700)
        window._apply_responsive_layout()
        app.processEvents()
        assert window.project_progress.isVisible()
        assert window.project_progress_stage.isHidden()
    finally:
        _dispose(window, app)


def test_sole_project_is_selected_and_evidence_progress_is_accessible(
    app: QApplication,
    manager: WorkspaceManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = manager.create_project("progress-fixture")

    def inspect(_inspector, selected: Path):
        assert Path(selected).resolve() == project.resolve()
        return _snapshot(project, percent=30, completed=3, next_gate="baseline")

    monkeypatch.setattr(ProjectProgressInspector, "inspect", inspect)
    window = MainWindow(manager)
    window.show()
    app.processEvents()
    try:
        _wait_until(app, lambda: window.project_progress.value() == 30)
        assert Path(str(window.active_project.currentData())).resolve() == project.resolve()
        assert window.active_project_path == project.resolve()
        assert window.project_progress_label.text() == "PROJECT PROGRESS"
        assert window.project_progress.format() == "30% complete"
        assert window.project_progress_stage.text() == "Next gate: Baseline"
        description = window.project_progress.accessibleDescription()
        assert "progress-fixture" in description
        assert "3 of 10 evidence gates" in description

        progress_before_navigation = window.project_progress.value()
        assert window.navigate("guard")
        assert window.project_progress.value() == progress_before_navigation
    finally:
        _dispose(window, app)


def test_creating_a_project_activates_its_progress_immediately(
    app: QApplication,
    manager: WorkspaceManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def inspect(_inspector, selected: Path):
        project = Path(selected).resolve()
        return _snapshot(project, percent=0, completed=0, next_gate="discovery")

    monkeypatch.setattr(ProjectProgressInspector, "inspect", inspect)
    monkeypatch.setattr(QInputDialog, "getText", lambda *_args, **_kwargs: ("new-ai", True))
    window = MainWindow(manager)
    window.show()
    app.processEvents()
    try:
        mission = window.pages["mission"]
        mission._create_project()
        expected = (manager.projects_dir / "new-ai").resolve()
        _wait_until(app, lambda: window.project_progress.format() == "0% complete")
        assert window.active_project_path == expected
        assert Path(str(window.active_project.currentData())).resolve() == expected
        assert window.project_progress_stage.text() == "Next gate: Discovery"
    finally:
        _dispose(window, app)


def test_project_progress_discards_late_results_and_handoff_selects_project(
    app: QApplication,
    manager: WorkspaceManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    alpha = manager.create_project("alpha")
    beta = manager.create_project("beta")
    alpha_started = threading.Event()
    release_alpha = threading.Event()

    def inspect(_inspector, selected: Path):
        project = Path(selected).resolve()
        if project == alpha.resolve():
            alpha_started.set()
            release_alpha.wait(3.0)
            return _snapshot(alpha, percent=10, completed=1, next_gate="recovery")
        return _snapshot(beta, percent=70, completed=7, next_gate="deployment")

    monkeypatch.setattr(ProjectProgressInspector, "inspect", inspect)
    window = MainWindow(manager)
    window.show()
    app.processEvents()
    try:
        assert window.active_project_path is None
        assert window.set_active_project(alpha)
        _wait_until(app, alpha_started.is_set)

        window._handle_developer_tool("training", {"project_root": str(beta)})
        _wait_until(app, lambda: window.project_progress.value() == 70)
        assert window.active_project_path == beta.resolve()
        assert Path(str(window.active_project.currentData())).resolve() == beta.resolve()
        assert window.project_progress_stage.text() == "Next gate: Deployment"

        release_alpha.set()
        _wait_until(app, lambda: not window._running_background_tasks())
        assert window.project_progress.value() == 70
        assert "beta" in window.project_progress.accessibleDescription()
    finally:
        release_alpha.set()
        _dispose(window, app)
