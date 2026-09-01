"""Focused GUI coverage for manual and live project diagnostics."""

from __future__ import annotations

import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from daedalus.developer.progress import ProjectProgressInspector
from daedalus.gui.main_window import MainWindow
from daedalus.workspace.manager import WorkspaceManager


@pytest.fixture(scope="module")
def app() -> QApplication:
    return QApplication.instance() or QApplication(["daedalus-diagnostics-ui-test"])


@pytest.fixture()
def manager(tmp_path: Path) -> WorkspaceManager:
    source = tmp_path / "source"
    source.mkdir()
    instance = WorkspaceManager(source, tmp_path / "workspace", tmp_path / "backup")
    instance.bootstrap()
    return instance


def _fast_progress(_inspector, project: Path):
    return SimpleNamespace(
        project_name=Path(project).name,
        project_root=Path(project),
        completed=0,
        total=10,
        percent=0,
        next_gate="discovery",
        next_gate_title="Discovery",
        findings=(),
    )


def _wait(app: QApplication, predicate, timeout: float = 8.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        time.sleep(0.01)
        app.processEvents()
    assert predicate()


def _dispose(window: MainWindow, app: QApplication) -> None:
    window._live_scan_timer.stop()
    _wait(app, lambda: not window._running_background_tasks())
    if window._diagnostics_dialog is not None:
        window._diagnostics_dialog.close()
    window.close()
    window.deleteLater()
    app.processEvents()


def test_opening_legacy_project_creates_logs_and_enables_scan_controls(
    app: QApplication,
    manager: WorkspaceManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ProjectProgressInspector, "inspect", _fast_progress)
    project = manager.projects_dir / "legacy-project"
    project.mkdir()
    (project / "main.py").write_text("print('ready')\n", encoding="utf-8")
    (project / "README.md").write_text("# Legacy\n", encoding="utf-8")
    (project / "project.json").write_text(
        '{"schema": 1, "name": "legacy-project"}', encoding="utf-8"
    )

    window = MainWindow(manager)
    window.live_scan_button.setChecked(False)
    window._live_scan_timer.stop()
    window.show()
    app.processEvents()
    try:
        assert window.active_project_path == project.resolve()
        assert (project / "logs" / "README.txt").is_file()
        assert window.diagnostics_button.isEnabled()
        assert window.live_scan_button.isEnabled()
        assert window.diagnostics_button.accessibleName() == (
            "Scan active project and logs for problems"
        )
        assert "metadata" in window.live_scan_button.accessibleDescription()
    finally:
        _dispose(window, app)


def test_manual_scan_opens_redacted_actionable_report(
    app: QApplication,
    manager: WorkspaceManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ProjectProgressInspector, "inspect", _fast_progress)
    project = manager.create_project("broken-project")
    (project / "broken.py").write_text("def broken(:\n", encoding="utf-8")
    private_value = "sk-private-diagnostic-fixture"
    (project / "logs" / "training.log").write_text(
        f"Traceback {private_value}\n", encoding="utf-8"
    )

    window = MainWindow(manager)
    window.live_scan_button.setChecked(False)
    window._live_scan_timer.stop()
    window.show()
    app.processEvents()
    try:
        window.scan_project_and_logs(show_report=True)
        _wait(app, lambda: not window._diagnostics_running)
        dialog = window._diagnostics_dialog
        assert dialog is not None
        assert dialog.isVisible()
        report_text = dialog.report_text.toPlainText()
        assert "Python could not parse" in report_text
        assert "A log line reports traceback" in report_text
        assert "project/broken-project/broken.py:1" in report_text
        assert private_value not in report_text
        assert window._diagnostic_issue_count >= 2
        assert "(" in window.diagnostics_button.text()
    finally:
        _dispose(window, app)


def test_live_watch_scans_after_log_metadata_changes_without_opening_dialog(
    app: QApplication,
    manager: WorkspaceManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ProjectProgressInspector, "inspect", _fast_progress)
    project = manager.create_project("live-project")
    window = MainWindow(manager)
    window.live_scan_button.setChecked(False)
    window._live_scan_timer.stop()
    window.show()
    app.processEvents()
    try:
        time.sleep(0.05)
        app.processEvents()
        window.live_scan_button.setChecked(True)
        _wait(
            app,
            lambda: window._live_change_token is not None and not window._live_token_running,
        )
        baseline_token = window._live_change_token
        generation = window._live_token_generation
        window._live_token_finished(generation, project, str(baseline_token))
        window._live_token_finished(generation, project, str(baseline_token))
        assert window._live_scan_timer.interval() > window.LIVE_SCAN_BASE_INTERVAL_MS
        (project / "logs" / "live.log").write_text("fatal training failure\n", encoding="utf-8")
        window._poll_live_diagnostics()
        _wait(
            app,
            lambda: window._diagnostics_running or window._last_diagnostics_report is not None,
        )
        _wait(
            app,
            lambda: not window._diagnostics_running and window._last_diagnostics_report is not None,
        )
        assert window._last_diagnostics_report is not None
        assert window._diagnostics_dialog is None
        assert window._diagnostic_issue_count >= 1
        assert "Live Project Change" in window.status_strip.message.text()
        assert window._live_scan_timer.interval() == window.LIVE_SCAN_BASE_INTERVAL_MS
    finally:
        _dispose(window, app)
