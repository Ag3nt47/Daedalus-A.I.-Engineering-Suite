from __future__ import annotations

import hashlib
import os
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import QUrl
from PySide6.QtWidgets import QApplication

import daedalus.gui.weight_lab_panel as weight_lab_module
from daedalus.engine.weight_tools import WEIGHT_TOOL_SPECS
from daedalus.gui.pages import CalculatorLabPage
from daedalus.gui.weight_lab_panel import WeightLabPanel
from daedalus.services.weight_sandbox import SANDBOX_RELATIVE_DIRECTORY, sandbox_template
from daedalus.workspace.manager import WorkspaceManager

EXPECTED_TOOL_KEYS = [
    "meta_weight",
    "logic_compiler",
    "recurrent_kernel",
    "constraint_optimizer",
    "matrix_inverter",
    "uncertainty_sampler",
]
EXPECTED_TOOL_TITLES = [
    "Meta-Weight Synthesizer",
    "Direct Logic Compiler",
    "Recurrent Kernel Engine",
    "Constraint Optimizer",
    "Matrix Inverter",
    "Uncertainty Sampler",
]


@pytest.fixture(scope="module")
def app() -> QApplication:
    application = QApplication.instance() or QApplication(["daedalus-weight-lab-tests"])
    application.setApplicationName("Daedalus Weight Lab Tests")
    return application


@pytest.fixture()
def manager(tmp_path: Path) -> WorkspaceManager:
    source = tmp_path / "source"
    source.mkdir()
    instance = WorkspaceManager(source, tmp_path / "workspace", tmp_path / "backup")
    instance.bootstrap()
    return instance


def _project_snapshot(project: Path) -> dict[str, tuple[str, str]]:
    snapshot: dict[str, tuple[str, str]] = {}
    for path in sorted(project.rglob("*")):
        relative = path.relative_to(project).as_posix()
        if path.is_dir():
            snapshot[relative] = ("directory", "")
        else:
            snapshot[relative] = (
                "file",
                hashlib.sha256(path.read_bytes()).hexdigest(),
            )
    return snapshot


def test_panel_and_calculator_expose_the_exact_information_architecture(
    app: QApplication,
    manager: WorkspaceManager,
) -> None:
    calculator = CalculatorLabPage(manager)
    calculator.show()
    app.processEvents()
    try:
        panel = calculator.weight_lab
        assert [calculator.tabs.tabText(index) for index in range(calculator.tabs.count())] == [
            "Tools",
            "Weight Lab",
            "Advanced",
            "Info",
        ]
        assert list(panel.launch_buttons) == EXPECTED_TOOL_KEYS
        assert [spec.key for spec in WEIGHT_TOOL_SPECS] == EXPECTED_TOOL_KEYS
        assert [spec.title for spec in WEIGHT_TOOL_SPECS] == EXPECTED_TOOL_TITLES
        assert [
            panel.inner_tabs.tabText(index) for index in range(panel.inner_tabs.count())
        ] == ["Guided", "Sandbox", "More Info"]
        assert panel.guided_stack.count() == 6

        for key, title in zip(EXPECTED_TOOL_KEYS, EXPECTED_TOOL_TITLES):
            launcher = panel.launch_buttons[key]
            assert launcher.accessibleName() == f"Open {title}"
            assert title in launcher.text()
            assert launcher.toolTip().startswith("Use when:")
    finally:
        calculator.close()
        calculator.deleteLater()
        app.processEvents()


def test_calculator_large_what_if_plan_uses_shape_only_memory_math(
    app: QApplication,
    manager: WorkspaceManager,
) -> None:
    calculator = CalculatorLabPage(manager)
    try:
        calculator.layer_sizes.setText("2000, 2000")
        calculator.batch.setValue(16)
        calculator.precision.setCurrentIndex(1)  # FP32
        calculator.optimizer.setCurrentIndex(2)  # Adam
        report = calculator.calculate()

        assert report is not None
        parameters = 2_000 * 2_000 + 2_000
        assert report["parameters"] == parameters
        assert report["parameter_bytes"] == parameters * 4
        assert report["activation_bytes"] == 16 * (2_000 + 2_000) * 4
        assert report["used_engine"] == 0
        assert "bounded arithmetic-only" in calculator.results.toPlainText()
    finally:
        calculator.close()
        calculator.deleteLater()
        app.processEvents()


def test_every_guided_example_completes_and_formats_assurance(
    app: QApplication,
    manager: WorkspaceManager,
) -> None:
    panel = WeightLabPanel(manager)
    try:
        for index, spec in enumerate(WEIGHT_TOOL_SPECS):
            panel._select_tool(spec.key)
            assert panel.guided_stack.currentIndex() == index

            result = panel._guided_callable()()
            rendered = panel._format_result(result)

            assert result.record.tool_key == spec.key
            assert result.record.assurance == spec.assurance
            assert "RESULT\n======" in rendered
            assert f"Algorithm: {result.record.algorithm}" in rendered
            assert f"Assurance: {spec.assurance.replace('_', ' ')}" in rendered
            assert "HELPFUL HINTS" in rendered
    finally:
        panel.close()
        panel.deleteLater()
        app.processEvents()


def test_more_info_builds_exact_links_and_opens_only_after_explicit_action(
    app: QApplication,
    manager: WorkspaceManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opened: list[QUrl] = []

    class FakeDesktopServices:
        @staticmethod
        def openUrl(url: QUrl) -> bool:
            opened.append(QUrl(url))
            return True

    monkeypatch.setattr(weight_lab_module, "QDesktopServices", FakeDesktopServices)
    panel = WeightLabPanel(manager)
    try:
        assert not panel.info_browser.openExternalLinks()
        assert opened == []

        for spec in WEIGHT_TOOL_SPECS:
            open_count = len(opened)
            panel._select_tool(spec.key)
            assert len(opened) == open_count
            youtube = panel.youtube_search_url
            parsed = urlsplit(youtube.toString())
            assert parsed.scheme == "https"
            assert parsed.hostname == "www.youtube.com"
            assert parsed.path == "/results"
            assert parse_qs(parsed.query) == {"search_query": [spec.youtube_query]}
            assert panel.primary_source_url.toString() == spec.primary_source_url
            assert panel.primary_source_url.scheme() == "https"
            assert spec.primary_source_title in panel.info_browser.toPlainText()
            assert "last reviewed 2026-08-30" in panel.info_browser.toPlainText()

            assert panel.open_youtube_search()
            assert opened[-1].toString() == youtube.toString()
            assert panel.open_primary_source()
            assert opened[-1].toString() == spec.primary_source_url

        explicit_open_count = len(opened)
        expected_youtube = panel.youtube_search_url
        unsafe_youtube = (
            QUrl("http://www.youtube.com/results?search_query=weights"),
            QUrl("https://www.youtube.com.evil.invalid/results?search_query=weights"),
            QUrl("https://www.youtube.com/watch?v=unrelated"),
            QUrl("javascript:alert(1)"),
        )
        for candidate in unsafe_youtube:
            assert not panel._open_exact_url(candidate, expected_youtube, youtube=True)

        expected_primary = panel.primary_source_url
        assert not panel._open_exact_url(
            QUrl(expected_primary.toString() + "?unexpected=1"),
            expected_primary,
            youtube=False,
        )
        assert len(opened) == explicit_open_count
    finally:
        panel.close()
        panel.deleteLater()
        app.processEvents()


def test_tool_selection_only_previews_sandbox_and_does_not_write_project(
    app: QApplication,
    manager: WorkspaceManager,
) -> None:
    project = manager.create_project("selection-is-read-only")
    before = _project_snapshot(project)
    panel = WeightLabPanel(manager)
    try:
        assert panel.set_project(project)
        panel.set_reduced_motion(True)
        panel.show()
        panel.open_tool(EXPECTED_TOOL_KEYS[0])
        app.processEvents()
        for spec in WEIGHT_TOOL_SPECS:
            panel._select_tool(spec.key)
            assert "Starter preview; not written yet" in panel.sandbox_status.text()
            assert panel.sandbox_editor.toPlainText() == sandbox_template(spec.key)
            assert panel.create_draft_button.isEnabled()

        assert _project_snapshot(project) == before
        assert not (project / "experiments").exists()
    finally:
        panel.close()
        panel.deleteLater()
        app.processEvents()


def test_explicit_draft_creation_enables_verified_workshop_handoff(
    app: QApplication,
    manager: WorkspaceManager,
) -> None:
    project = manager.create_project("explicit-draft")
    panel = WeightLabPanel(manager)
    handed_off: list[Path] = []
    panel.open_in_workshop_requested.connect(lambda path: handed_off.append(Path(path)))
    try:
        panel.set_project(project)
        panel.set_reduced_motion(True)
        panel.show()
        panel.open_tool("logic_compiler")
        app.processEvents()
        expected = project / SANDBOX_RELATIVE_DIRECTORY / "logic_compiler.py"

        assert not expected.exists()
        assert not panel.open_sandbox_in_workshop()
        assert handed_off == []
        assert panel.create_sandbox_draft()
        assert expected.is_file()
        assert expected.read_text(encoding="utf-8") == sandbox_template("logic_compiler")
        assert not panel.sandbox_editor.editor.document().isModified()
        assert panel.open_workshop_button.isEnabled()

        assert panel.open_sandbox_in_workshop()
        assert handed_off == [expected]
    finally:
        panel.close()
        panel.deleteLater()
        app.processEvents()
