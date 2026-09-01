"""Offscreen smoke coverage for the native Daedalus desktop shell."""

from __future__ import annotations

import os
import sqlite3
import threading
import time
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import Qt
from PySide6.QtGui import QFontDatabase, QFontInfo, QFontMetrics, QPalette
from PySide6.QtWidgets import QApplication, QLabel

from daedalus.developer import (
    STAGE_ORDER,
    DeveloperAdvisor,
    DeveloperSessionStore,
    ExperienceMode,
    ProjectBrief,
    TaskKind,
)
from daedalus.gui.editor import CodeEditorPanel
from daedalus.gui.fonts import ensure_runtime_fonts
from daedalus.gui.main_window import PAGE_SPECS, MainWindow
from daedalus.gui.pages import DeveloperBotPage, LearningAtlasPage, TrainingLabPage
from daedalus.gui.theme import THEMES, build_stylesheet, reduced_motion, tokens_for
from daedalus.gui.widgets import PathField, run_in_background
from daedalus.resources import load_json
from daedalus.workspace.manager import WorkspaceManager


@pytest.fixture(scope="module")
def app() -> QApplication:
    application = QApplication.instance() or QApplication(["daedalus-gui-test"])
    application.setApplicationName("Daedalus GUI Tests")
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


def test_main_window_contains_accessible_page_suite(
    app: QApplication,
    manager: WorkspaceManager,
) -> None:
    window = MainWindow(manager)
    window.show()
    app.processEvents()
    try:
        expected = [title for _key, title, _icon, _description in PAGE_SPECS]
        assert window.page_titles == expected
        assert list(window.pages) == [key for key, _title, _icon, _description in PAGE_SPECS]
        assert len(window._nav_buttons) == len(expected)

        for key, title, _icon, _description in PAGE_SPECS:
            page = window.pages[key]
            button = window._nav_buttons[key]
            assert page.accessibleName() == f"{title} workspace"
            assert button.accessibleName() == f"Navigate to {title}"
            assert button.text().replace("&&", "&") == title
            assert not button.icon().isNull()
            assert page.tabs.count() >= 2
            assert page.tabs.tabText(0) == "Tools"
            assert page.tabs.tabText(page.tabs.count() - 1) == "Info"

        assert [title for _key, title, _icon, _description in PAGE_SPECS[1:-1]] == [
            "1 · Define",
            "2 · Learn",
            "3 · Design",
            "4 · Plan",
            "5 · Data & Train",
            "6 · Build",
            "7 · Evaluate",
            "8 · Protect",
            "9 · Release",
        ]

        window.navigate("guard")
        assert window.current_page_key == "guard"
        assert window._nav_buttons["guard"].isChecked()
        assert "External workspace" in window.workspace_path.accessibleName()
        assert window.workspace_path.line_edit.isReadOnly()
        assert not window.workspace_path.browse_button.isVisible()
        settings = window.pages["settings"]
        assert settings.workspace_path.line_edit.isReadOnly()
        assert settings.backup_path.line_edit.isReadOnly()

        guard = window.pages["guard"]
        guard.source_path.set_path(manager.source_root / "missing-source")
        guard.commit_message.setText("test fail-closed boundary")
        guard.start_safe_push()
        assert "has not started" in guard.output.toPlainText()
    finally:
        window._live_scan_timer.stop()
        deadline = time.monotonic() + 5.0
        while window._running_background_tasks() and time.monotonic() < deadline:
            time.sleep(0.01)
            app.processEvents()
        assert not window._running_background_tasks()
        window.close()
        window.deleteLater()
        app.processEvents()


def test_navigation_surfaces_refresh_failure(
    app: QApplication,
    manager: WorkspaceManager,
) -> None:
    window = MainWindow(manager)
    window.show()
    app.processEvents()
    try:
        page = window.pages["mission"]

        def broken_refresh() -> None:
            raise RuntimeError("fixture refresh damage")

        page.refresh = broken_refresh  # type: ignore[attr-defined]
        assert window.navigate("mission")
        assert "refresh failed safely" in window.status_strip.message.text()
        assert "fixture refresh damage" in window.status_strip.message.text()
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()


def test_main_window_materializes_workspaces_on_demand(
    app: QApplication,
    manager: WorkspaceManager,
) -> None:
    window = MainWindow(manager)
    try:
        assert window.pages.loaded_count == 1
        assert window.pages.loaded("mission") is not None
        assert window.pages.loaded("architecture") is None

        assert window.navigate("architecture")
        assert window.current_page_key == "architecture"
        assert window.pages.loaded_count == 2
        architecture = window.pages["architecture"]
        assert window.pages.loaded("architecture") is architecture
        assert architecture._model_3d_viewer is None

        assert window.navigate("calculator")
        calculator = window.pages["calculator"]
        assert calculator._weight_lab is None
        assert calculator._advanced_panel is None

        assert window.navigate("developer")
        developer = window.pages["developer"]
        assert developer._setup_panel is None
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()


def test_startup_reuses_inventory_and_unchanged_appearance_skips_restyle(
    app: QApplication,
    manager: WorkspaceManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    original_list_projects = WorkspaceManager.list_projects

    def counted_list_projects(instance: WorkspaceManager):
        nonlocal calls
        calls += 1
        return original_list_projects(instance)

    monkeypatch.setattr(WorkspaceManager, "list_projects", counted_list_projects)
    window = MainWindow(manager)
    window.show()
    app.processEvents()
    try:
        assert calls == 1
        styles: list[str] = []
        monkeypatch.setattr(window, "setStyleSheet", styles.append)
        window.apply_appearance(
            window._theme_name,
            window._fixed_scale,
            window._auto_scale,
            window._reduced_motion,
        )
        assert styles == []
        assert "already up to date" in window.status_strip.message.text()

        window.resize(1600, 1000)
        app.processEvents()
        assert styles == []

        window.apply_appearance(
            "cyber" if window._theme_name != "cyber" else "slate",
            window._fixed_scale,
            window._auto_scale,
            window._reduced_motion,
        )
        assert len(styles) == 1
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()


def test_minimum_window_navigation_keeps_full_hit_targets_separate(
    app: QApplication,
    manager: WorkspaceManager,
) -> None:
    window = MainWindow(manager)
    window.resize(780, 560)
    window.show()
    app.processEvents()
    try:
        buttons = list(window._nav_buttons.values())
        for upper, lower in zip(buttons, buttons[1:]):
            assert upper.minimumHeight() >= 42
            assert upper.geometry().bottom() < lower.geometry().top()
        assert window.nav_scroll.verticalScrollBar().maximum() > 0

        assert window.navigate("developer")
        developer = window.pages["developer"]
        assert developer.panel._compact_tabs is not None
        assert developer.panel._compact_tabs.count() == 3

        assert window.navigate("calculator")
        calculator = window.pages["calculator"]
        assert calculator._compact_tool_tabs is not None
        assert calculator._compact_tool_tabs.count() == 3

        assert window.navigate("vault")
        vault = window.pages["vault"]
        positions = [
            vault.metric_layout.getItemPosition(index)[:2]
            for index in range(vault.metric_layout.count())
        ]
        assert positions == [(0, 0), (0, 1), (1, 0), (1, 1)]

        window.resize(1440, 900)
        app.processEvents()
        assert developer.panel._compact_tabs is None
        assert developer.panel.splitter.count() == 3
        assert calculator._compact_tool_tabs is None
        assert calculator.tool_splitter.count() == 3
        positions = [
            vault.metric_layout.getItemPosition(index)[:2]
            for index in range(vault.metric_layout.count())
        ]
        assert positions == [(0, 0), (0, 1), (0, 2), (0, 3)]
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()


def test_window_defers_close_until_background_work_finishes(
    app: QApplication,
    manager: WorkspaceManager,
) -> None:
    window = MainWindow(manager)
    window.show()
    release = threading.Event()
    run_in_background(window, lambda: release.wait(2.0), lambda _result: None)
    app.processEvents()
    try:
        window.close()
        app.processEvents()
        assert window.isVisible()
        assert "active operation" in window.status_strip.message.text()

        release.set()
        deadline = time.monotonic() + 3.0
        while window.isVisible() and time.monotonic() < deadline:
            time.sleep(0.01)
            app.processEvents()
        assert not window.isVisible()
    finally:
        release.set()
        window.close()
        window.deleteLater()
        app.processEvents()


def test_path_field_exposes_external_git_excluded_state(
    app: QApplication,
    manager: WorkspaceManager,
    tmp_path: Path,
) -> None:
    field = PathField(
        "Projects",
        manager.projects_dir,
        manager=manager,
        git_excluded=True,
    )
    field.show()
    field.setStyleSheet(build_stylesheet("slate"))
    app.processEvents()
    try:
        assert field.is_valid()
        assert field.path.resolve() == manager.projects_dir.resolve()
        assert "EXTERNAL" in field.state_label.text()
        assert "GIT EXCLUDED" in field.state_label.text()
        assert "READY" in field.state_label.text()
        assert field.line_edit.accessibleName() == "Projects path"
        assert field.browse_button.accessibleName() == "Browse for Projects"
        assert field.copy_button.accessibleName() == "Copy Projects path"
        assert field.reveal_button.accessibleName() == "Reveal Projects in file manager"
        assert (
            field.state_label.palette().color(QPalette.ColorRole.WindowText).name()
            == tokens_for("slate").success
        )

        field.set_path(tmp_path / "does-not-exist")
        app.processEvents()
        assert not field.is_valid()
        assert "PATH NOT FOUND" in field.state_label.text()
        assert field.property("valid") is False
        assert (
            field.state_label.palette().color(QPalette.ColorRole.WindowText).name()
            == tokens_for("slate").warning
        )

        pending = PathField(
            "Future backup",
            tmp_path / "not-created-yet",
            manager=manager,
            git_excluded=True,
            allow_missing=True,
        )
        assert pending.is_valid()
        assert "READY TO CREATE" in pending.state_label.text()
        pending.deleteLater()
    finally:
        field.close()
        field.deleteLater()


def test_theme_tokens_and_reduced_motion_override(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("slate", "cyber", "crt"):
        tokens = tokens_for(name)
        stylesheet = build_stylesheet(name, scale=1.1)
        assert name in THEMES
        assert tokens.accent in stylesheet
        assert tokens.background in stylesheet
        assert "QToolButton#NavButton" in stylesheet
        assert "QLabel, QCheckBox, QRadioButton { background: transparent; }" in stylesheet

    monkeypatch.setenv("DAEDALUS_REDUCE_MOTION", "1")
    assert reduced_motion()
    monkeypatch.setenv("DAEDALUS_REDUCE_MOTION", "0")
    assert not reduced_motion()


def test_runtime_font_recovery_resolves_basic_glyphs(app: QApplication) -> None:
    runtime = ensure_runtime_fonts()
    families = QFontDatabase.families()
    assert runtime.ui_family in families
    assert runtime.code_family in families

    label = QLabel("Daedalus ABC xyz 0123")
    label.setStyleSheet(build_stylesheet("slate"))
    label.show()
    app.processEvents()
    try:
        assert QFontInfo(label.font()).family() == runtime.ui_family
        metrics = QFontMetrics(label.font())
        assert all(metrics.inFontUcs4(ord(character)) for character in "Daedalus0123")
    finally:
        label.close()
        label.deleteLater()


def test_code_editor_has_find_and_line_number_support(app: QApplication) -> None:
    panel = CodeEditorPanel()
    panel.setPlainText("alpha = 1\nbeta = alpha + 1\nprint(beta)\n")
    panel.show()
    app.processEvents()
    try:
        assert panel.editor.accessibleName() == "Python code editor"
        assert panel.editor.blockCount() == 4
        assert panel.editor.line_number_area_width() > 0
        panel.show_find()
        panel.find_input.setText("alpha")
        assert panel.find_next()
        assert panel.editor.textCursor().selectedText() == "alpha"
    finally:
        panel.close()
        panel.deleteLater()
        app.processEvents()


def test_code_editor_pauses_highlighting_for_large_documents(app: QApplication) -> None:
    panel = CodeEditorPanel()
    try:
        panel.setPlainText("value = 1\n" * 6_100)
        assert not panel.syntax_highlighting_enabled
        assert panel.editor.highlighter.document() is None
        assert "paused" in panel.editor.accessibleDescription().casefold()

        panel.setPlainText("value = 1\n")
        assert panel.syntax_highlighting_enabled
        assert panel.editor.highlighter.document() is panel.editor.document()
    finally:
        panel.deleteLater()
        app.processEvents()


def test_learning_atlas_uses_all_packaged_resources(
    app: QApplication,
    manager: WorkspaceManager,
) -> None:
    atlas = LearningAtlasPage(manager)
    atlas.show()
    app.processEvents()
    try:
        learning = load_json("learning_paths.json")
        glossary = load_json("glossary.json")
        errors = load_json("error_cards.json")
        recipes = load_json("project_recipes.json")
        sources = load_json("sources.json")

        assert atlas.track_list.count() == len(learning["tracks"])
        assert atlas.module_list.count() == len(learning["tracks"][0]["modules"])
        assert "Checkpoint gates" in atlas.topic.toPlainText()
        assert [atlas.resource_tabs.tabText(index) for index in range(5)] == [
            "Learning Paths",
            "Glossary",
            "Error Clinic",
            "Project Recipes",
            "Official Sources",
        ]

        assert atlas.glossary_list.count() == len(glossary["entries"])
        atlas.glossary_search.setText("zero-dimensional")
        assert atlas.glossary_list.count() >= 1
        assert "Scalar" in atlas.glossary_detail.toPlainText()

        assert len(atlas.error_cards) == len(errors["cards"])
        atlas.error_search.setText("broadcast")
        assert atlas.error_list.count() >= 1
        assert "Safe fixes" in atlas.error_detail.toPlainText()

        assert len(atlas.recipes) == len(recipes["recipes"])
        assert len(atlas.sources) == len(sources["sources"])
        assert "Offline summary" in atlas.source_detail.toPlainText()
        assert atlas.open_source_button.isEnabled()
    finally:
        atlas.close()
        atlas.deleteLater()
        app.processEvents()


def test_training_csv_import_and_load_helper_without_dialog(
    app: QApplication,
    manager: WorkspaceManager,
    tmp_path: Path,
) -> None:
    numpy = pytest.importorskip("numpy")
    source = tmp_path / "teaching.csv"
    source.write_text(
        "feature_a,feature_b,target\n1.0,2.0,0\n3.0,4.0,1\n",
        encoding="utf-8",
    )
    page = TrainingLabPage(manager)
    page.show()
    app.processEvents()
    try:
        metadata = page.import_csv_path(
            source,
            name="teaching-fixture",
            target_column="target",
        )
        assert metadata.name == "teaching-fixture"
        assert metadata.rows == 2
        assert metadata.feature_columns == ("feature_a", "feature_b")

        features, targets, loaded = page.load_dataset(metadata.name)
        assert loaded.sha256 == metadata.sha256
        assert features.shape == (2, 2)
        numpy.testing.assert_allclose(features, [[1.0, 2.0], [3.0, 4.0]])
        numpy.testing.assert_allclose(targets, [0.0, 1.0])

        page.refresh_datasets(metadata.name)
        assert page.dataset.currentData() == metadata.name
        assert "teaching-fixture" in page.dataset_metadata.toPlainText()
        assessment = page.analyze_dataset(metadata.name, task="auto")
        assert assessment.sample_count == 2
        assert assessment.feature_count == 2
    finally:
        page.close()
        page.deleteLater()
        app.processEvents()


def test_training_call_journals_run_and_checkpoint(
    app: QApplication,
    manager: WorkspaceManager,
) -> None:
    from daedalus.workspace.run_registry import RunRegistry

    page = TrainingLabPage(manager)
    page.epochs.setValue(2)
    page.seed.setValue(47)
    result = page._training_call()
    try:
        assert result["available"] is True
        assert result["run_id"]
        metadata_path = Path(result["checkpoint"]["metadata"])
        arrays_path = Path(result["checkpoint"]["arrays"])
        assert metadata_path.is_file()
        assert arrays_path.is_file()

        registry = RunRegistry(manager.runs_dir / "runs.sqlite3")
        record = registry.get(result["run_id"])
        assert record.status == "completed"
        assert record.checkpoint == str(metadata_path)
        assert record.metrics
    finally:
        page.close()
        page.deleteLater()
        app.processEvents()


def test_developer_bot_resumes_session_and_renders_recovery_gate(
    app: QApplication,
    manager: WorkspaceManager,
) -> None:
    project = manager.create_project("bot-workflow")
    store = DeveloperSessionStore(
        manager.settings_dir / "developer-sessions.sqlite3",
        allowed_root=manager.workspace_root,
    )
    advisor = DeveloperAdvisor()
    session = advisor.start(
        project,
        ProjectBrief(
            "bot-workflow",
            "Detect a bounded teaching pattern",
            "Learners and reviewers",
            TaskKind.CLASSIFICATION,
            "Two finite numeric features",
            "One of two class labels",
            "Held-out F1 at least 0.80",
        ),
        ExperienceMode.BEGINNER,
    )
    session = advisor.answer(session, "recovery_owner", "Local project operator")
    session = advisor.answer(session, "restore_destination", str(project))
    session = advisor.answer(
        session,
        "recovery_drill_result",
        "No safe drill has passed yet; the entered active path must be corrected.",
    )
    persisted = store.save(session)

    page = DeveloperBotPage(manager)
    page.show()
    deadline = time.monotonic() + 15.0
    while page.panel.health_report is None and time.monotonic() < deadline:
        time.sleep(0.01)
        app.processEvents()
    try:
        assert page.panel.current_session is not None
        assert page.panel.current_session.id == persisted.id
        assert page.panel.stages.count() == len(STAGE_ORDER)
        assert "Inventory" in page.panel.headline.text()
        assert page.panel.health_report is not None, page.panel.health.toPlainText()
        assert page.panel.recovery_inventory is not None
        assert page.panel.recovery_inventory.session_present
        assert set(page.panel.question_editors) == {
            "recovery_owner",
            "restore_destination",
            "recovery_drill_result",
        }
        restore_editor = page.panel.question_editors["restore_destination"][1]
        assert restore_editor.text() == str(project)
        assert "OFFLINE" in page.panel.boundary.text()
    finally:
        # The evidence scan has completed, so this standalone widget can be
        # disposed without leaving an owned QThread behind.
        page.close()
        page.deleteLater()
        app.processEvents()


def test_developer_restore_handoff_preserves_destination_without_confirming(
    app: QApplication,
    manager: WorkspaceManager,
    tmp_path: Path,
) -> None:
    window = MainWindow(manager)
    target = tmp_path / "new-restored-workspace"
    window._handle_developer_tool("vault", {"destination": str(target)})
    try:
        vault = window.pages["vault"]
        assert window.current_page_key == "vault"
        assert vault.restore_destination.text() == str(target)
        assert not vault.restore_confirmation.isChecked()
        assert "independently validate" in vault.activity_output.toPlainText()
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()


def test_developer_tool_handoff_preserves_private_project_context(
    app: QApplication,
    manager: WorkspaceManager,
) -> None:
    project = manager.create_project("routed-project")
    window = MainWindow(manager)
    try:
        window._handle_developer_tool("training", {"project_root": str(project)})
        training = window.pages["training"]
        assert window.current_page_key == "training"
        assert Path(str(training.project.currentData())).resolve() == project.resolve()

        window._handle_developer_tool("workshop", {"project_root": str(project)})
        workshop = window.pages["workshop"]
        current = workshop.tree.currentItem()
        assert window.current_page_key == "workshop"
        assert current is not None
        assert Path(str(current.data(0, Qt.ItemDataRole.UserRole))).resolve() == project.resolve()
        assert current.isExpanded()
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()


def test_code_workshop_loads_project_folders_only_when_expanded(
    app: QApplication,
    manager: WorkspaceManager,
) -> None:
    project = manager.create_project("lazy-tree-project")
    window = MainWindow(manager)
    try:
        assert window.navigate("workshop")
        workshop = window.pages["workshop"]
        project_item = workshop.tree.topLevelItem(0)
        assert Path(str(project_item.data(0, Qt.ItemDataRole.UserRole))) == project
        assert project_item.data(0, workshop.TREE_LOADED_ROLE) is False
        assert project_item.childCount() == 1
        assert "Expand to load" in project_item.child(0).text(0)

        project_item.setExpanded(True)
        app.processEvents()
        assert project_item.data(0, workshop.TREE_LOADED_ROLE) is True
        assert "main.py" in {
            project_item.child(index).text(0) for index in range(project_item.childCount())
        }
    finally:
        window._live_scan_timer.stop()
        deadline = time.monotonic() + 5.0
        while window._running_background_tasks() and time.monotonic() < deadline:
            time.sleep(0.01)
            app.processEvents()
        assert not window._running_background_tasks()
        window.close()
        window.deleteLater()
        app.processEvents()


def test_developer_catalog_keeps_healthy_session_visible_and_recovers_one_bad_head(
    app: QApplication,
    manager: WorkspaceManager,
) -> None:
    advisor = DeveloperAdvisor()
    store = DeveloperSessionStore(
        manager.settings_dir / "developer-sessions.sqlite3",
        allowed_root=manager.workspace_root,
    )

    def make_session(name: str):
        project = manager.create_project(name)
        return advisor.start(
            project,
            ProjectBrief(
                name,
                f"Build the {name} teaching outcome",
                "Local learners",
                TaskKind.CLASSIFICATION,
                "Finite numeric features",
                "Two class labels",
                "Held-out F1 at least 0.80",
            ),
        )

    damaged_v1 = store.save(make_session("damaged-bot"))
    damaged_v2 = store.save(
        advisor.answer(damaged_v1, "data_source", "A versioned local teaching fixture"),
        expected_revision=damaged_v1.revision,
    )
    healthy = store.save(make_session("healthy-bot"))
    with sqlite3.connect(store.database) as connection:
        connection.execute(
            "UPDATE developer_revisions SET sha256=? WHERE session_id=? AND revision=?",
            ("0" * 64, damaged_v2.id, damaged_v2.revision),
        )

    page = DeveloperBotPage(manager)
    page.show()
    try:
        assert page.panel.sessions.count() == 2
        states = {
            str(page.panel.sessions.item(row).data(Qt.ItemDataRole.UserRole)): str(
                page.panel.sessions.item(row).data(Qt.ItemDataRole.UserRole + 1)
            )
            for row in range(page.panel.sessions.count())
        }
        assert states[healthy.id] == "healthy"
        assert states[damaged_v2.id] == "recovery_required"

        damaged_row = next(
            row
            for row in range(page.panel.sessions.count())
            if page.panel.sessions.item(row).data(Qt.ItemDataRole.UserRole) == damaged_v2.id
        )
        page.panel.sessions.setCurrentRow(damaged_row)
        assert page.panel.current_session is None
        assert "explicit recovery" in page.panel.reasons.toPlainText()
        page.panel._recover_session()
        assert store.load(damaged_v2.id).revision == damaged_v1.revision

        deadline = time.monotonic() + 5.0
        while page.panel.health_report is None and time.monotonic() < deadline:
            time.sleep(0.01)
            app.processEvents()
    finally:
        page.close()
        page.deleteLater()
        app.processEvents()
