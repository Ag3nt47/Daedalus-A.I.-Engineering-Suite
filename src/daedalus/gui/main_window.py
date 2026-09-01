"""Responsive native application shell for the Daedalus AI Engineering Suite."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Any

from PySide6.QtCore import QSize, Qt, QTimer
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QStackedWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)
from shiboken6 import isValid as is_qt_object_valid

from daedalus.gui.fonts import ensure_runtime_fonts
from daedalus.gui.icons import semantic_icon
from daedalus.gui.pages import (
    ArchitectureBuilderPage,
    CalculatorLabPage,
    CodeWorkshopPage,
    DeveloperBotPage,
    LearningAtlasPage,
    MissionControlPage,
    ModelEvaluatorPage,
    ReleaseGuardPage,
    SettingsPage,
    TrainingLabPage,
    VaultBackupPage,
)
from daedalus.gui.theme import build_stylesheet, clamp_scale, reduced_motion
from daedalus.gui.widgets import PathField, StatusStrip, run_in_background

PAGE_SPECS = (
    ("mission", "Overview", "mission", "Progress, next action, and recent experiments"),
    ("developer", "1 · Define", "developer", "Choose the problem, evidence, risks, and success test"),
    ("learn", "2 · Learn", "learn", "Understand the concepts needed for the next build step"),
    ("architecture", "3 · Design", "architecture", "Assemble the network and validate tensor shapes"),
    (
        "calculator",
        "4 · Plan",
        "calculator",
        "Estimate resources and prototype bounded weight-generation workflows",
    ),
    ("training", "5 · Data & Train", "training", "Import, inspect, split, train, and monitor"),
    ("workshop", "6 · Build", "workshop", "Customize private project code safely"),
    ("evaluate", "7 · Evaluate", "evaluate", "Test checkpoints against held-out evidence"),
    ("vault", "8 · Protect", "backup", "Back up and verify recoverability"),
    ("guard", "9 · Release", "guard", "Check privacy, quality, and publication safety"),
    ("settings", "Settings", "settings", "Appearance and workspace custody"),
)


class NavigationButton(QToolButton):
    def __init__(self, key: str, title: str, icon_kind: str, description: str, parent=None) -> None:
        super().__init__(parent)
        self.key = key
        # Qt uses ampersands as mnemonic markers on buttons.  Escape them so
        # workspace names such as “Vault & Backup” render literally.
        self.full_title = title.replace("&", "&&")
        self.setObjectName("NavButton")
        self.setText(self.full_title)
        self.setIcon(semantic_icon(icon_kind, size=21))
        self.setIconSize(QSize(21, 21))
        self.setCheckable(True)
        self.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip(f"<b>{title}</b><br>{description}")
        self.setAccessibleName(f"Navigate to {title}")
        self.setAccessibleDescription(description)

    def set_compact(self, compact: bool) -> None:
        self.setText("" if compact else self.full_title)
        self.setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonIconOnly
            if compact
            else Qt.ToolButtonStyle.ToolButtonTextBesideIcon
        )
        self.setMinimumHeight(42)
        if compact:
            self.setMinimumWidth(48)
            self.setMaximumWidth(58)
        else:
            self.setMinimumWidth(0)
            self.setMaximumWidth(16_777_215)


class ProjectDiagnosticsDialog(QDialog):
    """Non-modal, selectable report for one bounded project diagnostics pass."""

    def __init__(self, report: object, manager: object, project: Path, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Project diagnostics — {project.name}")
        self.setAccessibleName(f"Diagnostics report for {project.name}")
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.resize(820, 600)
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(10)

        errors = max(0, int(getattr(report, "error_count", 0)))
        warnings = max(0, int(getattr(report, "warning_count", 0)))
        summary = QLabel(
            "No errors found in the bounded scan."
            if not errors and not warnings
            else f"{errors} error(s) · {warnings} warning(s)"
        )
        summary.setObjectName("Danger" if errors else "Warning" if warnings else "Success")
        summary.setAccessibleName("Project diagnostics result summary")
        root.addWidget(summary)

        self.report_text = QPlainTextEdit()
        self.report_text.setReadOnly(True)
        self.report_text.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.report_text.setAccessibleName("Full project diagnostics report")
        formatter = getattr(report, "format_text", None)
        self.report_text.setPlainText(
            str(formatter()) if callable(formatter) else str(report)
        )
        root.addWidget(self.report_text, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        open_logs = QPushButton("Open project logs folder")
        open_logs.setAccessibleName(f"Open logs folder for {project.name}")
        logs = project / "logs"
        open_logs.setEnabled(logs.is_dir() and not logs.is_symlink())

        def reveal_logs() -> None:
            opener = getattr(manager, "open_in_file_manager", None)
            if callable(opener):
                opener(logs)

        open_logs.clicked.connect(reveal_logs)
        buttons.addButton(open_logs, QDialogButtonBox.ButtonRole.ActionRole)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)


class LazyPageRegistry(Mapping[str, QWidget]):
    """Ordered page mapping that constructs each workspace on first use.

    Keeping the complete key set visible preserves the public ``window.pages``
    contract while avoiding the cost of building every tool, editor, table, and
    3D view before the first frame can be shown.
    """

    def __init__(
        self,
        factories: Mapping[str, Callable[[], QWidget]],
        on_created: Callable[[str, QWidget], None],
    ) -> None:
        self._factories = dict(factories)
        self._on_created = on_created
        self._loaded: dict[str, QWidget] = {}

    def __getitem__(self, key: str) -> QWidget:
        if key not in self._factories:
            raise KeyError(key)
        page = self._loaded.get(key)
        if page is None:
            page = self._factories[key]()
            self._loaded[key] = page
            self._on_created(key, page)
        return page

    def __iter__(self) -> Iterator[str]:
        return iter(self._factories)

    def __len__(self) -> int:
        return len(self._factories)

    def loaded(self, key: str) -> QWidget | None:
        """Return an existing page without triggering construction."""

        return self._loaded.get(key)

    @property
    def loaded_count(self) -> int:
        return len(self._loaded)

    def loaded_pages(self) -> tuple[QWidget, ...]:
        return tuple(self._loaded.values())


class MainWindow(QMainWindow):
    """Application shell; private work and public source stay visibly separated."""

    PAGE_SPECS = PAGE_SPECS
    LIVE_SCAN_BASE_INTERVAL_MS = 3_000
    LIVE_SCAN_MAX_INTERVAL_MS = 12_000

    def __init__(self, manager, parent=None) -> None:
        super().__init__(parent)
        # Offscreen/minimal Qt plugins can expose no system fonts at all.  Load
        # an existing local face before child widgets and QSS are constructed.
        self.runtime_fonts = ensure_runtime_fonts()
        self.manager = manager
        self._initial_project_inventory = self._project_paths()
        self._theme_name = "slate"
        self._ui_scale = 1.0
        self._fixed_scale = 1.0
        self._auto_scale = True
        self._reduced_motion = reduced_motion()
        self._close_when_idle = False
        self._watched_background_tasks: set[int] = set()
        self._active_project_path: Path | None = None
        self._project_progress_generation = 0
        self._diagnostics_generation = 0
        self._diagnostics_running = False
        self._diagnostics_pending_report = False
        self._last_diagnostics_report: object | None = None
        self._diagnostic_issue_count = 0
        self._live_change_token: str | None = None
        self._live_token_generation = 0
        self._live_token_running = False
        self._live_unchanged_polls = 0
        self._diagnostics_dialog: ProjectDiagnosticsDialog | None = None
        self._live_scan_timer = QTimer(self)
        self._live_scan_timer.setInterval(self.LIVE_SCAN_BASE_INTERVAL_MS)
        self._live_scan_timer.timeout.connect(self._poll_live_diagnostics)
        self._compact_layout: bool | None = None
        self._nav_buttons: dict[str, NavigationButton] = {}
        self._visited_page_keys: set[str] = set()
        self.pages: LazyPageRegistry
        self.page_titles = [spec[1] for spec in PAGE_SPECS]
        self.setWindowTitle("Daedalus AI Engineering Suite")
        self.setAccessibleName("Daedalus AI Engineering Suite main window")
        self.resize(1440, 900)
        self.setMinimumSize(780, 560)

        central = QWidget()
        central.setObjectName("AppRoot")
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        self.header = self._build_header()
        root.addWidget(self.header)
        root.addWidget(self._build_body(), 1)
        self.status_strip = StatusStrip()
        root.addWidget(self.status_strip)
        self._set_workspace_status()
        self.status_strip.set_motion_reduced(self._reduced_motion)
        self.setStyleSheet(build_stylesheet(self._theme_name, scale=self._ui_scale))
        self.navigate("mission")
        self._select_initial_project(self._initial_project_inventory)
        self._initial_project_inventory = []
        self._live_scan_timer.start()
        self._apply_responsive_layout()

    def _manager_path(self, name: str) -> Path:
        value = getattr(self.manager, name, None)
        return Path(value) if value is not None else Path()

    def _project_paths(self) -> list[Path]:
        """Return canonical private projects without trusting stale UI data."""

        try:
            values = self.manager.list_projects()
        except Exception:
            return []
        projects: dict[str, Path] = {}
        for value in values or ():
            try:
                path = Path(value).resolve(strict=False)
            except (OSError, TypeError, ValueError):
                continue
            projects[str(path).casefold()] = path
        return sorted(projects.values(), key=lambda item: item.name.casefold())

    def _populate_active_projects(
        self,
        selected: Path | None = None,
        *,
        projects: list[Path] | None = None,
    ) -> None:
        projects = self._project_paths() if projects is None else projects
        expected = selected.resolve(strict=False) if selected is not None else None
        self.active_project.blockSignals(True)
        try:
            self.active_project.clear()
            self.active_project.addItem("No active project", None)
            selected_index = 0
            for project in projects:
                self.active_project.addItem(project.name, str(project))
                if expected is not None and project == expected:
                    selected_index = self.active_project.count() - 1
            self.active_project.setCurrentIndex(selected_index)
        finally:
            self.active_project.blockSignals(False)

    def _select_initial_project(self, projects: list[Path] | None = None) -> None:
        """Restore the newest healthy guided project, or the sole project."""

        projects = self._project_paths() if projects is None else projects
        available = {str(path).casefold(): path for path in projects}
        selected: Path | None = None
        database = self._manager_path("settings_dir") / "developer-sessions.sqlite3"
        if database.is_file():
            try:
                from daedalus.developer import DeveloperSessionStore

                store = DeveloperSessionStore(
                    database,
                    allowed_root=self._manager_path("workspace_root"),
                )
                for entry in store.list_catalog():
                    session = entry.session
                    if entry.needs_recovery or session is None:
                        continue
                    candidate = Path(session.project_root).resolve(strict=False)
                    selected = available.get(str(candidate).casefold())
                    if selected is not None:
                        break
            except Exception:
                selected = None
        if selected is None and len(projects) == 1:
            selected = projects[0]
        # Reuse the canonical inventory already read above. Large workspaces
        # should not be re-enumerated for validation and combo population.
        self.set_active_project(selected, _verified_projects=projects)

    @property
    def active_project_path(self) -> Path | None:
        return self._active_project_path

    def _active_project_changed(self, _index: int) -> None:
        value = self.active_project.currentData()
        self.set_active_project(Path(str(value)) if value else None)

    def set_active_project(
        self,
        project: str | Path | None,
        *,
        _verified_projects: list[Path] | None = None,
    ) -> bool:
        """Select one verified private project and refresh its evidence progress."""

        projects = self._project_paths() if _verified_projects is None else _verified_projects
        candidate: Path | None = None
        if project is not None and str(project).strip():
            try:
                requested = Path(project).resolve(strict=False)
            except (OSError, TypeError, ValueError):
                return False
            available = {
                str(path).casefold(): path
                for path in projects
            }
            candidate = available.get(str(requested).casefold())
            if candidate is None:
                return False
        self._active_project_path = candidate
        self._populate_active_projects(candidate, projects=projects)
        self._diagnostics_generation += 1
        self._diagnostics_running = False
        self._diagnostics_pending_report = False
        self._last_diagnostics_report = None
        self._diagnostic_issue_count = 0
        self.diagnostics_button.setEnabled(candidate is not None)
        self.live_scan_button.setEnabled(candidate is not None)
        self._update_diagnostics_button_text()
        self._live_change_token = None
        self._live_token_generation += 1
        self._reset_live_scan_backoff()
        if candidate is not None:
            try:
                ensure_logs = getattr(self.manager, "ensure_project_logs", None)
                if callable(ensure_logs):
                    ensure_logs(candidate)
            except (OSError, PermissionError, RuntimeError, ValueError) as exc:
                if hasattr(self, "status_strip"):
                    self.set_status(
                        f"Project selected, but its automatic logs folder needs attention: {exc}",
                        "warning",
                    )
        developer = self.pages.loaded("developer")
        if isinstance(developer, DeveloperBotPage):
            developer.set_project(candidate)
        calculator = self.pages.loaded("calculator")
        if isinstance(calculator, CalculatorLabPage):
            calculator.set_project(candidate)
        self.refresh_project_progress()
        if candidate is not None and self.live_scan_button.isChecked():
            QTimer.singleShot(0, self._poll_live_diagnostics)
        return True

    @staticmethod
    def _gate_title(value: object) -> str:
        raw = getattr(value, "value", value)
        text = str(raw or "").strip().replace("_", " ")
        return text.title()

    def _show_no_project_progress(self) -> None:
        self.project_progress.setRange(0, 100)
        self.project_progress.setValue(0)
        self.project_progress.setFormat("No active project")
        self.project_progress_stage.setText("Create or open a project to begin")
        description = "No active project. Create or choose a private project to begin."
        self.project_progress.setAccessibleDescription(description)
        self.project_progress.setToolTip(description)

    def refresh_project_progress(self) -> None:
        """Inspect progress off the GUI thread and ignore stale completions."""

        self._project_progress_generation += 1
        generation = self._project_progress_generation
        project = self._active_project_path
        if project is None:
            self._show_no_project_progress()
            return

        self.project_progress.setRange(0, 100)
        self.project_progress.setValue(0)
        self.project_progress.setFormat("Checking evidence…")
        self.project_progress_stage.setText("Checking saved project evidence…")
        self.project_progress.setAccessibleDescription(
            f"Checking saved AI engineering evidence for {project.name}."
        )

        def inspect():
            from daedalus.developer.progress import ProjectProgressInspector

            return ProjectProgressInspector(self.manager).inspect(project)

        run_in_background(
            self,
            inspect,
            lambda snapshot, current=generation, expected=project: self._project_progress_finished(
                current, expected, snapshot
            ),
            lambda error, current=generation, expected=project: self._project_progress_failed(
                current, expected, error
            ),
        )

    def _project_progress_finished(
        self,
        generation: int,
        expected_project: Path,
        snapshot: object,
    ) -> None:
        if generation != self._project_progress_generation:
            return
        active = self._active_project_path
        if active is None or active.resolve(strict=False) != expected_project.resolve(strict=False):
            return
        try:
            reported = Path(str(getattr(snapshot, "project_root"))).resolve(strict=False)
        except (OSError, TypeError, ValueError):
            self._project_progress_failed(
                generation,
                expected_project,
                "Progress inspector returned no valid project identity.",
            )
            return
        if reported != expected_project.resolve(strict=False):
            return

        total = max(0, int(getattr(snapshot, "total", 0)))
        completed = min(total, max(0, int(getattr(snapshot, "completed", 0))))
        percent = min(100, max(0, int(getattr(snapshot, "percent", 0))))
        project_name = str(getattr(snapshot, "project_name", "") or expected_project.name)
        next_gate = getattr(snapshot, "next_gate", None)
        next_gate_title = str(getattr(snapshot, "next_gate_title", "") or "").strip()
        if total and completed >= total:
            stage_text = "All evidence gates complete"
        elif next_gate is not None:
            stage_text = f"Next gate: {next_gate_title or self._gate_title(next_gate)}"
        else:
            stage_text = "Start a guided build to create evidence gates"

        self.project_progress.setRange(0, 100)
        self.project_progress.setValue(percent)
        self.project_progress.setFormat(f"{percent}% complete")
        self.project_progress_stage.setText(stage_text)
        description = (
            f"{project_name}: {completed} of {total} evidence gates complete, "
            f"{percent} percent. {stage_text}."
        )
        self.project_progress.setAccessibleDescription(description)
        findings = tuple(getattr(snapshot, "findings", ()) or ())
        finding_text = "\n".join(str(item) for item in findings[:4])
        self.project_progress.setToolTip(
            description if not finding_text else f"{description}\n{finding_text}"
        )
        self.active_project.setToolTip(
            f"Active private project: {expected_project}\n"
            "Progress comes from saved evidence gates, not from visiting tabs."
        )

    def _project_progress_failed(
        self,
        generation: int,
        expected_project: Path,
        error: str,
    ) -> None:
        if generation != self._project_progress_generation:
            return
        active = self._active_project_path
        if active is None or active.resolve(strict=False) != expected_project.resolve(strict=False):
            return
        detail = str(error).strip().splitlines()[-1] if str(error).strip() else "unknown error"
        detail = detail[:180]
        self.project_progress.setRange(0, 100)
        self.project_progress.setValue(0)
        self.project_progress.setFormat("Progress unavailable")
        self.project_progress_stage.setText("Saved evidence could not be inspected")
        description = f"Progress for {expected_project.name} is unavailable: {detail}"
        self.project_progress.setAccessibleDescription(description)
        self.project_progress.setToolTip(description)

    def _project_change_token(self, project: Path) -> str:
        from daedalus.developer.diagnostics import ProjectDiagnosticsScanner

        return ProjectDiagnosticsScanner(self.manager).change_token(project)

    def _live_scan_toggled(self, enabled: bool) -> None:
        project = self._active_project_path
        self._live_token_generation += 1
        self._live_change_token = None
        self._reset_live_scan_backoff()
        if not enabled or project is None:
            if project is not None:
                self.set_status("Live project diagnostics paused.", "muted")
            return
        self.set_status(
            "Live project diagnostics enabled; changes will be scanned automatically.",
            "success",
        )
        QTimer.singleShot(0, self._poll_live_diagnostics)

    def _reset_live_scan_backoff(self) -> None:
        self._live_unchanged_polls = 0
        self._live_scan_timer.setInterval(self.LIVE_SCAN_BASE_INTERVAL_MS)

    def _back_off_live_scan(self) -> None:
        self._live_unchanged_polls += 1
        multiplier = 2 ** min(2, self._live_unchanged_polls // 2)
        self._live_scan_timer.setInterval(
            min(
                self.LIVE_SCAN_MAX_INTERVAL_MS,
                self.LIVE_SCAN_BASE_INTERVAL_MS * multiplier,
            )
        )

    def _poll_live_diagnostics(self) -> None:
        project = self._active_project_path
        if (
            project is None
            or not self.live_scan_button.isChecked()
            or self._diagnostics_running
            or self._live_token_running
        ):
            return
        generation = self._live_token_generation
        self._live_token_running = True
        run_in_background(
            self,
            lambda: self._project_change_token(project),
            lambda token, current=generation, expected=project: self._live_token_finished(
                current, expected, str(token)
            ),
            lambda _error, current=generation: self._live_token_failed(current),
        )

    def _live_token_finished(
        self,
        generation: int,
        expected_project: Path,
        token: str,
    ) -> None:
        self._live_token_running = False
        if generation != self._live_token_generation:
            return
        active = self._active_project_path
        if (
            active is None
            or active.resolve(strict=False) != expected_project.resolve(strict=False)
            or not self.live_scan_button.isChecked()
        ):
            return
        if self._live_change_token is None:
            self._live_change_token = token
            self._reset_live_scan_backoff()
            return
        if token == self._live_change_token:
            self._back_off_live_scan()
            return
        self._live_change_token = token
        self._reset_live_scan_backoff()
        self.scan_project_and_logs(show_report=False, reason="live project change")

    def _live_token_failed(self, generation: int) -> None:
        if generation != self._live_token_generation:
            return
        self._live_token_running = False
        self._back_off_live_scan()

    def scan_project_and_logs(
        self,
        _checked: bool = False,
        *,
        show_report: bool = True,
        reason: str = "manual request",
    ) -> None:
        """Run the bounded parser/log/integrity scanner outside the GUI thread."""

        project = self._active_project_path
        if project is None:
            return
        if self._diagnostics_running:
            self._diagnostics_pending_report |= bool(show_report)
            return
        self._diagnostics_generation += 1
        generation = self._diagnostics_generation
        self._diagnostics_running = True
        self._diagnostics_pending_report = False
        self.diagnostics_button.setEnabled(False)
        self.diagnostics_button.setText("Scanning…")
        self.set_status(f"Scanning {project.name} code, logs, runs, data, and checkpoints…")

        def inspect():
            from daedalus.developer.diagnostics import ProjectDiagnosticsScanner

            return ProjectDiagnosticsScanner(self.manager).scan(project)

        run_in_background(
            self,
            inspect,
            lambda report, current=generation, expected=project, reveal=show_report: (
                self._project_diagnostics_finished(current, expected, reveal, reason, report)
            ),
            lambda error, current=generation, expected=project: (
                self._project_diagnostics_failed(current, expected, error)
            ),
        )

    def _project_diagnostics_finished(
        self,
        generation: int,
        expected_project: Path,
        show_report: bool,
        reason: str,
        report: object,
    ) -> None:
        if generation != self._diagnostics_generation:
            return
        active = self._active_project_path
        if active is None or active.resolve(strict=False) != expected_project.resolve(strict=False):
            return
        try:
            reported = Path(str(getattr(report, "project_root"))).resolve(strict=False)
        except (OSError, TypeError, ValueError):
            self._project_diagnostics_failed(
                generation,
                expected_project,
                "Diagnostics returned no valid project identity.",
            )
            return
        if reported != expected_project.resolve(strict=False):
            return

        self._diagnostics_running = False
        self._last_diagnostics_report = report
        errors = max(0, int(getattr(report, "error_count", 0)))
        warnings = max(0, int(getattr(report, "warning_count", 0)))
        self._diagnostic_issue_count = errors + warnings
        self.diagnostics_button.setEnabled(True)
        self._update_diagnostics_button_text()
        self.diagnostics_button.setAccessibleDescription(
            f"Last read-only scan found {errors} errors and {warnings} warnings in "
            f"{expected_project.name}. Activate to scan again and open the report."
        )
        if self._live_change_token is None and self.live_scan_button.isChecked():
            QTimer.singleShot(0, self._poll_live_diagnostics)

        if errors:
            self.set_status(
                f"{reason.title()} scan found {errors} error(s) and {warnings} warning(s). "
                "Open the diagnostics report for exact locations.",
                "danger",
            )
        elif warnings:
            self.set_status(
                f"{reason.title()} scan found no errors and {warnings} warning(s).",
                "warning",
            )
        else:
            self.set_status(
                f"{reason.title()} scan found no errors in the bounded project scope.",
                "success",
            )
        self.refresh_project_progress()
        if show_report or self._diagnostics_pending_report:
            self._show_project_diagnostics(report, expected_project)
        self._diagnostics_pending_report = False

    def _project_diagnostics_failed(
        self,
        generation: int,
        expected_project: Path,
        error: str,
    ) -> None:
        if generation != self._diagnostics_generation:
            return
        active = self._active_project_path
        if active is None or active.resolve(strict=False) != expected_project.resolve(strict=False):
            return
        self._diagnostics_running = False
        self.diagnostics_button.setEnabled(True)
        self._update_diagnostics_button_text()
        detail = str(error).strip().splitlines()[-1] if str(error).strip() else "unknown error"
        self.set_status(f"Project diagnostics failed safely: {detail[:180]}", "danger")

    def _show_project_diagnostics(self, report: object, project: Path) -> None:
        dialog = ProjectDiagnosticsDialog(report, self.manager, project, self)
        self._diagnostics_dialog = dialog
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def _update_diagnostics_button_text(self) -> None:
        compact = self.width() < 1080
        suffix = f" ({self._diagnostic_issue_count})" if self._diagnostic_issue_count else ""
        self.diagnostics_button.setText(
            f"Scan{suffix}" if compact else f"Scan project & logs{suffix}"
        )

    def _build_header(self) -> QFrame:
        header = QFrame()
        header.setObjectName("AppHeader")
        header.setAccessibleName("Application header")
        layout = QVBoxLayout(header)
        layout.setContentsMargins(16, 8, 16, 8)
        layout.setSpacing(7)
        primary_row = QHBoxLayout()
        primary_row.setContentsMargins(0, 0, 0, 0)
        primary_row.setSpacing(14)

        brand = QWidget()
        brand_layout = QHBoxLayout(brand)
        brand_layout.setContentsMargins(0, 0, 0, 0)
        emblem = QLabel()
        emblem.setPixmap(semantic_icon("wing", size=38).pixmap(38, 38))
        emblem.setAccessibleName("Daedalus wing and neural labyrinth mark")
        brand_layout.addWidget(emblem)
        brand_text = QVBoxLayout()
        brand_text.setContentsMargins(0, 0, 0, 0)
        brand_text.setSpacing(0)
        title = QLabel("DAEDALUS")
        title.setObjectName("Brand")
        self.brand_tagline = QLabel("AI ENGINEERING SUITE")
        self.brand_tagline.setObjectName("BrandTagline")
        brand_text.addWidget(title)
        brand_text.addWidget(self.brand_tagline)
        brand_layout.addLayout(brand_text)
        primary_row.addWidget(brand)

        self.workspace_path = PathField(
            "External workspace",
            self._manager_path("workspace_root"),
            manager=self.manager,
            git_excluded=True,
        )
        self.workspace_path.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.workspace_path.line_edit.setReadOnly(True)
        self.workspace_path.line_edit.setCursorPosition(0)
        self.workspace_path.line_edit.setClearButtonEnabled(False)
        self.workspace_path.browse_button.setVisible(False)
        self.workspace_path.line_edit.setToolTip(
            "Active private workspace. Change custody paths through supported settings or migration tooling."
        )
        primary_row.addWidget(self.workspace_path, 1)

        self.backup_button = QPushButton("Backup now")
        self.backup_button.setIcon(semantic_icon("backup", size=20))
        self.backup_button.setObjectName("Success")
        self.backup_button.setAccessibleName("Back up external workspace now")
        self.backup_button.setToolTip(
            "Copy private projects and artifacts to the configured backup root. Nothing is staged to Git."
        )
        self.backup_button.clicked.connect(self.run_backup)
        primary_row.addWidget(self.backup_button)

        self.push_button = QPushButton("Safe Push")
        self.push_button.setIcon(semantic_icon("push", size=20))
        self.push_button.setObjectName("Warning")
        self.push_button.setAccessibleName("Run guarded GitHub safe push")
        self.push_button.setToolTip(
            "Open Release Guard and run its fail-closed Safe Push workflow. Private workspace paths remain excluded."
        )
        self.push_button.clicked.connect(self.run_safe_push)
        primary_row.addWidget(self.push_button)
        layout.addLayout(primary_row)

        progress_row = QHBoxLayout()
        progress_row.setContentsMargins(0, 0, 0, 0)
        progress_row.setSpacing(10)
        self.project_progress_label = QLabel("PROJECT PROGRESS")
        self.project_progress_label.setObjectName("BrandTagline")
        self.project_progress_label.setAccessibleName("Project progress heading")
        progress_row.addWidget(self.project_progress_label)

        self.active_project = QComboBox()
        self.active_project.setAccessibleName("Active private AI project")
        self.active_project.setAccessibleDescription(
            "Choose which private project the evidence-based build progress describes."
        )
        self.active_project.setToolTip(
            "Select the active private project. Progress comes from saved evidence gates, not from visiting tabs."
        )
        self.active_project.setMinimumContentsLength(16)
        self.active_project.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.active_project.addItem("No active project", None)
        self.active_project.currentIndexChanged.connect(self._active_project_changed)
        progress_row.addWidget(self.active_project)

        self.project_progress = QProgressBar()
        self.project_progress.setObjectName("ProjectProgress")
        self.project_progress.setRange(0, 100)
        self.project_progress.setValue(0)
        self.project_progress.setFormat("No active project")
        self.project_progress.setTextVisible(True)
        self.project_progress.setMinimumHeight(18)
        self.project_progress.setMaximumHeight(22)
        self.project_progress.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self.project_progress.setAccessibleName("Active AI project completion")
        self.project_progress.setAccessibleDescription(
            "No active project. Create or choose a private project to begin."
        )
        self.project_progress.setToolTip(
            "Evidence-based completion across the guided AI engineering gates."
        )
        progress_row.addWidget(self.project_progress, 1)

        self.project_progress_stage = QLabel("Create or open a project to begin")
        self.project_progress_stage.setObjectName("Muted")
        self.project_progress_stage.setAccessibleName("Current AI build gate")
        self.project_progress_stage.setToolTip(
            "The next evidence gate for the active private project."
        )
        progress_row.addWidget(self.project_progress_stage)

        self.diagnostics_button = QPushButton("Scan project & logs")
        self.diagnostics_button.setIcon(semantic_icon("guard", size=18))
        self.diagnostics_button.setEnabled(False)
        self.diagnostics_button.setAccessibleName("Scan active project and logs for problems")
        self.diagnostics_button.setAccessibleDescription(
            "Read-only scan of Python syntax, logs, run failures, data, checkpoints, and project health."
        )
        self.diagnostics_button.setToolTip(
            "Parse project code without executing it; check logs, failed runs, data, and checkpoints."
        )
        self.diagnostics_button.clicked.connect(self.scan_project_and_logs)
        progress_row.addWidget(self.diagnostics_button)

        self.live_scan_button = QPushButton("Live watch")
        self.live_scan_button.setIcon(semantic_icon("evaluate", size=18))
        self.live_scan_button.setCheckable(True)
        self.live_scan_button.setChecked(True)
        self.live_scan_button.setEnabled(False)
        self.live_scan_button.setAccessibleName("Toggle live project diagnostics")
        self.live_scan_button.setAccessibleDescription(
            "When enabled, Daedalus watches bounded file metadata and rescans after changes."
        )
        self.live_scan_button.setToolTip(
            "Near-real-time diagnostics after project code, logs, runs, data, or checkpoints change."
        )
        self.live_scan_button.toggled.connect(self._live_scan_toggled)
        progress_row.addWidget(self.live_scan_button)
        layout.addLayout(progress_row)
        return header

    def _build_body(self) -> QSplitter:
        body = QSplitter(Qt.Orientation.Horizontal)
        self.body_splitter = body
        body.setOpaqueResize(False)
        body.setChildrenCollapsible(False)
        body.setHandleWidth(6)

        self.sidebar = QFrame()
        self.sidebar.setObjectName("Sidebar")
        self.sidebar.setAccessibleName("Primary navigation")
        nav_layout = QVBoxLayout(self.sidebar)
        nav_layout.setContentsMargins(8, 14, 8, 14)
        nav_layout.setSpacing(5)
        self.nav_label = QLabel("GUIDED BUILD")
        self.nav_label.setObjectName("BrandTagline")
        nav_layout.addWidget(self.nav_label)

        self.nav_scroll = QScrollArea()
        self.nav_scroll.setObjectName("NavScroll")
        self.nav_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.nav_scroll.setWidgetResizable(True)
        self.nav_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.nav_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.nav_scroll.setAccessibleName("Scrollable primary navigation")
        self.nav_scroll.viewport().setProperty("navViewport", True)
        self.nav_host = QWidget()
        self.nav_host.setObjectName("NavHost")
        button_layout = QVBoxLayout(self.nav_host)
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.setSpacing(5)
        self.nav_group = QButtonGroup(self)
        self.nav_group.setExclusive(True)
        for key, title, icon, description in PAGE_SPECS:
            button = NavigationButton(key, title, icon, description)
            button.clicked.connect(lambda _checked=False, page_key=key: self.navigate(page_key))
            self.nav_group.addButton(button)
            self._nav_buttons[key] = button
            button_layout.addWidget(button)
        button_layout.addStretch(1)
        self.nav_scroll.setWidget(self.nav_host)
        self.nav_host.setAutoFillBackground(False)
        nav_layout.addWidget(self.nav_scroll, 1)
        self.boundary_label = QLabel("PRIVATE WORKSPACE\n≠ PUBLIC SOURCE")
        self.boundary_label.setObjectName("Success")
        self.boundary_label.setWordWrap(True)
        self.boundary_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.boundary_label.setAccessibleName(
            "Private workspace is separate from public source"
        )
        nav_layout.addWidget(self.boundary_label)
        body.addWidget(self.sidebar)

        self.page_stack = QStackedWidget()
        self.page_stack.setAccessibleName("Daedalus workspace pages")
        callbacks = {
            "backup": self.run_backup,
            "push": lambda: self.navigate("guard"),
            "navigate": self.navigate,
            "project": self.set_active_project,
            "status": self.set_status,
        }
        self.pages = LazyPageRegistry(
            {
                "mission": lambda: MissionControlPage(
                    self.manager,
                    callbacks,
                    initial_projects=self._initial_project_inventory,
                ),
                "developer": lambda: DeveloperBotPage(self.manager),
                "learn": lambda: LearningAtlasPage(self.manager),
                "architecture": lambda: ArchitectureBuilderPage(self.manager),
                "calculator": lambda: CalculatorLabPage(self.manager),
                "training": lambda: TrainingLabPage(self.manager),
                "workshop": lambda: CodeWorkshopPage(self.manager),
                "evaluate": lambda: ModelEvaluatorPage(self.manager),
                "vault": lambda: VaultBackupPage(self.manager),
                "guard": lambda: ReleaseGuardPage(self.manager),
                "settings": lambda: SettingsPage(self.manager),
            },
            self._page_created,
        )
        # Mission Control is the only workspace required for the first frame.
        self.pages["mission"]
        body.addWidget(self.page_stack)
        body.setStretchFactor(0, 0)
        body.setStretchFactor(1, 1)
        body.setSizes([230, 1210])
        return body

    def _page_created(self, key: str, page: QWidget) -> None:
        """Attach a newly materialized page and its page-specific signals."""

        title = next(
            (label for page_key, label, _icon, _description in PAGE_SPECS if page_key == key),
            key,
        )
        page.setAccessibleName(f"{title} workspace")
        page.setProperty("daedalusPageKey", key)
        self.page_stack.addWidget(page)

        if isinstance(page, ReleaseGuardPage | VaultBackupPage):
            page.status_changed.connect(self.set_status)
        if isinstance(page, SettingsPage):
            page.appearance_changed.connect(self.apply_appearance)
            page.set_theme(self._theme_name)
        if isinstance(page, DeveloperBotPage):
            page.status_changed.connect(self.set_status)
            page.navigate_requested.connect(self._handle_developer_tool)
            page.set_project(self._active_project_path)
        if isinstance(page, TrainingLabPage):
            page.project.currentIndexChanged.connect(self._training_project_selected)
        if isinstance(page, CodeWorkshopPage):
            page.project_selected.connect(self.set_active_project)
        if isinstance(page, CalculatorLabPage):
            page.status_changed.connect(self.set_status)
            page.open_in_workshop_requested.connect(self._open_weight_sandbox_in_workshop)
            page.set_reduced_motion(self._reduced_motion)
            page.set_project(self._active_project_path)
        if isinstance(page, ArchitectureBuilderPage):
            page.set_reduced_motion(self._reduced_motion)
        compact_setter = getattr(page, "set_compact_layout", None)
        if callable(compact_setter):
            compact_setter(self.width() < 1080)

    def _open_weight_sandbox_in_workshop(self, path: object) -> None:
        """Revalidate a Weight Lab draft through Code Workshop before opening it."""

        workshop = self.pages.get("workshop")
        if not isinstance(workshop, CodeWorkshopPage):
            self.set_status("Code Workshop is unavailable for this draft.", "danger")
            return
        try:
            candidate = Path(path)
        except (TypeError, ValueError):
            self.set_status("The Weight Lab draft path was invalid.", "danger")
            return
        workshop.refresh_tree()
        if not workshop.open_file(candidate):
            self.set_status("Code Workshop blocked the draft outside its private boundary.", "danger")
            return
        self.navigate("workshop")
        self.set_status(f"Opened Weight Lab draft in Code Workshop: {candidate.name}", "success")

    def _training_project_selected(self, _index: int) -> None:
        if self.current_page_key != "training":
            return
        training = self.pages.get("training")
        if not isinstance(training, TrainingLabPage):
            return
        project = training.project.currentData()
        if project:
            self.set_active_project(Path(str(project)))

    def _handle_developer_tool(self, key: str, payload: object) -> None:
        """Preserve validated bot handoff data when routing to a suite workspace."""

        self.navigate(key)
        if not isinstance(payload, dict):
            return
        project_text = payload.get("project_root")
        if project_text:
            self.set_active_project(Path(str(project_text)))
        if project_text and key == "training":
            training = self.pages.get("training")
            if isinstance(training, TrainingLabPage):
                expected = Path(str(project_text)).resolve(strict=False)
                training.refresh_projects()
                for index in range(training.project.count()):
                    value = training.project.itemData(index)
                    if value and Path(str(value)).resolve(strict=False) == expected:
                        training.project.setCurrentIndex(index)
                        break
        if project_text and key == "workshop":
            workshop = self.pages.get("workshop")
            if isinstance(workshop, CodeWorkshopPage):
                expected = Path(str(project_text)).resolve(strict=False)
                workshop.refresh_tree()
                for index in range(workshop.tree.topLevelItemCount()):
                    item = workshop.tree.topLevelItem(index)
                    value = item.data(0, Qt.ItemDataRole.UserRole)
                    if value and Path(str(value)).resolve(strict=False) == expected:
                        workshop.tree.setCurrentItem(item)
                        item.setExpanded(True)
                        break
        if key != "vault":
            return
        destination = payload.get("destination")
        vault = self.pages.get("vault")
        if destination and isinstance(vault, VaultBackupPage):
            vault.restore_destination.setText(str(destination))
            vault.restore_confirmation.setChecked(False)
            vault.activity_output.setPlainText(
                "AI Developer Bot supplied a validated new-directory-only proposal. "
                "Vault will independently validate it again; review the destination and "
                "confirm isolation before restoring."
            )

    @property
    def current_page_key(self) -> str:
        current = self.page_stack.currentWidget()
        return str(current.property("daedalusPageKey") or "mission") if current else "mission"

    def navigate(self, key: str) -> bool:
        page = self.pages.get(key)
        if page is None:
            return False
        first_visit = key not in self._visited_page_keys
        self._visited_page_keys.add(key)
        self.page_stack.setCurrentWidget(page)
        button = self._nav_buttons.get(key)
        if button is not None:
            button.setChecked(True)
        refresh = getattr(page, "refresh", None)
        if callable(refresh) and not first_visit:
            try:
                refresh()
            except Exception as exc:
                detail = str(exc).strip().splitlines()[-1] if str(exc).strip() else "unknown error"
                detail = detail[:220]
                self.status_strip.set_message(
                    f"{getattr(page, 'page_title', key)} refresh failed safely: "
                    f"{type(exc).__name__}: {detail}",
                    "danger",
                )
                return True
        self.status_strip.set_message(f"{getattr(page, 'page_title', key)} ready")
        return True

    def set_status(self, message: str, level: str = "muted") -> None:
        self.status_strip.set_message(message, level)

    def _set_workspace_status(self) -> None:
        path = self._manager_path("workspace_root")
        if hasattr(self, "workspace_path") and self.workspace_path.path != path:
            self.workspace_path.set_path(path)
        ready = bool(path and path.is_dir())
        if hasattr(self, "status_strip"):
            self.status_strip.set_workspace(path, ready)

    def _backup_call(self) -> Any:
        from daedalus.services.backup import BackupService

        return BackupService(self.manager).run()

    def run_backup(self) -> None:
        if not self.backup_button.isEnabled():
            return
        self.backup_button.setEnabled(False)
        self.set_status("Backing up private workspace…", "muted")
        run_in_background(self, self._backup_call, self._backup_finished, self._backup_failed)

    def _backup_finished(self, result: Any) -> None:
        self.backup_button.setEnabled(True)
        destination = ""
        if isinstance(result, dict):
            destination = str(result.get("destination") or result.get("path") or "")
        else:
            destination = str(getattr(result, "destination", "") or getattr(result, "path", ""))
        message = "Backup completed and verified."
        if destination:
            message += f" Destination: {destination}"
        self.set_status(message, "success")
        mission = self.pages.get("mission")
        if isinstance(mission, MissionControlPage):
            mission.refresh()
        self.refresh_project_progress()

    def _backup_failed(self, error: str) -> None:
        self.backup_button.setEnabled(True)
        summary = error.strip().splitlines()[-1] if error.strip() else "unknown error"
        self.set_status(f"Backup failed safely: {summary}", "danger")

    def run_safe_push(self) -> None:
        self.navigate("guard")
        guard = self.pages.get("guard")
        if isinstance(guard, ReleaseGuardPage):
            guard.start_safe_push()

    def apply_appearance(
        self,
        theme: str,
        fixed_scale: float = 1.0,
        auto_scale: bool = True,
        reduce_motion: bool = False,
    ) -> None:
        previous = (
            self._theme_name,
            self._fixed_scale,
            self._auto_scale,
            self._reduced_motion,
            self._ui_scale,
        )
        self._theme_name = str(theme or "slate")
        self._fixed_scale = clamp_scale(fixed_scale)
        self._auto_scale = bool(auto_scale)
        self._reduced_motion = bool(reduce_motion) or reduced_motion()
        self._ui_scale = self._compute_scale()
        style_changed = (
            self._theme_name != previous[0]
            or abs(self._ui_scale - previous[4]) >= 0.001
        )
        if style_changed:
            self.setStyleSheet(build_stylesheet(self._theme_name, scale=self._ui_scale))
        self.status_strip.set_motion_reduced(self._reduced_motion)
        calculator = self.pages.loaded("calculator")
        if isinstance(calculator, CalculatorLabPage):
            calculator.set_reduced_motion(self._reduced_motion)
        architecture = self.pages.loaded("architecture")
        if isinstance(architecture, ArchitectureBuilderPage):
            architecture.set_reduced_motion(self._reduced_motion)
        current = (
            self._theme_name,
            self._fixed_scale,
            self._auto_scale,
            self._reduced_motion,
            self._ui_scale,
        )
        message = (
            "Appearance is already up to date."
            if current == previous
            else f"Applied {self._theme_name} appearance at {self._ui_scale:.2f}×."
        )
        self.set_status(message, "success")

    def _compute_scale(self) -> float:
        if not self._auto_scale:
            return clamp_scale(self._fixed_scale)
        # Qt already expresses widget geometry in display-scaled logical pixels.
        # Applying another window-size-driven QSS scale both double-scales high
        # DPI displays and forces a costly repolish of every loaded widget after
        # a resize. Compact layouts handle space; the OS/Qt handle display DPI.
        return 1.0

    def _maybe_rescale(self) -> None:
        scale = round(self._compute_scale() / 0.05) * 0.05
        scale = clamp_scale(scale)
        if abs(scale - self._ui_scale) >= 0.049:
            self._ui_scale = scale
            self.setStyleSheet(build_stylesheet(self._theme_name, scale=scale))

    def _apply_responsive_layout(self) -> None:
        compact = self.width() < 1080
        if compact == self._compact_layout:
            return
        self._compact_layout = compact
        self.sidebar.setMinimumWidth(68 if compact else 210)
        self.sidebar.setMaximumWidth(76 if compact else 250)
        sidebar_width = 76 if compact else 230
        body_width = max(sidebar_width + 1, self.body_splitter.width())
        self.body_splitter.setSizes([sidebar_width, body_width - sidebar_width])
        for button in self._nav_buttons.values():
            button.set_compact(compact)
        self.nav_label.setVisible(not compact)
        self.boundary_label.setText(
            "PRIVATE\n≠\nPUBLIC" if compact else "PRIVATE WORKSPACE\n≠ PUBLIC SOURCE"
        )
        self.brand_tagline.setVisible(not compact)
        self.backup_button.setText("" if compact else "Backup now")
        self.push_button.setText("" if compact else "Safe Push")
        self.backup_button.setMinimumWidth(42 if compact else 0)
        self.push_button.setMinimumWidth(42 if compact else 0)
        self.workspace_path.copy_button.setVisible(not compact)
        self.workspace_path.reveal_button.setVisible(not compact)
        self.active_project.setMaximumWidth(180 if compact else 280)
        self.project_progress_stage.setVisible(not compact)
        self._update_diagnostics_button_text()
        self.live_scan_button.setText("Live" if compact else "Live watch")
        for page in self.pages.loaded_pages():
            compact_setter = getattr(page, "set_compact_layout", None)
            if callable(compact_setter):
                compact_setter(compact)

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt API
        self._apply_responsive_layout()
        super().resizeEvent(event)

    def _running_background_tasks(self) -> list[Any]:
        owners = [self, *self.findChildren(QWidget)]
        tasks: dict[int, Any] = {}
        for owner in owners:
            for task in getattr(owner, "_background_tasks", set()):
                if task.isRunning():
                    tasks[id(task)] = task
        return list(tasks.values())

    def _finish_deferred_close(self) -> None:
        if self._close_when_idle and not self._running_background_tasks():
            self._close_when_idle = False

            def close_if_alive() -> None:
                if is_qt_object_valid(self):
                    self.close()

            QTimer.singleShot(0, close_if_alive)

    def _background_task_finished(self, identity: int) -> None:
        self._watched_background_tasks.discard(identity)
        self._finish_deferred_close()

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API
        running = self._running_background_tasks()
        if not running:
            super().closeEvent(event)
            return
        event.ignore()
        self._close_when_idle = True
        self.set_status(
            f"Waiting for {len(running)} active operation(s) to finish before closing safely…",
            "warning",
        )
        for task in running:
            identity = id(task)
            if identity not in self._watched_background_tasks:
                self._watched_background_tasks.add(identity)

                def finish_if_alive(task_identity=identity) -> None:
                    if is_qt_object_valid(self):
                        self._background_task_finished(task_identity)

                task.finished.connect(finish_if_alive)
