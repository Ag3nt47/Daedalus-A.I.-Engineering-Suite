"""Project setup and reproducibility controls for the guided build workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from daedalus.gui.icons import semantic_icon
from daedalus.gui.widgets import Card, run_in_background
from daedalus.services.project_standards import (
    ProjectStandardsInspector,
    ProjectStandardsReport,
    ProjectStandardsService,
)


class ProjectStandardsPanel(QWidget):
    """Inspect and initialize a professional, provider-neutral project baseline."""

    status_changed = Signal(str, str)

    def __init__(self, manager: object, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.manager = manager
        self._project: Path | None = None
        self._generation = 0
        self._busy = False
        self.setAccessibleName("Project setup and reproducibility")

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        intro = Card(
            "Professional project setup",
            "Audit the current environment, create only missing baseline files, and capture "
            "reviewable reproducibility evidence. Optional tools are detected, never installed.",
            accent=True,
        )
        selector_row = QHBoxLayout()
        selector_label = QLabel("Private project")
        self.project_combo = QComboBox()
        self.project_combo.setAccessibleName("Project to audit for professional setup")
        self.project_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.project_combo.currentIndexChanged.connect(self._selection_changed)
        selector_row.addWidget(selector_label)
        selector_row.addWidget(self.project_combo, 1)
        intro.body.addLayout(selector_row)

        actions = QHBoxLayout()
        self.audit_button = QPushButton("Audit setup")
        self.audit_button.setObjectName("Primary")
        self.audit_button.setIcon(semantic_icon("guard", size=17))
        self.audit_button.setAccessibleName("Audit project setup and reproducibility")
        self.audit_button.clicked.connect(self.audit)
        self.initialize_button = QPushButton("Initialize missing standards")
        self.initialize_button.setIcon(semantic_icon("architecture", size=17))
        self.initialize_button.setAccessibleName(
            "Create missing professional project standard files"
        )
        self.initialize_button.setToolTip("Existing project files are never replaced.")
        self.initialize_button.clicked.connect(self.initialize_missing)
        self.capture_button = QPushButton("Capture environment")
        self.capture_button.setIcon(semantic_icon("backup", size=17))
        self.capture_button.setAccessibleName("Capture project environment evidence")
        self.capture_button.clicked.connect(self.capture_environment)
        self.open_button = QPushButton("Open project")
        self.open_button.setIcon(semantic_icon("folder", size=17))
        self.open_button.setAccessibleName("Open selected project in file manager")
        self.open_button.clicked.connect(self.open_project)
        for button in (
            self.audit_button,
            self.initialize_button,
            self.capture_button,
            self.open_button,
        ):
            actions.addWidget(button)
        actions.addStretch(1)
        intro.body.addLayout(actions)

        self.summary = QLabel("Choose a project to inspect its setup.")
        self.summary.setWordWrap(True)
        self.summary.setAccessibleName("Project setup audit summary")
        intro.add_widget(self.summary)
        root.addWidget(intro)

        findings_card = Card(
            "Readiness findings",
            "Blocking setup gaps, warnings, and the next concrete action.",
        )
        self.findings = QTableWidget(0, 4)
        self.findings.setHorizontalHeaderLabels(
            ["Status", "Check", "Location", "Next action"]
        )
        self.findings.verticalHeader().setVisible(False)
        self.findings.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.findings.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.findings.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents
        )
        self.findings.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch
        )
        self.findings.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeMode.ResizeToContents
        )
        self.findings.horizontalHeader().setSectionResizeMode(
            3, QHeaderView.ResizeMode.Stretch
        )
        self.findings.setAccessibleName("Project standard readiness findings")
        findings_card.add_widget(self.findings)
        root.addWidget(findings_card)

        tools_card = Card(
            "Optional engineering capabilities",
            "Detected integrations are informative. A missing optional tool does not fail the project.",
        )
        self.tools = QTableWidget(0, 4)
        self.tools.setHorizontalHeaderLabels(["Area", "Tool", "Detected", "Purpose"])
        self.tools.verticalHeader().setVisible(False)
        self.tools.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.tools.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.tools.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents
        )
        self.tools.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.ResizeToContents
        )
        self.tools.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeMode.ResizeToContents
        )
        self.tools.horizontalHeader().setSectionResizeMode(
            3, QHeaderView.ResizeMode.Stretch
        )
        self.tools.setAccessibleName("Detected optional AI engineering tools")
        tools_card.add_widget(self.tools)
        root.addWidget(tools_card)

        evidence_card = Card(
            "Reproducibility evidence",
            "Bounded local evidence only; project code is hashed but never imported or executed.",
        )
        self.evidence = QPlainTextEdit()
        self.evidence.setReadOnly(True)
        self.evidence.setMaximumBlockCount(500)
        self.evidence.setAccessibleName("Project reproducibility evidence summary")
        evidence_card.add_widget(self.evidence)
        root.addWidget(evidence_card)

        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.HLine)
        divider.setAccessibleName("End of project setup tools")
        root.addWidget(divider)
        root.addStretch(1)
        self.refresh_projects()

    @property
    def project(self) -> Path | None:
        return self._project

    def _project_paths(self) -> list[Path]:
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
        return sorted(projects.values(), key=lambda path: path.name.casefold())

    def refresh_projects(self, selected: Path | None = None) -> None:
        expected = selected or self._project
        if expected is not None:
            expected = expected.resolve(strict=False)
        projects = self._project_paths()
        self.project_combo.blockSignals(True)
        try:
            self.project_combo.clear()
            self.project_combo.addItem("Choose a project…", None)
            selected_index = 0
            for project in projects:
                self.project_combo.addItem(project.name, str(project))
                if expected is not None and project == expected:
                    selected_index = self.project_combo.count() - 1
            if selected_index == 0 and len(projects) == 1:
                selected_index = 1
            self.project_combo.setCurrentIndex(selected_index)
        finally:
            self.project_combo.blockSignals(False)
        value = self.project_combo.currentData()
        self._set_project_value(Path(str(value)) if value else None)

    def set_project(self, project: str | Path | None) -> bool:
        if project is None or not str(project).strip():
            self.refresh_projects(None)
            self.project_combo.setCurrentIndex(0)
            self._set_project_value(None)
            return True
        try:
            expected = Path(project).resolve(strict=False)
        except (OSError, TypeError, ValueError):
            return False
        self.refresh_projects(expected)
        current = self.project
        return current is not None and current == expected

    def _selection_changed(self, _index: int) -> None:
        value = self.project_combo.currentData()
        self._set_project_value(Path(str(value)) if value else None)

    def _set_project_value(self, project: Path | None) -> None:
        self._generation += 1
        self._project = project.resolve(strict=False) if project is not None else None
        enabled = self._project is not None and not self._busy
        for button in (
            self.audit_button,
            self.initialize_button,
            self.capture_button,
            self.open_button,
        ):
            button.setEnabled(enabled)
        if self._project is None:
            self.summary.setText("Choose a project to inspect its setup.")
            self.findings.setRowCount(0)
            self.tools.setRowCount(0)
            self.evidence.clear()
        else:
            self.summary.setText(
                f"Ready to audit {self._project.name}. The audit is read-only and offline."
            )

    def _set_busy(self, busy: bool, message: str = "") -> None:
        self._busy = busy
        enabled = self._project is not None and not busy
        for button in (
            self.audit_button,
            self.initialize_button,
            self.capture_button,
            self.open_button,
        ):
            button.setEnabled(enabled)
        if message:
            self.summary.setText(message)

    @staticmethod
    def _failure_message(error: str) -> str:
        lines = [line.strip() for line in str(error).splitlines() if line.strip()]
        return lines[-1] if lines else "Unknown project setup error"

    def _run_operation(
        self,
        label: str,
        function: Callable[[Path], Any],
        success: Callable[[Any, Path], None],
    ) -> None:
        project = self._project
        if project is None or self._busy:
            return
        self._generation += 1
        generation = self._generation
        self._set_busy(True, f"{label} {project.name}…")

        def complete(result: Any) -> None:
            if generation != self._generation or self._project != project:
                return
            self._set_busy(False)
            success(result, project)

        def failed(error: str) -> None:
            if generation != self._generation or self._project != project:
                return
            self._set_busy(False)
            message = self._failure_message(error)
            self.summary.setText(f"{label} failed safely: {message}")
            self.status_changed.emit(f"{label} failed: {message}", "danger")

        run_in_background(self, lambda: function(project), complete, failed)

    def audit(self) -> None:
        self._run_operation(
            "Auditing",
            lambda project: ProjectStandardsInspector(self.manager).inspect(project),
            self._audit_finished,
        )

    def _audit_finished(self, report: ProjectStandardsReport, project: Path) -> None:
        errors = report.error_count
        warnings = report.warning_count
        if errors:
            message = f"{project.name}: {errors} blocking setup issue(s), {warnings} warning(s)."
            level = "danger"
        elif warnings:
            message = f"{project.name}: setup is usable with {warnings} warning(s)."
            level = "warning"
        else:
            message = f"{project.name}: professional project baseline is ready."
            level = "success"
        self.summary.setText(message)
        self.summary.setObjectName(
            "Danger" if errors else "Warning" if warnings else "Success"
        )
        self.summary.style().unpolish(self.summary)
        self.summary.style().polish(self.summary)

        rows = list(report.findings)
        self.findings.setRowCount(max(1, len(rows)))
        if not rows:
            values = ("READY", "No blocking project-standard issue found", "—", "Continue")
            for column, value in enumerate(values):
                self.findings.setItem(0, column, QTableWidgetItem(value))
        else:
            for row, finding in enumerate(rows):
                status = finding.status.value.upper()
                values = (
                    status,
                    finding.summary,
                    finding.location or "—",
                    finding.action or "—",
                )
                for column, value in enumerate(values):
                    item = QTableWidgetItem(value)
                    if column == 0:
                        item.setData(Qt.ItemDataRole.UserRole, finding.status.value)
                    self.findings.setItem(row, column, item)

        self.tools.setRowCount(len(report.tools))
        for row, tool in enumerate(report.tools):
            detected = "Yes" if tool.available else "Not detected"
            if tool.version:
                detected = f"{detected} · {tool.version}"
            values = (tool.category, tool.label, detected, tool.purpose)
            for column, value in enumerate(values):
                self.tools.setItem(row, column, QTableWidgetItem(value))

        environment = report.environment
        self.evidence.setPlainText(
            "\n".join(
                (
                    f"Python: {environment.python_implementation} {environment.python_version}",
                    f"Daedalus: {environment.daedalus_version}",
                    f"Platform: {environment.operating_system} {environment.architecture}",
                    f"Environment: {environment.environment_kind}",
                    f"Compute: {environment.device_capability}",
                    f"Dependencies inventoried: {len(environment.dependencies)}",
                    f"Project source fingerprint: {environment.source_sha256}",
                    f"Source files/bytes hashed: {environment.source_files_hashed} / "
                    f"{environment.source_bytes_hashed}",
                    "Privacy: no project code imported or executed; no network request made.",
                )
            )
        )
        self.status_changed.emit(message, level)

    def initialize_missing(self) -> None:
        self._run_operation(
            "Initializing",
            lambda project: ProjectStandardsService(self.manager).initialize_missing(project),
            self._initialize_finished,
        )

    def _initialize_finished(self, created: tuple[Path, ...], project: Path) -> None:
        count = len(created)
        message = (
            f"Created {count} missing standard file(s) for {project.name}; existing files were preserved."
            if count
            else f"{project.name} already has every Daedalus project-standard file."
        )
        self.status_changed.emit(message, "success")
        self.summary.setText(message + " Auditing the result…")
        self.audit()

    def capture_environment(self) -> None:
        self._run_operation(
            "Capturing",
            lambda project: ProjectStandardsService(self.manager).capture_environment(project),
            self._capture_finished,
        )

    def _capture_finished(self, destination: Path, project: Path) -> None:
        message = f"Captured reviewable environment evidence for {project.name}: {destination.name}"
        self.summary.setText(message)
        self.status_changed.emit(message, "success")

    def open_project(self) -> None:
        project = self._project
        opener = getattr(self.manager, "open_in_file_manager", None)
        if project is None or not callable(opener):
            return
        try:
            opener(project)
        except Exception as exc:
            message = f"Could not open {project.name}: {exc}"
            self.summary.setText(message)
            self.status_changed.emit(message, "warning")


__all__ = ["ProjectStandardsPanel"]
