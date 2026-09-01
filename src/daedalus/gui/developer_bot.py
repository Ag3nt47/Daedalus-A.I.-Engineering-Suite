"""Native, offline UI for the deterministic AI Developer Bot workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTabWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from daedalus.developer import (
    ARTIFACT_FILENAMES,
    NON_WAIVABLE_STAGES,
    STAGE_ORDER,
    ArtifactGenerator,
    ArtifactKind,
    DeveloperAdvisor,
    DeveloperSession,
    DeveloperSessionStore,
    ExperienceMode,
    GateState,
    HealthReport,
    ProjectBrief,
    ProjectEvidence,
    ProjectHealthInspector,
    Question,
    RecoveryInventory,
    RecoveryPlanner,
    SessionCatalogState,
    Stage,
    TaskKind,
)
from daedalus.gui.icons import semantic_icon
from daedalus.gui.widgets import run_in_background


class NewBuildDialog(QDialog):
    """Collect one canonical project brief before a session is persisted."""

    def __init__(self, project_name: str, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Start an AI Developer Bot session")
        self.setMinimumWidth(620)
        root = QVBoxLayout(self)
        notice = QLabel(
            "This is a local deterministic engineering advisor—not a language model. "
            "It does not use an API key, contact the internet, execute code, or publish files."
        )
        notice.setWordWrap(True)
        notice.setObjectName("Success")
        root.addWidget(notice)
        form = QFormLayout()
        self.project_name = QLineEdit(project_name)
        self.project_name.setReadOnly(True)
        self.outcome = QLineEdit()
        self.outcome.setPlaceholderText("What useful change should the system create?")
        self.users = QLineEdit()
        self.users.setPlaceholderText("Who uses or is affected by the output?")
        self.inputs = QLineEdit()
        self.inputs.setPlaceholderText("What is available at prediction time?")
        self.outputs = QLineEdit()
        self.outputs.setPlaceholderText("Exact classes, shape, units, or response format")
        self.success = QLineEdit()
        self.success.setPlaceholderText("Metric, population, threshold, and comparator")
        self.task = QComboBox()
        for kind in TaskKind:
            self.task.addItem(kind.value.replace("_", " ").title(), kind)
        self.mode = QComboBox()
        for mode in ExperienceMode:
            descriptions = {
                ExperienceMode.BEGINNER: "one explained decision at a time",
                ExperienceMode.BUILDER: "compact engineering checklist",
                ExperienceMode.EXPERT: "gate matrix and evidence review",
            }
            self.mode.addItem(f"{mode.value.title()} — {descriptions[mode]}", mode)
        fields = (
            ("Private project", self.project_name),
            ("Experience mode", self.mode),
            ("Task kind", self.task),
            ("Outcome", self.outcome),
            ("Users / affected people", self.users),
            ("Inputs", self.inputs),
            ("Outputs", self.outputs),
            ("Success measure", self.success),
        )
        for label, widget in fields:
            form.addRow(label, widget)
        root.addLayout(form)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Cancel | QDialogButtonBox.StandardButton.Ok
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Create local session")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    def accept(self) -> None:
        required = (self.outcome, self.users, self.inputs, self.outputs, self.success)
        if any(not field.text().strip() for field in required):
            QMessageBox.information(
                self,
                "Complete the brief",
                "Outcome, users, inputs, outputs, and success measure are all required.",
            )
            return
        super().accept()

    def brief(self) -> ProjectBrief:
        return ProjectBrief(
            project_name=self.project_name.text(),
            outcome=self.outcome.text(),
            users=self.users.text(),
            task_kind=self.task.currentData(),
            inputs=self.inputs.text(),
            outputs=self.outputs.text(),
            success_metric=self.success.text(),
        )


class DeveloperBotPanel(QWidget):
    """Three-pane project interview, evidence gate, and tool-routing surface."""

    navigate_requested = Signal(str, object)
    status_changed = Signal(str, str)

    def __init__(self, manager, parent=None) -> None:
        super().__init__(parent)
        self.manager = manager
        self.manager.bootstrap()
        self.advisor = DeveloperAdvisor()
        self.store = DeveloperSessionStore(
            Path(self.manager.settings_dir) / "developer-sessions.sqlite3",
            allowed_root=Path(self.manager.workspace_root),
        )
        self.current_session: DeveloperSession | None = None
        self.health_report: HealthReport | None = None
        self.recovery_inventory: RecoveryInventory | None = None
        self.recovery_proposal = None
        self.question_editors: dict[str, tuple[Question, QWidget]] = {}
        self.current_stage: Stage | None = None
        self._health_generation = 0
        self._build_ui()
        self.refresh()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        self._root_layout = root
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(8)
        self.boundary = QLabel(
            "OFFLINE EXPERT SYSTEM  •  NO API KEY  •  NO NETWORK  •  NO AUTOMATIC CODE EXECUTION"
        )
        self.boundary.setObjectName("Success")
        self.boundary.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.boundary.setAccessibleName("AI Developer Bot safety boundary")
        root.addWidget(self.boundary)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter = splitter
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(6)
        root.addWidget(splitter, 1)

        left = QFrame()
        left.setObjectName("Sidebar")
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(10, 10, 10, 10)
        left_layout.addWidget(QLabel("PRIVATE PROJECT"))
        self.project = QComboBox()
        self.project.setAccessibleName("AI Developer Bot project")
        left_layout.addWidget(self.project)
        project_row = QHBoxLayout()
        refresh_projects = QPushButton("Refresh")
        refresh_projects.clicked.connect(self._refresh_projects)
        self.new_button = QPushButton("New build")
        self.new_button.setObjectName("Primary")
        self.new_button.setIcon(semantic_icon("developer", size=17))
        self.new_button.clicked.connect(self._new_session)
        project_row.addWidget(refresh_projects)
        project_row.addWidget(self.new_button)
        left_layout.addLayout(project_row)
        left_layout.addWidget(QLabel("GUIDANCE MODE"))
        self.session_mode = QComboBox()
        for mode in ExperienceMode:
            self.session_mode.addItem(mode.value.title(), mode)
        self.session_mode.setAccessibleName("Current developer guidance mode")
        self.session_mode.currentIndexChanged.connect(self._change_session_mode)
        self.session_mode.setEnabled(False)
        left_layout.addWidget(self.session_mode)
        left_layout.addWidget(QLabel("RESUMABLE SESSIONS"))
        self.sessions = QListWidget()
        self.sessions.setAccessibleName("Saved AI Developer Bot sessions")
        self.sessions.currentItemChanged.connect(self._session_selected)
        left_layout.addWidget(self.sessions, 1)
        left_layout.addWidget(QLabel("BUILD FLOW"))
        self.stages = QListWidget()
        self.stages.setAccessibleName("AI engineering stage gates")
        self.stages.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        left_layout.addWidget(self.stages, 1)
        splitter.addWidget(left)

        center = QWidget()
        center_layout = QVBoxLayout(center)
        center_layout.setContentsMargins(10, 4, 10, 4)
        self.headline = QLabel("Start with a private project")
        self.headline.setStyleSheet("font-size:22px;font-weight:800;")
        self.headline.setWordWrap(True)
        center_layout.addWidget(self.headline)
        self.summary = QLabel(
            "Create or select a project, then start a local session. Every accepted answer is saved as an append-only revision."
        )
        self.summary.setWordWrap(True)
        self.summary.setObjectName("Muted")
        center_layout.addWidget(self.summary)
        self.reasons = QPlainTextEdit()
        self.reasons.setReadOnly(True)
        self.reasons.setMaximumHeight(120)
        self.reasons.setAccessibleName("Current engineering rationale")
        center_layout.addWidget(self.reasons)
        self.question_scroll = QScrollArea()
        self.question_scroll.setWidgetResizable(True)
        self.question_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.question_host = QWidget()
        self.question_form = QFormLayout(self.question_host)
        self.question_form.setContentsMargins(0, 4, 0, 4)
        self.question_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self.question_scroll.setWidget(self.question_host)
        self.question_host.setAutoFillBackground(False)
        center_layout.addWidget(self.question_scroll, 1)
        answer_row = QHBoxLayout()
        self.save_answers = QPushButton("Save responses")
        self.save_answers.setObjectName("Primary")
        self.save_answers.clicked.connect(self._save_responses)
        self.waive_button = QPushButton("Waive gate with rationale")
        self.waive_button.setToolTip(
            "Expert-mode exception for lower-risk gates. Protected gates can never be waived."
        )
        self.waive_button.clicked.connect(self._waive_current_gate)
        self.health_button = QPushButton("Re-scan evidence")
        self.health_button.clicked.connect(self._scan_health)
        answer_row.addWidget(self.save_answers)
        answer_row.addWidget(self.waive_button)
        answer_row.addWidget(self.health_button)
        answer_row.addStretch(1)
        center_layout.addLayout(answer_row)
        splitter.addWidget(center)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(10, 4, 10, 4)
        right_layout.addWidget(QLabel("EVIDENCE GATES"))
        self.gates = QTreeWidget()
        self.gates.setHeaderLabels(["Stage", "State"])
        self.gates.setRootIsDecorated(False)
        self.gates.setAccessibleName("AI engineering evidence gates")
        self.gates.header().setStretchLastSection(False)
        self.gates.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.gates.header().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        right_layout.addWidget(self.gates, 1)
        right_layout.addWidget(QLabel("READ-ONLY HEALTH"))
        self.health = QPlainTextEdit()
        self.health.setReadOnly(True)
        self.health.setMaximumHeight(150)
        self.health.setAccessibleName("Project health findings")
        right_layout.addWidget(self.health)
        right_layout.addWidget(QLabel("RECOVERY INVENTORY"))
        self.recovery = QPlainTextEdit()
        self.recovery.setReadOnly(True)
        self.recovery.setMaximumHeight(130)
        self.recovery.setAccessibleName("Recoverable work inventory")
        right_layout.addWidget(self.recovery)
        self.restore_destination = QLineEdit()
        self.restore_destination.setPlaceholderText(
            "Absolute new, nonexistent restore directory"
        )
        self.restore_destination.setAccessibleName("Proposed non-overwriting restore destination")
        right_layout.addWidget(self.restore_destination)
        self.validate_restore_button = QPushButton("Validate restore proposal")
        self.validate_restore_button.clicked.connect(self._validate_restore_proposal)
        right_layout.addWidget(self.validate_restore_button)
        right_layout.addWidget(QLabel("RECOMMENDED TOOLS"))
        self.tool_host = QWidget()
        self.tool_layout = QVBoxLayout(self.tool_host)
        self.tool_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(self.tool_host)
        artifact_row = QHBoxLayout()
        self.artifact_button = QPushButton("Generate missing draft plans")
        self.artifact_button.setIcon(semantic_icon("copy", size=17))
        self.artifact_button.clicked.connect(self._generate_artifacts)
        artifact_row.addWidget(self.artifact_button)
        right_layout.addLayout(artifact_row)
        transfer_row = QHBoxLayout()
        export_button = QPushButton("Export")
        export_button.clicked.connect(self._export_session)
        import_button = QPushButton("Import")
        import_button.clicked.connect(self._import_session)
        recover_button = QPushButton("Recover revision")
        recover_button.clicked.connect(self._recover_session)
        transfer_row.addWidget(export_button)
        transfer_row.addWidget(import_button)
        transfer_row.addWidget(recover_button)
        right_layout.addLayout(transfer_row)
        splitter.addWidget(right)
        self._responsive_panes = (left, center, right)
        self._compact_tabs: QTabWidget | None = None
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 1)
        splitter.setSizes([245, 560, 420])

    def set_compact_layout(self, compact: bool) -> None:
        """Present the three dense work areas as tabs at narrow widths."""

        compact = bool(compact)
        if compact and self._compact_tabs is None:
            tabs = QTabWidget(self)
            tabs.setAccessibleName("Compact Developer Bot work areas")
            scrolls: list[QScrollArea] = []
            for pane, title, minimum_height in zip(
                self._responsive_panes,
                ("Projects / flow", "Guidance", "Evidence / recovery"),
                (560, 460, 680),
            ):
                pane.setMinimumHeight(minimum_height)
                area = QScrollArea(tabs)
                area.setFrameShape(QFrame.Shape.NoFrame)
                area.setWidgetResizable(True)
                area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
                area.setWidget(pane)
                pane.setAutoFillBackground(False)
                tabs.addTab(area, title)
                scrolls.append(area)
            self._root_layout.removeWidget(self.splitter)
            self.splitter.hide()
            self._root_layout.insertWidget(1, tabs, 1)
            self._compact_tabs = tabs
            self._compact_scrolls = scrolls
            return
        if not compact and self._compact_tabs is not None:
            tabs = self._compact_tabs
            self._root_layout.removeWidget(tabs)
            for area, pane in zip(self._compact_scrolls, self._responsive_panes):
                area.takeWidget()
                pane.setMinimumHeight(0)
                self.splitter.addWidget(pane)
            self.splitter.setStretchFactor(0, 0)
            self.splitter.setStretchFactor(1, 1)
            self.splitter.setStretchFactor(2, 1)
            self.splitter.setSizes([245, 560, 420])
            self._root_layout.insertWidget(1, self.splitter, 1)
            self.splitter.show()
            tabs.deleteLater()
            self._compact_tabs = None
            self._compact_scrolls = []

    def _selected_project(self) -> Path | None:
        value = self.project.currentData()
        return Path(value) if value else None

    def _refresh_projects(self) -> None:
        selected = self.project.currentData()
        self.project.clear()
        try:
            projects = self.manager.list_projects()
        except Exception as exc:
            projects = []
            self.status_changed.emit(f"Project inventory failed safely: {exc}", "danger")
        for path in projects:
            self.project.addItem(Path(path).name, str(Path(path)))
        if selected:
            index = self.project.findData(selected)
            if index >= 0:
                self.project.setCurrentIndex(index)
        if not self.project.count():
            self.project.addItem("Create a project in Mission Control", None)
        self.new_button.setEnabled(self._selected_project() is not None)

    def refresh(self) -> None:
        current_id = self.current_session.id if self.current_session else None
        self._refresh_projects()
        self.sessions.blockSignals(True)
        self.sessions.clear()
        try:
            catalog = self.store.list_catalog()
        except Exception as exc:
            catalog = ()
            self.health.setPlainText(f"Session catalog needs recovery:\n{exc}")
        for entry in catalog:
            session = entry.session
            if session is None:
                title = f"Damaged session {entry.session_id[:8]}"
                detail = "NO VALID REVISION"
            else:
                title = f"{session.brief.project_name} · {session.mode.value.title()}"
                detail = f"rev {session.revision} · {entry.updated_utc[:19]}Z"
            if entry.state != SessionCatalogState.HEALTHY:
                detail += f" · {entry.state.value.replace('_', ' ').upper()}"
            item = QListWidgetItem(f"{title}\n{detail}")
            item.setData(Qt.ItemDataRole.UserRole, entry.session_id)
            item.setData(Qt.ItemDataRole.UserRole + 1, entry.state.value)
            if entry.state != SessionCatalogState.HEALTHY:
                item.setToolTip(
                    "This head is damaged. Select it and use Recover revision; healthy sessions remain available."
                )
            self.sessions.addItem(item)
        self.sessions.blockSignals(False)
        row = -1
        if current_id:
            for index in range(self.sessions.count()):
                if self.sessions.item(index).data(Qt.ItemDataRole.UserRole) == current_id:
                    row = index
                    break
        if row < 0 and self.sessions.count():
            row = 0
        if row >= 0:
            self.sessions.setCurrentRow(row)
        elif self.current_session is None:
            self._render_empty()

    def _new_session(self) -> None:
        project = self._selected_project()
        if project is None:
            self.status_changed.emit("Create a private project in Mission Control first.", "warning")
            self.navigate_requested.emit("mission", {})
            return
        dialog = NewBuildDialog(project.name, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        try:
            session = self.advisor.start(project, dialog.brief(), dialog.mode.currentData())
            self.current_session = self.store.save(session, event="saved")
            self.status_changed.emit("Created a crash-resumable local developer session.", "success")
            self.refresh()
        except Exception as exc:
            QMessageBox.warning(self, "Create developer session", f"Session creation failed safely:\n{exc}")

    def _session_selected(self, current: QListWidgetItem | None, _previous=None) -> None:
        if current is not None:
            state = str(current.data(Qt.ItemDataRole.UserRole + 1) or "healthy")
            if state != SessionCatalogState.HEALTHY.value:
                self.current_session = None
                self._render_empty(
                    "This session head needs explicit recovery. Healthy sessions are still listed; "
                    "select Recover revision to rewind only this damaged session."
                )
                return
            self._load_session(str(current.data(Qt.ItemDataRole.UserRole)))

    def _load_session(self, session_id: str) -> None:
        try:
            self.current_session = self.store.load(session_id)
            project = str(Path(self.current_session.project_root))
            index = self.project.findData(project)
            if index >= 0:
                self.project.setCurrentIndex(index)
            self.health_report = None
            self.recovery_inventory = None
            self.recovery_proposal = None
            self._render_session()
            self._scan_health()
        except Exception as exc:
            self.current_session = None
            self._render_empty(f"Session could not be loaded. Use Recover revision.\n{exc}")

    def _render_empty(self, detail: str = "") -> None:
        self.headline.setText("Start with a private project")
        self.summary.setText(
            "The bot turns an idea into staged engineering evidence and routes each next step to a Daedalus tool."
        )
        self.reasons.setPlainText(detail or "No developer session is selected.")
        self._clear_questions()
        self.gates.clear()
        self.stages.clear()
        for stage in STAGE_ORDER:
            self.stages.addItem(f"○ {stage.value.replace('_', ' ').title()}")
        self.save_answers.setEnabled(False)
        self.waive_button.setEnabled(False)
        self.waive_button.setVisible(False)
        self.artifact_button.setEnabled(False)
        self.health_button.setEnabled(False)
        self.session_mode.setEnabled(False)
        self.validate_restore_button.setEnabled(False)
        self.recovery.setPlainText("Select a session to inventory recoverable work.")
        self._clear_layout(self.tool_layout)

    @staticmethod
    def _clear_layout(layout) -> None:
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def _clear_questions(self) -> None:
        self.question_editors.clear()
        while self.question_form.count():
            item = self.question_form.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def _render_session(self) -> None:
        session = self.current_session
        if session is None:
            self._render_empty()
            return
        evidence = self.health_report.evidence if self.health_report else ProjectEvidence()
        turn = self.advisor.next_turn(session, evidence)
        self.current_stage = turn.stage
        self.session_mode.blockSignals(True)
        mode_index = self.session_mode.findData(session.mode)
        if mode_index >= 0:
            self.session_mode.setCurrentIndex(mode_index)
        self.session_mode.blockSignals(False)
        self.session_mode.setEnabled(True)
        plan = self.advisor.build_plan(session, evidence)
        self.headline.setText(f"{turn.headline} · {session.brief.project_name}")
        self.summary.setText(turn.summary)
        rationale = [*turn.reasons]
        if turn.gate.missing:
            rationale.append("\nRequired evidence:")
            rationale.extend(f"• {item}" for item in turn.gate.missing)
        self.reasons.setPlainText("\n".join(rationale))

        self.stages.clear()
        self.gates.clear()
        current_row = 0
        for index, step in enumerate(plan.steps):
            marker = {
                GateState.PASSED: "✓",
                GateState.WAIVED: "◇",
                GateState.BLOCKED: "!",
                GateState.UNKNOWN: "○",
            }[step.gate.state]
            self.stages.addItem(f"{marker} {step.title}")
            if step.stage == turn.stage:
                current_row = index
            item = QTreeWidgetItem(
                [step.title, step.gate.state.value.upper()]
            )
            item.setToolTip(0, "\n".join(step.gate.missing) or step.objective)
            self.gates.addTopLevelItem(item)
        self.stages.setCurrentRow(current_row)

        self._clear_questions()
        answers = dict(session.answers)
        stage_questions = self.advisor.questions(session, turn.stage)
        for question in stage_questions:
            editor: QWidget
            if question.value_type == "bool":
                combo = QComboBox()
                combo.addItem("Choose…", None)
                combo.addItem("Yes", True)
                combo.addItem("No", False)
                if question.id in answers:
                    combo.setCurrentIndex(1 if answers[question.id] is True else 2)
                editor = combo
            else:
                line = QLineEdit()
                if question.id in answers:
                    line.setText(str(answers[question.id]))
                elif question.recommended_answer is not None:
                    line.setPlaceholderText(f"Recommended starting point: {question.recommended_answer}")
                elif question.example:
                    line.setPlaceholderText(question.example)
                editor = line
            editor.setAccessibleName(question.prompt)
            editor.setToolTip(question.explanation)
            field = QWidget()
            field_layout = QVBoxLayout(field)
            field_layout.setContentsMargins(0, 0, 0, 6)
            field_layout.addWidget(editor)
            hint = QLabel(question.explanation)
            hint.setWordWrap(True)
            hint.setObjectName("Muted")
            field_layout.addWidget(hint)
            self.question_form.addRow(question.prompt, field)
            self.question_editors[question.id] = (question, editor)
        if not stage_questions:
            done = QLabel(
                "All interview responses for this stage are saved. Use the recommended tools "
                "to create the missing evidence, then re-scan."
            )
            done.setWordWrap(True)
            done.setObjectName("Success")
            self.question_form.addRow(done)
        self.save_answers.setEnabled(bool(stage_questions))
        self.waive_button.setVisible(session.mode == ExperienceMode.EXPERT)
        self.waive_button.setEnabled(
            session.mode == ExperienceMode.EXPERT
            and turn.stage not in NON_WAIVABLE_STAGES
            and turn.gate.state == GateState.BLOCKED
        )
        self.health_button.setEnabled(True)
        self.artifact_button.setEnabled(True)
        self.validate_restore_button.setEnabled(
            bool(self.recovery_inventory and self.recovery_inventory.backup_verified)
        )

        self._clear_layout(self.tool_layout)
        intents = list(turn.tool_intents)
        if turn.stage == Stage.RECOVERY:
            intents = [intent for intent in intents if intent.tool_key.value != "vault"]
            if self.recovery_proposal is not None:
                intents.append(RecoveryPlanner.tool_intent(self.recovery_proposal))
        for intent in intents:
            payload = dict(intent.payload)
            payload.setdefault("project_root", session.project_root)
            payload.setdefault("session_id", session.id)
            button = QPushButton(intent.label)
            button.setToolTip(intent.reason)
            button.setAccessibleDescription(intent.reason)
            button.setIcon(semantic_icon(intent.tool_key.value, size=17))
            button.clicked.connect(
                lambda _checked=False, key=intent.tool_key.value, handoff=payload: (
                    self.navigate_requested.emit(key, handoff)
                )
            )
            self.tool_layout.addWidget(button)

        if self.health_report is None:
            self.health.setPlainText("Evidence scan pending…")
        else:
            report = self.health_report
            lines = [
                f"{'READY' if report.ok else 'ATTENTION'} · {report.files_checked} files / "
                f"{report.bytes_checked:,} bytes inspected"
            ]
            lines.extend(
                f"[{finding.severity.value.upper()}] {finding.summary}"
                + (f" ({finding.location})" if finding.location else "")
                for finding in report.findings
            )
            if len(lines) == 1:
                lines.append("No blocking project-health findings.")
            self.health.setPlainText("\n".join(lines))
        inventory = self.recovery_inventory
        if inventory is None:
            self.recovery.setPlainText("Recovery inventory pending…")
        else:
            recovery_lines = [
                f"Project: {'present' if inventory.project_present else 'missing'}",
                f"Session: {'present' if inventory.session_present else 'missing'} "
                f"· {inventory.session_revision_count} revision(s)",
                f"Runs: {inventory.completed_run_count}/{inventory.run_count} completed",
                f"Checkpoints: {inventory.valid_checkpoint_count}/{inventory.checkpoint_count} verified",
                f"Backup: {'verified' if inventory.backup_verified else 'not verified'} "
                f"· {inventory.backup_file_count} file(s)",
            ]
            recovery_lines.extend(f"• {finding}" for finding in inventory.findings)
            self.recovery.setPlainText("\n".join(recovery_lines))

    def _save_responses(self) -> None:
        session = self.current_session
        if session is None:
            return
        updated = session
        try:
            for question, editor in self.question_editors.values():
                if isinstance(editor, QComboBox):
                    value: Any = editor.currentData()
                    if value is None:
                        continue
                else:
                    raw = editor.text().strip()  # type: ignore[attr-defined]
                    if not raw:
                        continue
                    value = int(raw) if question.value_type == "int" else raw
                updated = self.advisor.answer(updated, question.id, value)
            if updated.answers == session.answers:
                self.status_changed.emit("No new responses were entered.", "warning")
                return
            self.current_session = self.store.save(
                updated, expected_revision=session.revision, event="saved"
            )
            self.status_changed.emit("Responses saved as a new recoverable revision.", "success")
            self.refresh()
        except Exception as exc:
            QMessageBox.warning(self, "Save responses", f"Responses were not saved:\n{exc}")

    def _change_session_mode(self, _index: int) -> None:
        session = self.current_session
        mode = self.session_mode.currentData()
        if session is None or mode is None or not self.session_mode.isEnabled():
            return
        if ExperienceMode(mode) == session.mode:
            return
        try:
            updated = self.advisor.change_mode(session, ExperienceMode(mode))
            self.current_session = self.store.save(
                updated, expected_revision=session.revision, event="saved"
            )
            self.status_changed.emit(
                f"Switched this canonical session to {ExperienceMode(mode).value.title()} guidance.",
                "success",
            )
            self.refresh()
        except Exception as exc:
            QMessageBox.warning(self, "Change guidance mode", f"Mode change was not saved:\n{exc}")

    def _waive_current_gate(self) -> None:
        session = self.current_session
        stage = self.current_stage
        if session is None or stage is None or not self.waive_button.isEnabled():
            return
        reason, accepted = QInputDialog.getMultiLineText(
            self,
            "Record accountable gate waiver",
            "Explain the tradeoff, owner, and follow-up. This becomes revisioned evidence:",
        )
        if not accepted or not reason.strip():
            return
        try:
            updated = self.advisor.waive(session, stage, reason)
            self.current_session = self.store.save(
                updated, expected_revision=session.revision, event="saved"
            )
            self.status_changed.emit(
                f"Recorded an explicit waiver for the {stage.value} gate.", "warning"
            )
            self.refresh()
        except Exception as exc:
            QMessageBox.warning(self, "Waive gate", f"Gate waiver was refused:\n{exc}")

    def _health_call(self, session: DeveloperSession) -> tuple[HealthReport, RecoveryInventory]:
        inspector = ProjectHealthInspector(
            Path(session.project_root),
            Path(self.manager.workspace_root),
            source_root=Path(self.manager.source_root),
            backup_root=Path(self.manager.backup_root),
        )
        report = inspector.inspect(session)
        recovery = RecoveryPlanner(
            Path(session.project_root),
            Path(self.manager.workspace_root),
            Path(self.manager.backup_root),
            source_root=Path(self.manager.source_root),
        ).inventory(session_store=self.store, session_id=session.id)
        return report, recovery

    def _scan_health(self) -> None:
        session = self.current_session
        if session is None or not self.health_button.isEnabled():
            return
        self._health_generation += 1
        generation = self._health_generation
        session_id = session.id
        self.health_button.setEnabled(False)
        self.health.setPlainText("Running bounded read-only evidence scan…")
        run_in_background(
            self,
            lambda: self._health_call(session),
            lambda result: self._health_finished(generation, session_id, result),
            lambda error: self._health_failed(generation, session_id, error),
        )

    def _health_finished(self, generation: int, session_id: str, result: Any) -> None:
        if generation != self._health_generation:
            return
        self.health_button.setEnabled(True)
        if self.current_session is None or self.current_session.id != session_id:
            return
        self.health_report, self.recovery_inventory = result
        self._render_session()
        level = "success" if self.health_report.ok else "warning"
        self.status_changed.emit("Developer project evidence scan completed.", level)

    def _health_failed(self, generation: int, session_id: str, error: str) -> None:
        if generation != self._health_generation:
            return
        self.health_button.setEnabled(True)
        if self.current_session is None or self.current_session.id != session_id:
            return
        summary = error.strip().splitlines()[-1] if error.strip() else "unknown error"
        self.health.setPlainText(f"Evidence scan failed safely:\n{summary}")
        self.status_changed.emit("Developer evidence scan failed safely.", "danger")

    def _generate_artifacts(self) -> None:
        session = self.current_session
        if session is None:
            return
        evidence = self.health_report.evidence if self.health_report else ProjectEvidence()
        missing = tuple(
            kind
            for kind in ArtifactKind
            if not (Path(session.project_root) / ARTIFACT_FILENAMES[kind]).exists()
        )
        if not missing:
            self.status_changed.emit("All known planning artifacts already exist; none were overwritten.", "muted")
            return
        try:
            plan = self.advisor.build_plan(session, evidence)
            generator = ArtifactGenerator(Path(session.project_root), Path(self.manager.projects_dir))
            references = generator.generate(session, plan, evidence, kinds=missing)
            updated = session.with_artifacts(references)
            self.current_session = self.store.save(
                updated, expected_revision=session.revision, event="saved"
            )
            self.status_changed.emit(
                f"Generated {len(references)} non-overwriting draft plan artifact(s). "
                "Replace placeholder sections with evidence before gates can pass.",
                "success",
            )
            self.refresh()
        except Exception as exc:
            QMessageBox.warning(self, "Generate project plans", f"Artifacts were not generated:\n{exc}")

    def _export_session(self) -> None:
        session = self.current_session
        if session is None:
            return
        default = Path(session.project_root) / f"developer-session-{session.id[:8]}.json"
        selected, _filter = QFileDialog.getSaveFileName(
            self,
            "Export versioned developer session",
            str(default),
            "Daedalus session (*.json)",
        )
        if not selected:
            return
        destination = Path(selected).resolve(strict=False)
        if destination.exists():
            QMessageBox.warning(self, "Export session", "Export refused: the destination already exists.")
            return
        try:
            with destination.open("x", encoding="utf-8", newline="\n") as handle:
                handle.write(self.store.export_json(session.id))
            self.status_changed.emit(f"Exported session without overwriting: {destination}", "success")
        except Exception as exc:
            QMessageBox.warning(self, "Export session", f"Export failed safely:\n{exc}")

    def _import_session(self) -> None:
        project = self._selected_project()
        if project is None:
            self.status_changed.emit("Select the session's private project before importing.", "warning")
            return
        selected, _filter = QFileDialog.getOpenFileName(
            self,
            "Import versioned developer session",
            str(project),
            "Daedalus session (*.json)",
        )
        if not selected:
            return
        try:
            source = Path(selected)
            if source.stat().st_size > 2 * 1024 * 1024:
                raise ValueError("session export exceeds the 2 MiB import limit")
            imported = self.store.import_json(
                source.read_bytes(), expected_project_root=project, allow_replace=False
            )
            self.current_session = imported
            self.status_changed.emit("Imported a validated session as a recoverable revision.", "success")
            self.refresh()
        except Exception as exc:
            QMessageBox.warning(self, "Import session", f"Import was refused:\n{exc}")

    def _recover_session(self) -> None:
        item = self.sessions.currentItem()
        if item is None:
            self.status_changed.emit("Select a session to recover.", "warning")
            return
        session_id = str(item.data(Qt.ItemDataRole.UserRole))
        try:
            recovered = self.store.recover_last_valid(session_id)
            self.current_session = recovered
            self.status_changed.emit(
                f"Recovered the last valid committed session revision ({recovered.revision}).",
                "success",
            )
            self.refresh()
        except Exception as exc:
            QMessageBox.warning(self, "Recover session", f"Recovery could not find a valid revision:\n{exc}")

    def _validate_restore_proposal(self) -> None:
        session = self.current_session
        inventory = self.recovery_inventory
        if session is None or inventory is None:
            self.status_changed.emit("Run the recovery inventory before proposing a restore.", "warning")
            return
        raw = self.restore_destination.text().strip()
        if not raw:
            self.status_changed.emit("Enter an absolute, new restore destination.", "warning")
            return
        try:
            planner = RecoveryPlanner(
                Path(session.project_root),
                Path(self.manager.workspace_root),
                Path(self.manager.backup_root),
                source_root=Path(self.manager.source_root),
            )
            self.recovery_proposal = planner.propose_restore(Path(raw), inventory)
            self._render_session()
            self.status_changed.emit(
                "Restore proposal verified as new-directory-only. Review it in Vault & Backup.",
                "success",
            )
        except Exception as exc:
            self.recovery_proposal = None
            QMessageBox.warning(self, "Validate restore proposal", f"Proposal was refused:\n{exc}")


__all__ = ["DeveloperBotPanel", "NewBuildDialog"]
