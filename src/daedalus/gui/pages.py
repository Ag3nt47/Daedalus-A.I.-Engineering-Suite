"""Native workspaces for learning, building, evaluating, and releasing AI projects."""

from __future__ import annotations

import importlib
import inspect
import json
import re
import time
from dataclasses import asdict, is_dataclass
from html import escape
from pathlib import Path
from typing import Any, Callable, Iterable
from urllib.parse import urlencode

from PySide6.QtCore import Qt, QUrl, Signal
from PySide6.QtGui import QColor, QDesktopServices
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextBrowser,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from daedalus.gui.icons import semantic_icon
from daedalus.gui.theme import available_themes, reduced_motion
from daedalus.gui.widgets import (
    Card,
    InfoPanel,
    MetricTile,
    PageHeader,
    PathField,
    human_bytes,
    repolish,
    run_in_background,
)
from daedalus.resources import load_json

YOUTUBE_SEARCH_QUERIES = {
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

PROFESSIONAL_STAGE_GUIDANCE = {
    "mission": (
        "Professional teams manage the same lifecycle with Git, issue and decision records, "
        "configuration files, automated tests, artifact stores, and CI/CD. Daedalus keeps that "
        "route local and visible, then records evidence at each gate."
    ),
    "developer": (
        "Typical equivalents include product briefs, measurable acceptance criteria, risk "
        "registers, NIST AI RMF evidence, dataset cards, and model cards. The Developer Bot "
        "turns those practices into guided questions and persisted gates."
    ),
    "learn": (
        "Engineers commonly use Jupyter or VS Code alongside official PyTorch, JAX, "
        "TensorFlow, scikit-learn, and Hugging Face documentation. Daedalus provides an "
        "offline concept path before you choose an optional production framework."
    ),
    "architecture": (
        "Production modeling usually uses PyTorch, JAX, TensorFlow/Keras, scikit-learn, or "
        "Hugging Face. Graph viewers, ONNX/Netron, profilers, shape tests, and memory estimates "
        "complement Daedalus's inspectable layers and 3D architecture explorer."
    ),
    "calculator": (
        "Teams pair static estimates with PyTorch Profiler, TensorBoard, CUDA or ROCm tools, "
        "and one-batch smoke runs. Accelerate, DeepSpeed, Ray, FSDP, or cluster schedulers are "
        "introduced only when the model actually needs distributed compute."
    ),
    "training": (
        "A common stack combines a framework such as PyTorch with DVC or lakeFS for data, "
        "Hydra for resolved configuration, and MLflow or Weights & Biases for experiments. "
        "Daedalus supplies local checksum, split, run, metric, and checkpoint equivalents."
    ),
    "workshop": (
        "Professional code loops use Git, isolated Python environments, pytest, lint and type "
        "checks, CI, notebooks or an IDE, and often Docker. Daedalus keeps private project code "
        "separate and executes it through a bounded local runner."
    ),
    "evaluate": (
        "Production evaluation uses held-out and slice metrics, confusion matrices, robustness "
        "cases, baseline comparisons, latency budgets, and explicit promotion thresholds. "
        "Generative systems add curated prompts, groundedness, safety, and human review."
    ),
    "backup": (
        "Teams combine tested backups with object storage, DVC or lakeFS, and a model registry "
        "such as MLflow. Code, data, run metadata, and model artifacts have separate retention "
        "and recovery policies; a Git push is not a model backup."
    ),
    "guard": (
        "Release stacks commonly add CI/CD, dependency and secret scanning, SBOM/provenance, "
        "Docker/OCI, ONNX Runtime, Triton, KServe or vLLM, and OpenTelemetry metrics, logs, and "
        "traces. Daedalus validates local release evidence before any adapter is enabled."
    ),
    "settings": (
        "Professional environments may include uv, conda, Docker, CUDA, ROCm, cloud CLIs, and "
        "observability exporters. The Setup audit detects compatible tools without installing, "
        "importing, authenticating, or sending project information anywhere."
    ),
}
_YOUTUBE_SEARCH_ROOT = "https://www.youtube.com/results"
_YOUTUBE_SEARCH_HOSTS = frozenset({"youtube.com", "www.youtube.com"})


def _path_attr(manager, name: str, fallback: str = "") -> Path:
    value = getattr(manager, name, None)
    if value is None:
        return Path(fallback) if fallback else Path()
    return Path(value)


def _project_items(manager) -> list[Any]:
    try:
        values = manager.list_projects()
        return list(values or ())
    except Exception:
        root = _path_attr(manager, "projects_dir")
        if root.is_dir():
            return sorted((path for path in root.iterdir() if path.is_dir()), key=lambda p: p.name)
        return []


def _project_name(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("name") or value.get("project") or value.get("path") or "project")
    if hasattr(value, "name"):
        return str(value.name)
    return Path(str(value)).name or str(value)


def _pretty(value: Any) -> str:
    if value is None:
        return "Completed successfully."
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if hasattr(value, "to_dict") and callable(value.to_dict):
        value = value.to_dict()
    elif is_dataclass(value):
        value = asdict(value)
    elif hasattr(value, "__dict__"):
        value = vars(value)
    try:
        return json.dumps(value, indent=2, default=str)
    except (TypeError, ValueError):
        return str(value)


def _resolve_symbol(candidates: Iterable[tuple[str, str]]):
    for module_name, symbol_name in candidates:
        try:
            module = importlib.import_module(module_name)
            return getattr(module, symbol_name)
        except (ImportError, AttributeError):
            continue
    return None


def _call_supported(function: Callable[..., Any], **values: Any) -> Any:
    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):
        return function()
    if any(p.kind == p.VAR_KEYWORD for p in signature.parameters.values()):
        return function(**values)
    accepted = {name: value for name, value in values.items() if name in signature.parameters}
    return function(**accepted)


def _scroll(widget: QWidget) -> QScrollArea:
    area = QScrollArea()
    area.setFrameShape(QFrame.Shape.NoFrame)
    area.setWidgetResizable(True)
    area.setWidget(widget)
    # QScrollArea enables auto-fill on its content widget, which can paint the
    # system palette (bright white on Windows) through a dark application theme.
    widget.setAutoFillBackground(False)
    return area


def _deferred_tab(title: str, detail: str) -> QWidget:
    """Return a lightweight, accessible shell for an on-demand tool tab."""

    tab = QWidget()
    tab.setAccessibleName(f"{title} loads when opened")
    layout = QVBoxLayout(tab)
    layout.setContentsMargins(24, 24, 24, 24)
    layout.addStretch(1)
    heading = QLabel(title)
    heading.setObjectName("SectionTitle")
    heading.setAlignment(Qt.AlignmentFlag.AlignCenter)
    layout.addWidget(heading)
    explanation = QLabel(detail)
    explanation.setObjectName("Muted")
    explanation.setWordWrap(True)
    explanation.setAlignment(Qt.AlignmentFlag.AlignCenter)
    explanation.setAccessibleName(f"{title} loading behavior")
    layout.addWidget(explanation)
    layout.addStretch(1)
    return tab


def _replace_deferred_tab(
    tabs: QTabWidget,
    placeholder: QWidget,
    replacement: QWidget,
    label: str,
    tooltip: str,
) -> int:
    """Atomically replace a tab shell while preserving selection and ordering."""

    index = tabs.indexOf(placeholder)
    if index < 0:
        return tabs.indexOf(replacement)
    current = tabs.currentIndex()
    was_blocked = tabs.blockSignals(True)
    try:
        tabs.removeTab(index)
        tabs.insertTab(index, replacement, label)
        tabs.setTabToolTip(index, tooltip)
        tabs.setCurrentIndex(index if current == index else current)
    finally:
        tabs.blockSignals(was_blocked)
    placeholder.deleteLater()
    return index


class WorkspacePage(QWidget):
    """Common header and contextual Info tab for every primary workspace."""

    def __init__(
        self,
        manager,
        title: str,
        subtitle: str,
        icon: str,
        info_sections: Iterable[tuple[str, str]],
        parent=None,
        *,
        scroll_workspace: bool = True,
    ) -> None:
        super().__init__(parent)
        self.manager = manager
        self.page_title = title
        self.setObjectName(title.replace(" ", "") + "Page")
        self.setProperty("workspacePage", True)
        self.setAccessibleName(f"{title} workspace")
        root = QVBoxLayout(self)
        root.setContentsMargins(18, 14, 18, 14)
        root.setSpacing(12)
        root.addWidget(PageHeader(title, subtitle, icon, self))
        self.tabs = QTabWidget()
        self.tabs.setAccessibleName(f"{title} tabs")
        root.addWidget(self.tabs, 1)

        self.workspace_widget = QWidget()
        self.workspace_layout = QVBoxLayout(self.workspace_widget)
        self.workspace_layout.setContentsMargins(12, 12, 12, 12)
        self.workspace_layout.setSpacing(10)
        self.tabs.addTab(
            _scroll(self.workspace_widget) if scroll_workspace else self.workspace_widget,
            "Tools",
        )

        info = QWidget()
        info_layout = QVBoxLayout(info)
        info_layout.setContentsMargins(12, 12, 12, 12)
        info_layout.setSpacing(10)
        for index, (heading, text) in enumerate(info_sections):
            info_layout.addWidget(InfoPanel(heading, text, expanded=index == 0))

        professional_guidance = PROFESSIONAL_STAGE_GUIDANCE.get(icon)
        if professional_guidance:
            self.professional_guidance_panel = InfoPanel(
                "Professional equivalents",
                professional_guidance,
            )
            info_layout.addWidget(self.professional_guidance_panel)

        self.youtube_search_query = YOUTUBE_SEARCH_QUERIES.get(
            icon,
            f"{title} artificial intelligence tutorial",
        )
        self.youtube_search_url = QUrl(
            f"{_YOUTUBE_SEARCH_ROOT}?"
            f"{urlencode({'search_query': self.youtube_search_query})}"
        )
        youtube_href = escape(self.youtube_search_url.toString(), quote=True)
        youtube_title = escape(title)
        self.youtube_help_panel = InfoPanel(
            "Video tutorials",
            (
                f'<a href="{youtube_href}">Search YouTube for {youtube_title} tutorials</a>'
                "<br><small>Opens a pre-filled search in your default browser.</small>"
            ),
            expanded=True,
        )
        self.youtube_help_panel.setAccessibleName(f"Video tutorials for {title}")
        self.youtube_help_link = self.youtube_help_panel.content
        self.youtube_help_link.setTextFormat(Qt.TextFormat.RichText)
        self.youtube_help_link.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
        self.youtube_help_link.setOpenExternalLinks(False)
        self.youtube_help_link.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.youtube_help_link.setAccessibleName(f"Search YouTube tutorials for {title}")
        self.youtube_help_link.setAccessibleDescription(
            "Opens a pre-filled YouTube search for this step in the default browser."
        )
        self.youtube_help_link.setToolTip(
            "Open a stage-specific YouTube search in the default browser"
        )
        self.youtube_help_link.linkActivated.connect(self._open_youtube_search)
        info_layout.addWidget(self.youtube_help_panel)
        info_layout.addStretch(1)
        self.info_widget = info
        self.tabs.addTab(_scroll(info), "Info")

    def _open_youtube_search(self, href: str) -> bool:
        """Open only the exact, generated YouTube search link for this workspace."""

        url = QUrl(str(href))
        expected_href = self.youtube_search_url.toString()
        if (
            str(href) != expected_href
            or not url.isValid()
            or url.scheme().casefold() != "https"
            or url.host().casefold() not in _YOUTUBE_SEARCH_HOSTS
            or url.path() != "/results"
        ):
            return False
        return bool(QDesktopServices.openUrl(url))

    def show_info(self) -> None:
        self.tabs.setCurrentIndex(self.tabs.count() - 1)


class DeveloperBotPage(WorkspacePage):
    """Offline engineering mentor with persisted stage gates and recovery evidence."""

    status_changed = Signal(str, str)
    navigate_requested = Signal(str, object)

    def __init__(self, manager, parent=None) -> None:
        from daedalus.gui.developer_bot import DeveloperBotPanel

        super().__init__(
            manager,
            "AI Developer Bot",
            "Turn an idea into a recoverable, evidence-gated AI engineering project.",
            "developer",
            (
                (
                    "What this bot is",
                    "A deterministic offline expert system built into Daedalus. It asks the "
                    "questions an AI engineer should ask, explains each gate, inventories recoverable "
                    "work, and routes you to the correct suite tool. It is not a language model and "
                    "does not require an API key.",
                ),
                (
                    "Ten-gate workflow",
                    "Discovery, recovery, data readiness, baseline, architecture, experiment, "
                    "evaluation, deployment, security, and release each require visible answers "
                    "and evidence. High-risk gates cannot be waived.",
                ),
                (
                    "Crash and data safety",
                    "Every accepted answer becomes an append-only SQLite revision with a checksum. "
                    "Planning artifacts never overwrite existing project files. Recovery inventory "
                    "is read-only; restore is proposed only to a new directory and remains an explicit "
                    "Vault & Backup action.",
                ),
            ),
            parent,
            scroll_workspace=False,
        )
        self.panel = DeveloperBotPanel(manager)
        self.panel.status_changed.connect(self.status_changed.emit)
        self.panel.navigate_requested.connect(self.navigate_requested.emit)
        self.workspace_layout.addWidget(self.panel, 1)
        self._setup_panel: Any | None = None
        self._pending_setup_project: str | Path | None = None
        self._setup_placeholder = _deferred_tab(
            "Professional project setup",
            "Environment discovery and reproducibility tooling load when Setup is opened. "
            "The guided Developer Bot remains ready without importing those optional inspectors.",
        )
        self.tabs.insertTab(self.tabs.count() - 1, self._setup_placeholder, "Setup")
        self.tabs.setTabToolTip(
            self.tabs.indexOf(self._setup_placeholder),
            "Audit local tools and reproducibility evidence on demand",
        )
        self.tabs.currentChanged.connect(self._materialize_developer_tab)

    @property
    def setup_panel(self):
        return self._ensure_setup_panel()

    def _materialize_developer_tab(self, index: int) -> None:
        if self.tabs.widget(index) is self._setup_placeholder:
            self._ensure_setup_panel()

    def _ensure_setup_panel(self):
        panel = self._setup_panel
        if panel is not None:
            return panel
        from daedalus.gui.project_standards import ProjectStandardsPanel

        panel = ProjectStandardsPanel(self.manager)
        panel.status_changed.connect(self.status_changed.emit)
        if self._pending_setup_project is not None:
            panel.set_project(self._pending_setup_project)
        self._setup_panel = panel
        scroll = _scroll(panel)
        _replace_deferred_tab(
            self.tabs,
            self._setup_placeholder,
            scroll,
            "Setup",
            "Audit local tools and reproducibility evidence on demand",
        )
        return panel

    def refresh(self) -> None:
        self.panel.refresh()
        if self._setup_panel is not None:
            self._setup_panel.refresh_projects(self._setup_panel.project)

    def set_project(self, project: str | Path | None) -> bool:
        self._pending_setup_project = project
        if self._setup_panel is not None:
            return bool(self._setup_panel.set_project(project))
        return True

    def set_compact_layout(self, compact: bool) -> None:
        self.panel.set_compact_layout(compact)


class MissionControlPage(WorkspacePage):
    def __init__(
        self,
        manager,
        callbacks: dict[str, Callable] | None = None,
        parent=None,
        *,
        initial_projects: list[Any] | None = None,
    ) -> None:
        super().__init__(
            manager,
            "Mission Control",
            "A guided route from first idea to trained, evaluated, and protected AI project.",
            "mission",
            (
                (
                    "How to use Daedalus",
                    "Follow the numbered build stages in the left rail. Every stage explains its outcome and "
                    "opens the tool that helps you complete it; status comes from artifacts in your "
                    "private workspace, not from a decorative checklist.",
                ),
                (
                    "Local-first boundary",
                    "Public application source stays in the install folder. Projects, datasets, "
                    "checkpoints, logs, credentials, and model artifacts live in the external "
                    "workspace and are excluded from Git publication.",
                ),
                (
                    "Backup is not publication",
                    "Backup copies private work to the configured backup root. Safe Push publishes "
                    "only repository-approved source after Release Guard checks; the two actions are independent.",
                ),
            ),
            parent,
        )
        self.callbacks = callbacks or {}
        self.metric_layout = QGridLayout()
        self.metric_layout.setHorizontalSpacing(10)
        self.metric_layout.setVerticalSpacing(10)
        self.projects_metric = MetricTile("Projects", "0", "External, private workspaces")
        self.datasets_metric = MetricTile("Datasets", "—", "Stored outside public source")
        self.checkpoints_metric = MetricTile("Checkpoints", "—", "Non-executable model arrays")
        self.backup_metric = MetricTile("Backup root", "—", "Independent of GitHub")
        self.metric_tiles = (
            self.projects_metric,
            self.datasets_metric,
            self.checkpoints_metric,
            self.backup_metric,
        )
        for column, tile in enumerate(self.metric_tiles):
            self.metric_layout.addWidget(tile, 0, column)
            self.metric_layout.setColumnStretch(column, 1)
        self.workspace_layout.addLayout(self.metric_layout)

        self.workspace_path = PathField(
            "External workspace",
            _path_attr(manager, "workspace_root"),
            manager=manager,
            git_excluded=True,
        )
        self.workspace_path.line_edit.setReadOnly(True)
        self.workspace_path.line_edit.setCursorPosition(0)
        self.workspace_path.line_edit.setClearButtonEnabled(False)
        self.workspace_path.browse_button.setVisible(False)
        self.workspace_path.line_edit.setToolTip(
            "Active private workspace. Change custody through supported migration tooling."
        )
        self.workspace_layout.addWidget(self.workspace_path)

        action_card = Card(
            "Next actions",
            "Start with a guided brief, or move directly to data when you already know the problem.",
            accent=True,
        )
        self.next_step_label = QLabel("Checking your next useful step…")
        self.next_step_label.setObjectName("Success")
        self.next_step_label.setWordWrap(True)
        self.next_step_label.setAccessibleName("Recommended next AI build step")
        action_card.add_widget(self.next_step_label)
        self.action_layout = QGridLayout()
        self.action_layout.setHorizontalSpacing(8)
        self.action_layout.setVerticalSpacing(8)
        guided = QPushButton("Start guided build")
        guided.setObjectName("Primary")
        guided.setIcon(semantic_icon("developer", size=18))
        guided.setAccessibleName("Start the guided AI build workflow")
        guided.clicked.connect(lambda: self._call("navigate", "developer"))
        create = QPushButton("Create project")
        create.setIcon(semantic_icon("architecture", size=18))
        create.setAccessibleName("Create a new private project")
        create.clicked.connect(self._create_project)
        data = QPushButton("Prepare training data")
        data.setIcon(semantic_icon("training", size=18))
        data.setAccessibleName("Open training data intake and analysis")
        data.clicked.connect(lambda: self._call("navigate", "training"))
        backup = QPushButton("Back up now")
        backup.setIcon(semantic_icon("backup", size=18))
        backup.setAccessibleName("Run workspace backup now")
        backup.clicked.connect(lambda: self._call("backup"))
        reveal = QPushButton("Open workspace")
        reveal.setIcon(semantic_icon("folder", size=18))
        reveal.setAccessibleName("Open external workspace in file manager")
        reveal.clicked.connect(self.workspace_path.reveal_path)
        self.action_buttons = (guided, create, data, backup, reveal)
        for column, button in enumerate(self.action_buttons):
            self.action_layout.addWidget(button, 0, column)
        self.action_layout.setColumnStretch(len(self.action_buttons), 1)
        action_card.body.addLayout(self.action_layout)
        self.workspace_layout.addWidget(action_card)

        locations = Card("Workspace map", "Every generated artifact has a declared owner and path.")
        self.location_table = QTableWidget(0, 3)
        self.location_table.setHorizontalHeaderLabels(["Area", "Location", "Git policy"])
        self.location_table.verticalHeader().setVisible(False)
        self.location_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.location_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.location_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.location_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.location_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        locations.add_widget(self.location_table)
        info_layout = self.info_widget.layout()
        if isinstance(info_layout, QVBoxLayout):
            info_layout.insertWidget(0, locations)

        recovery = Card(
            "Recovery trail",
            "Recent experiments are journaled before execution. Interrupted runs remain visible instead of disappearing.",
        )
        self.recent_runs = QTableWidget(0, 5)
        self.recent_runs.setHorizontalHeaderLabels(
            ["Status", "Project", "Dataset", "Updated (UTC)", "Checkpoint"]
        )
        self.recent_runs.verticalHeader().setVisible(False)
        self.recent_runs.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.recent_runs.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.recent_runs.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents
        )
        self.recent_runs.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.ResizeToContents
        )
        self.recent_runs.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeMode.ResizeToContents
        )
        self.recent_runs.horizontalHeader().setSectionResizeMode(
            3, QHeaderView.ResizeMode.ResizeToContents
        )
        self.recent_runs.horizontalHeader().setSectionResizeMode(
            4, QHeaderView.ResizeMode.Stretch
        )
        self.recent_runs.setAccessibleName("Recent recoverable training runs")
        recovery.add_widget(self.recent_runs)
        self.workspace_layout.addWidget(recovery, 1)
        self._compact_layout = False
        self._refresh_generation = 0
        self._refresh_running = False
        self._refresh_pending = False
        self._pending_projects: list[Any] | None = None
        self._run_registry: Any | None = None
        self._has_snapshot = False
        self.refresh(projects=initial_projects)

    def set_compact_layout(self, compact: bool) -> None:
        compact = bool(compact)
        if compact == self._compact_layout:
            return
        self._compact_layout = compact
        metric_columns = 2 if compact else 4
        action_columns = 2 if compact else len(self.action_buttons)
        for tile in self.metric_tiles:
            self.metric_layout.removeWidget(tile)
        for column in range(4):
            self.metric_layout.setColumnStretch(column, 0)
        for index, tile in enumerate(self.metric_tiles):
            self.metric_layout.addWidget(tile, index // metric_columns, index % metric_columns)
        for column in range(metric_columns):
            self.metric_layout.setColumnStretch(column, 1)
        for button in self.action_buttons:
            self.action_layout.removeWidget(button)
        for index, button in enumerate(self.action_buttons):
            self.action_layout.addWidget(button, index // action_columns, index % action_columns)
        for column in range(len(self.action_buttons) + 1):
            self.action_layout.setColumnStretch(column, 0)
        if compact:
            for column in range(action_columns):
                self.action_layout.setColumnStretch(column, 1)
        else:
            self.action_layout.setColumnStretch(len(self.action_buttons), 1)

    def _call(self, name: str, *args: Any) -> None:
        callback = self.callbacks.get(name)
        if callable(callback):
            callback(*args)

    def _create_project(self) -> None:
        name, ok = QInputDialog.getText(self, "Create project", "Project name:")
        if not ok or not name.strip():
            return
        cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", name.strip()).strip("-.")
        if not cleaned:
            QMessageBox.information(self, "Create project", "Enter a project name using letters or numbers.")
            return
        try:
            project = self.manager.create_project(cleaned, template="minimal")
            self.refresh()
            self._call("project", project)
            callback = self.callbacks.get("status")
            if callable(callback):
                callback(f"Created private project: {cleaned}", "success")
        except Exception as exc:
            QMessageBox.warning(self, "Create project", f"Project creation failed:\n{exc}")

    def refresh(self, *, projects: list[Any] | None = None) -> None:
        if self._refresh_running:
            self._refresh_pending = True
            self._pending_projects = projects
            return
        self._refresh_generation += 1
        generation = self._refresh_generation
        self._refresh_running = True
        if not self._has_snapshot:
            if projects is not None:
                self.projects_metric.set_value(len(projects), "External, private workspaces")
            else:
                self.projects_metric.set_value("…", "Refreshing private projects")
            self.datasets_metric.set_value("…", "Counting private dataset metadata")
            self.checkpoints_metric.set_value("…", "Counting checkpoint metadata")
            self.backup_metric.set_value("…", "Checking independent recovery custody")

        dataset_root = _path_attr(self.manager, "datasets_dir")
        checkpoint_root = _path_attr(self.manager, "checkpoints_dir")
        backup = _path_attr(self.manager, "backup_root")
        self.datasets_metric.setToolTip(str(dataset_root))
        self.checkpoints_metric.setToolTip(str(checkpoint_root))
        self.backup_metric.setToolTip(str(backup))
        rows = (
            ("Public source", _path_attr(self.manager, "source_root"), "Tracked after guard scan"),
            ("Projects", _path_attr(self.manager, "projects_dir"), "Excluded"),
            ("Datasets", _path_attr(self.manager, "datasets_dir"), "Excluded"),
            ("Checkpoints", _path_attr(self.manager, "checkpoints_dir"), "Excluded"),
            ("Logs", _path_attr(self.manager, "logs_dir"), "Excluded"),
            ("Backup", backup, "Never staged"),
        )
        self.location_table.setRowCount(len(rows))
        for row, values in enumerate(rows):
            for column, value in enumerate(values):
                self.location_table.setItem(row, column, QTableWidgetItem(str(value)))

        run_in_background(
            self,
            lambda supplied=projects: self._mission_snapshot(supplied),
            lambda result, current=generation: self._mission_refresh_finished(current, result),
            lambda error, current=generation: self._mission_refresh_failed(current, error),
        )

    @staticmethod
    def _checkpoint_count(directory: Path) -> int:
        """Count the supported root/project checkpoint topology without recursive walking."""

        count = sum(1 for _item in directory.glob("*.json"))
        for child in directory.iterdir():
            if child.is_dir() and not child.is_symlink():
                count += sum(1 for _item in child.glob("*.json"))
        return count

    def _mission_snapshot(self, supplied_projects: list[Any] | None) -> dict[str, Any]:
        projects = _project_items(self.manager) if supplied_projects is None else supplied_projects
        artifact_counts: dict[str, int] = {}
        artifact_errors: set[str] = set()
        for attr in ("datasets_dir", "checkpoints_dir"):
            directory = _path_attr(self.manager, attr)
            try:
                count = (
                    sum(1 for _item in directory.glob("*.dataset.json"))
                    if attr == "datasets_dir"
                    else self._checkpoint_count(directory)
                )
            except OSError:
                count = 0
                artifact_errors.add(attr)
            artifact_counts[attr] = count
        backup = _path_attr(self.manager, "backup_root")
        try:
            backup_ready = backup.exists() and backup.is_dir()
        except OSError:
            backup_ready = False
        try:
            if self._run_registry is None:
                from daedalus.workspace.run_registry import RunRegistry

                self._run_registry = RunRegistry(
                    _path_attr(self.manager, "runs_dir") / "runs.sqlite3"
                )
            run_records = self._run_registry.list_runs(limit=12)
        except Exception:
            run_records = []
        return {
            "project_count": len(projects),
            "artifact_counts": artifact_counts,
            "artifact_errors": artifact_errors,
            "backup_ready": backup_ready,
            "run_records": run_records,
        }

    def _mission_refresh_finished(self, generation: int, result: Any) -> None:
        self._refresh_running = False
        if generation != self._refresh_generation:
            return
        payload = result if isinstance(result, dict) else {}
        project_count = max(0, int(payload.get("project_count") or 0))
        artifact_counts = payload.get("artifact_counts")
        if not isinstance(artifact_counts, dict):
            artifact_counts = {}
        artifact_errors = payload.get("artifact_errors")
        if not isinstance(artifact_errors, set):
            artifact_errors = set()
        dataset_count = max(0, int(artifact_counts.get("datasets_dir") or 0))
        checkpoint_count = max(0, int(artifact_counts.get("checkpoints_dir") or 0))
        backup_ready = bool(payload.get("backup_ready"))
        run_records = payload.get("run_records")
        if not isinstance(run_records, (list, tuple)):
            run_records = []

        self.projects_metric.set_value(project_count, "External, private workspaces")
        for metric, attr, detail in (
            (self.datasets_metric, "datasets_dir", "Checksum-verified private datasets"),
            (self.checkpoints_metric, "checkpoints_dir", "Reproducible model checkpoints"),
        ):
            metric.set_value(
                artifact_counts.get(attr, 0),
                "Private artifact directory unavailable" if attr in artifact_errors else detail,
            )
        self.backup_metric.set_value(
            "Ready" if backup_ready else "Setup",
            "Independent recoverable copy" if backup_ready else "Choose and validate a backup root",
        )
        self._has_snapshot = True
        self.recent_runs.setRowCount(len(run_records))
        for row, record in enumerate(run_records):
            checkpoint = record.checkpoint or "—"
            values = (
                record.status.upper(),
                record.project,
                record.dataset,
                record.updated_utc,
                checkpoint,
            )
            for column, value in enumerate(values):
                self.recent_runs.setItem(row, column, QTableWidgetItem(str(value)))

        completed_runs = sum(record.status == "completed" for record in run_records)
        if not project_count:
            next_step = "Next: define what you want the AI to do in AI Developer Bot."
        elif not dataset_count:
            next_step = "Next: import and analyze examples in Training Lab."
        elif not completed_runs:
            next_step = "Next: validate a model shape and run a held-out training experiment."
        elif not checkpoint_count:
            next_step = "Next: save and inspect a reproducible checkpoint."
        elif not backup_ready:
            next_step = "Next: protect the project with a verified backup."
        else:
            next_step = "Next: evaluate the checkpoint, compare it with a baseline, then iterate."
        self.next_step_label.setText(next_step)
        if self._refresh_pending:
            pending = self._pending_projects
            self._refresh_pending = False
            self._pending_projects = None
            self.refresh(projects=pending)

    def _mission_refresh_failed(self, generation: int, error: str) -> None:
        self._refresh_running = False
        if generation != self._refresh_generation:
            return
        detail = error.strip().splitlines()[-1] if error.strip() else "unknown error"
        for metric in self.metric_tiles:
            metric.set_value("—", "Background refresh failed safely")
        self.next_step_label.setText(f"Workspace summary is temporarily unavailable: {detail}")


class LearningAtlasPage(WorkspacePage):
    """Offline-first browser for the packaged Daedalus learning corpus."""

    def __init__(self, manager, parent=None) -> None:
        super().__init__(
            manager,
            "Learning Atlas",
            "A theory-to-code path for building neural networks from NumPy primitives.",
            "learn",
            (
                (
                    "Learning contract",
                    "Each topic should answer four questions: what is computed, why it is useful, "
                    "what shape flows through it, and how the result can be verified.",
                ),
                (
                    "Suggested route",
                    "Start with tensor shapes, implement a tiny autograd graph, build a dense layer, "
                    "train XOR, then inspect the checkpoint and evaluation path.",
                ),
            ),
            parent,
            scroll_workspace=False,
        )
        self.learning_resource = load_json("learning_paths.json")
        self.glossary_resource = load_json("glossary.json")
        self.error_resource = load_json("error_cards.json")
        self.recipe_resource = load_json("project_recipes.json")
        self.source_resource = load_json("sources.json")
        self.tracks = list(self.learning_resource.get("tracks", ()))
        self.glossary_entries = list(self.glossary_resource.get("entries", ()))
        self.error_cards = list(self.error_resource.get("cards", ()))
        self.recipes = list(self.recipe_resource.get("recipes", ()))
        self.sources = list(self.source_resource.get("sources", ()))

        self.resource_tabs = QTabWidget()
        self.resource_tabs.setAccessibleName("Learning Atlas resource tabs")
        self.resource_tabs.addTab(self._build_paths_tab(), "Learning Paths")
        self.resource_tabs.addTab(self._build_glossary_tab(), "Glossary")
        self.resource_tabs.addTab(self._build_error_tab(), "Error Clinic")
        self.resource_tabs.addTab(self._build_recipe_tab(), "Project Recipes")
        self.resource_tabs.addTab(self._build_sources_tab(), "Official Sources")
        self.workspace_layout.addWidget(self.resource_tabs, 1)

        self._populate_tracks()
        self._filter_glossary("")
        self._filter_errors("")
        self._filter_recipes("")
        self._filter_sources("")

    @staticmethod
    def _search_text(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, default=str).casefold()

    @staticmethod
    def _list_html(items: Iterable[Any]) -> str:
        values = [f"<li>{escape(str(item))}</li>" for item in items]
        return "<ul>" + "".join(values) + "</ul>" if values else "<p>None.</p>"

    @staticmethod
    def _browser(accessible_name: str) -> QTextBrowser:
        browser = QTextBrowser()
        browser.setOpenExternalLinks(False)
        browser.setOpenLinks(False)
        browser.setAccessibleName(accessible_name)
        return browser

    @staticmethod
    def _search_list_panel(
        placeholder: str,
        search_name: str,
        list_name: str,
    ) -> tuple[QWidget, QLineEdit, QListWidget]:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        search = QLineEdit()
        search.setPlaceholderText(placeholder)
        search.setClearButtonEnabled(True)
        search.setAccessibleName(search_name)
        listing = QListWidget()
        listing.setAccessibleName(list_name)
        layout.addWidget(search)
        layout.addWidget(listing, 1)
        return panel, search, listing

    def _build_paths_tab(self) -> QWidget:
        page = QWidget()
        root = QVBoxLayout(page)
        root.setContentsMargins(0, 0, 0, 0)
        policy = QLabel(
            f"{self.learning_resource.get('core_constraint', '')}\n\n"
            f"Completion gate: {self.learning_resource.get('completion_policy', '')}"
        )
        policy.setObjectName("Muted")
        policy.setWordWrap(True)
        policy.setAccessibleName("Learning Atlas completion policy")
        root.addWidget(policy)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        self.track_list = QListWidget()
        self.track_list.setAccessibleName("Learning tracks")
        splitter.addWidget(self.track_list)
        self.module_list = QListWidget()
        self.module_list.setAccessibleName("Modules in selected learning track")
        splitter.addWidget(self.module_list)
        self.topic = self._browser("Selected learning module and checkpoint gates")
        splitter.addWidget(self.topic)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)
        splitter.setStretchFactor(2, 6)
        splitter.setSizes([230, 300, 650])
        root.addWidget(splitter, 1)
        self.track_list.currentRowChanged.connect(self._show_track)
        self.module_list.currentRowChanged.connect(self._show_module)
        return page

    def _populate_tracks(self) -> None:
        self.track_list.clear()
        for track in self.tracks:
            modules = track.get("modules", ())
            title = f"{track.get('title', track.get('id', 'Track'))} · {len(modules)} modules"
            item = QListWidgetItem(semantic_icon("learn", size=18), title)
            item.setData(Qt.ItemDataRole.UserRole, track)
            item.setToolTip(str(track.get("summary", "")))
            self.track_list.addItem(item)
        if self.track_list.count():
            self.track_list.setCurrentRow(0)
        else:
            self.topic.setPlainText("No packaged learning tracks were found.")

    def _show_track(self, row: int) -> None:
        self.module_list.clear()
        item = self.track_list.item(row)
        if item is None:
            self.topic.clear()
            return
        track = item.data(Qt.ItemDataRole.UserRole) or {}
        for module in track.get("modules", ()):
            gate_count = len(module.get("checkpoint", {}).get("checks", ()))
            module_item = QListWidgetItem(
                semantic_icon("architecture", size=17),
                f"{module.get('title', module.get('id', 'Module'))} · {gate_count} gates",
            )
            module_item.setData(Qt.ItemDataRole.UserRole, {"track": track, "module": module})
            module_item.setToolTip(str(module.get("summary", "")))
            self.module_list.addItem(module_item)
        if self.module_list.count():
            self.module_list.setCurrentRow(0)
        else:
            self.topic.setHtml(self._track_html(track))

    def _track_html(self, track: dict[str, Any]) -> str:
        prerequisites = track.get("prerequisite_track_ids", ())
        return (
            f"<h2>{escape(str(track.get('title', 'Learning track')))}</h2>"
            f"<p><b>Level:</b> {escape(str(track.get('level', '')))}</p>"
            f"<p>{escape(str(track.get('summary', '')))}</p>"
            f"<h3>Prerequisite tracks</h3>{self._list_html(prerequisites)}"
            f"<p><b>Capstone recipe:</b> "
            f"{escape(str(track.get('capstone_recipe_id') or 'None'))}</p>"
        )

    def _show_module(self, row: int) -> None:
        item = self.module_list.item(row)
        if item is None:
            return
        payload = item.data(Qt.ItemDataRole.UserRole) or {}
        track = payload.get("track", {})
        module = payload.get("module", {})
        lab = module.get("lab", {})
        checkpoint = module.get("checkpoint", {})
        score = checkpoint.get("required_score", 1.0)
        try:
            score_text = f"{float(score) * 100:.0f}%"
        except (TypeError, ValueError):
            score_text = str(score)
        self.topic.setHtml(
            self._track_html(track)
            + f"<hr><h2>{escape(str(module.get('title', 'Module')))}</h2>"
            + f"<p>{escape(str(module.get('summary', '')))}</p>"
            + f"<p><code>{escape(str(module.get('id', '')))}</code></p>"
            + f"<h3>Objectives</h3>{self._list_html(module.get('objectives', ())) }"
            + f"<h3>Lab · {escape(str(lab.get('title', 'Practice')))}</h3>"
            + self._list_html(lab.get("instructions", ()))
            + f"<p><b>Deliverable:</b> {escape(str(lab.get('deliverable', '')))}</p>"
            + f"<h3>Checkpoint gates · required score {escape(score_text)}</h3>"
            + self._list_html(checkpoint.get("checks", ()))
            + f"<p><b>Glossary:</b> {escape(', '.join(module.get('glossary_ids', ())))}<br>"
            + f"<b>Error clinic:</b> {escape(', '.join(module.get('error_card_ids', ())))}<br>"
            + f"<b>Sources:</b> {escape(', '.join(module.get('source_ids', ())))}</p>"
        )

    def _build_glossary_tab(self) -> QWidget:
        splitter = QSplitter(Qt.Orientation.Horizontal)
        panel, self.glossary_search, self.glossary_list = self._search_list_panel(
            "Search terms, aliases, topics, or explanations…",
            "Search the offline glossary",
            "Offline glossary entries",
        )
        self.glossary_detail = self._browser("Selected glossary definition")
        splitter.addWidget(panel)
        splitter.addWidget(self.glossary_detail)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 5)
        self.glossary_search.textChanged.connect(self._filter_glossary)
        self.glossary_list.currentRowChanged.connect(self._show_glossary)
        return splitter

    def _filter_glossary(self, query: str) -> None:
        needle = query.strip().casefold()
        self.glossary_list.clear()
        for entry in self.glossary_entries:
            if needle and needle not in self._search_text(entry):
                continue
            aliases = ", ".join(entry.get("aliases", ()))
            label = str(entry.get("term", entry.get("id", "Term")))
            if aliases:
                label += f" · {aliases}"
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, entry)
            self.glossary_list.addItem(item)
        if self.glossary_list.count():
            self.glossary_list.setCurrentRow(0)
        else:
            self.glossary_detail.setPlainText("No glossary entries match this search.")

    def _show_glossary(self, row: int) -> None:
        item = self.glossary_list.item(row)
        if item is None:
            return
        entry = item.data(Qt.ItemDataRole.UserRole) or {}
        modes = entry.get("modes", {})
        mode_html = "".join(
            f"<h3>{escape(str(mode).replace('_', ' ').title())}</h3>"
            f"<p>{escape(str(text))}</p>"
            for mode, text in modes.items()
        )
        self.glossary_detail.setHtml(
            f"<h2>{escape(str(entry.get('term', 'Term')))}</h2>"
            f"<p><code>{escape(str(entry.get('id', '')))}</code></p>{mode_html}"
            f"<p><b>Related:</b> {escape(', '.join(entry.get('related_ids', ())))}<br>"
            f"<b>Offline source IDs:</b> {escape(', '.join(entry.get('source_ids', ())))}</p>"
        )

    def _build_error_tab(self) -> QWidget:
        splitter = QSplitter(Qt.Orientation.Horizontal)
        panel, self.error_search, self.error_list = self._search_list_panel(
            "Search an exception, symptom, cause, or safe fix…",
            "Search the offline error clinic",
            "Offline error clinic cards",
        )
        self.error_detail = self._browser("Selected error clinic card")
        splitter.addWidget(panel)
        splitter.addWidget(self.error_detail)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 5)
        self.error_search.textChanged.connect(self._filter_errors)
        self.error_list.currentRowChanged.connect(self._show_error)
        return splitter

    def _filter_errors(self, query: str) -> None:
        needle = query.strip().casefold()
        self.error_list.clear()
        for card in self.error_cards:
            if needle and needle not in self._search_text(card):
                continue
            item = QListWidgetItem(
                semantic_icon("guard", size=17),
                f"{card.get('title', card.get('id', 'Error'))} · {card.get('severity', 'info')}",
            )
            item.setData(Qt.ItemDataRole.UserRole, card)
            self.error_list.addItem(item)
        if self.error_list.count():
            self.error_list.setCurrentRow(0)
        else:
            self.error_detail.setPlainText("No Error Clinic cards match this search.")

    def _show_error(self, row: int) -> None:
        item = self.error_list.item(row)
        if item is None:
            return
        card = item.data(Qt.ItemDataRole.UserRole) or {}
        self.error_detail.setHtml(
            f"<h2>{escape(str(card.get('title', 'Error Clinic')))}</h2>"
            f"<p><b>Severity:</b> {escape(str(card.get('severity', '')))} · "
            f"<b>Level:</b> {escape(str(card.get('level', '')))}</p>"
            f"<p>{escape(str(card.get('plain_cause', '')))}</p>"
            f"<h3>Likely causes</h3>{self._list_html(card.get('likely_causes', ())) }"
            f"<h3>Evidence to collect</h3>{self._list_html(card.get('evidence', ())) }"
            f"<h3>Checks</h3>{self._list_html(card.get('checks', ())) }"
            f"<h3>Safe fixes</h3>{self._list_html(card.get('safe_fixes', ())) }"
            f"<h3>Never actions</h3>{self._list_html(card.get('never_actions', ())) }"
            f"<p><b>Sources:</b> {escape(', '.join(card.get('source_ids', ())))}</p>"
        )

    def _build_recipe_tab(self) -> QWidget:
        splitter = QSplitter(Qt.Orientation.Horizontal)
        panel, self.recipe_search, self.recipe_list = self._search_list_panel(
            "Search projects, deliverables, steps, or gates…",
            "Search offline project recipes",
            "Offline project recipes",
        )
        self.recipe_detail = self._browser("Selected project recipe")
        splitter.addWidget(panel)
        splitter.addWidget(self.recipe_detail)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 5)
        self.recipe_search.textChanged.connect(self._filter_recipes)
        self.recipe_list.currentRowChanged.connect(self._show_recipe)
        return splitter

    def _filter_recipes(self, query: str) -> None:
        needle = query.strip().casefold()
        self.recipe_list.clear()
        for recipe in self.recipes:
            if needle and needle not in self._search_text(recipe):
                continue
            item = QListWidgetItem(
                semantic_icon("workshop", size=17),
                f"{recipe.get('title', recipe.get('id', 'Recipe'))} · "
                f"{recipe.get('estimated_hours', '?')} hours",
            )
            item.setData(Qt.ItemDataRole.UserRole, recipe)
            self.recipe_list.addItem(item)
        if self.recipe_list.count():
            self.recipe_list.setCurrentRow(0)
        else:
            self.recipe_detail.setPlainText("No project recipes match this search.")

    def _show_recipe(self, row: int) -> None:
        item = self.recipe_list.item(row)
        if item is None:
            return
        recipe = item.data(Qt.ItemDataRole.UserRole) or {}
        self.recipe_detail.setHtml(
            f"<h2>{escape(str(recipe.get('title', 'Project recipe')))}</h2>"
            f"<p><b>Level:</b> {escape(str(recipe.get('level', '')))} · "
            f"<b>Estimate:</b> {escape(str(recipe.get('estimated_hours', '?')))} hours</p>"
            f"<p>{escape(str(recipe.get('summary', '')))}</p>"
            f"<p><b>Deliverable:</b> {escape(str(recipe.get('deliverable', '')))}</p>"
            f"<h3>Steps</h3>{self._list_html(recipe.get('steps', ())) }"
            f"<h3>Completion gates</h3>{self._list_html(recipe.get('gates', ())) }"
            f"<h3>Extensions</h3>{self._list_html(recipe.get('extensions', ())) }"
            f"<p><b>Sources:</b> {escape(', '.join(recipe.get('source_ids', ())))}</p>"
        )

    def _build_sources_tab(self) -> QWidget:
        page = QWidget()
        root = QVBoxLayout(page)
        root.setContentsMargins(0, 0, 0, 0)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        panel, self.source_search, self.source_list = self._search_list_panel(
            "Search title, publisher, topic, or offline summary…",
            "Search official sources",
            "Official source list",
        )
        self.source_detail = self._browser("Selected source offline summary")
        splitter.addWidget(panel)
        splitter.addWidget(self.source_detail)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 5)
        root.addWidget(splitter, 1)
        action_row = QHBoxLayout()
        self.source_status = QLabel(
            "Offline summaries are always available. Opening a source leaves Daedalus."
        )
        self.source_status.setObjectName("Muted")
        self.source_status.setWordWrap(True)
        action_row.addWidget(self.source_status, 1)
        self.open_source_button = QPushButton("Open selected official source")
        self.open_source_button.setAccessibleName(
            "Open the selected official source in the default browser"
        )
        self.open_source_button.setEnabled(False)
        self.open_source_button.clicked.connect(self._open_selected_source)
        action_row.addWidget(self.open_source_button)
        root.addLayout(action_row)
        self.source_search.textChanged.connect(self._filter_sources)
        self.source_list.currentRowChanged.connect(self._show_source)
        return page

    def _filter_sources(self, query: str) -> None:
        needle = query.strip().casefold()
        self.source_list.clear()
        for source in self.sources:
            if needle and needle not in self._search_text(source):
                continue
            item = QListWidgetItem(
                f"{source.get('title', source.get('id', 'Source'))} · "
                f"{source.get('publisher', 'Publisher')}",
            )
            item.setData(Qt.ItemDataRole.UserRole, source)
            self.source_list.addItem(item)
        if self.source_list.count():
            self.source_list.setCurrentRow(0)
        else:
            self._selected_source = None
            self.open_source_button.setEnabled(False)
            self.source_detail.setPlainText("No official sources match this search.")

    def _show_source(self, row: int) -> None:
        item = self.source_list.item(row)
        if item is None:
            self._selected_source = None
            self.open_source_button.setEnabled(False)
            return
        source = item.data(Qt.ItemDataRole.UserRole) or {}
        self._selected_source = source
        self.open_source_button.setEnabled(bool(source.get("url")))
        self.source_detail.setHtml(
            f"<h2>{escape(str(source.get('title', 'Official source')))}</h2>"
            f"<p><b>Publisher:</b> {escape(str(source.get('publisher', '')))}<br>"
            f"<b>Kind:</b> {escape(str(source.get('kind', '')))}<br>"
            f"<b>Levels:</b> {escape(', '.join(source.get('levels', ())))}<br>"
            f"<b>Topics:</b> {escape(', '.join(source.get('topics', ())))}</p>"
            f"<h3>Offline summary</h3><p>{escape(str(source.get('offline_summary', '')))}</p>"
            f"<p><b>URL (inactive until Open is clicked):</b><br>"
            f"{escape(str(source.get('url', '')))}</p>"
        )

    def _open_selected_source(self) -> bool:
        source = getattr(self, "_selected_source", None) or {}
        url = QUrl(str(source.get("url", "")))
        if not url.isValid() or url.scheme().casefold() not in {"https", "http"}:
            self.source_status.setObjectName("Warning")
            repolish(self.source_status)
            self.source_status.setText("The selected source URL is invalid and was not opened.")
            return False
        opened = QDesktopServices.openUrl(url)
        self.source_status.setObjectName("Success" if opened else "Warning")
        repolish(self.source_status)
        self.source_status.setText(
            "The selected official source was handed to the default browser."
            if opened
            else "The operating system could not open the selected source."
        )
        return bool(opened)


def _parse_sizes(text: str) -> list[int]:
    tokens = [part.strip() for part in re.split(r"[,xX→>\s]+", text) if part.strip()]
    if len(tokens) < 2:
        raise ValueError("Enter at least an input and output size, for example 784, 128, 10.")
    sizes = [int(token) for token in tokens]
    if any(value <= 0 for value in sizes):
        raise ValueError("Every layer width must be a positive integer.")
    if any(value > 10_000_000 for value in sizes):
        raise ValueError("A layer width is outside the calculator's 10,000,000-unit safety bound.")
    return sizes


def _dense_parameters(sizes: list[int]) -> tuple[list[int], int]:
    per_layer = [(left * right) + right for left, right in zip(sizes, sizes[1:])]
    return per_layer, sum(per_layer)


class ArchitectureBuilderPage(WorkspacePage):
    def __init__(self, manager, parent=None) -> None:
        super().__init__(
            manager,
            "Architecture Builder",
            "Assemble a dense network and validate every shape before allocating model memory.",
            "architecture",
            (
                (
                    "Dense-layer rule",
                    "A dense layer maps [batch, input features] through W[input, output] and "
                    "b[output], producing [batch, output]. Parameter count is input×output+output.",
                ),
                (
                    "Shape-first engineering",
                    "Validate adjacent dimensions before training. Shape errors are cheaper to solve "
                    "in a table than inside a long experiment.",
                ),
            ),
            parent,
        )
        controls = Card("Network definition", "Use commas, spaces, x, or arrows between layer widths.", accent=True)
        row = QHBoxLayout()
        row.addWidget(QLabel("Layer widths:"))
        self.sizes = QLineEdit("784, 128, 64, 10")
        self.sizes.setAccessibleName("Architecture layer widths")
        self.sizes.returnPressed.connect(self.validate_architecture)
        row.addWidget(self.sizes, 1)
        validate = QPushButton("Validate architecture")
        validate.setObjectName("Primary")
        validate.setAccessibleName("Validate architecture shapes and parameters")
        validate.clicked.connect(self.validate_architecture)
        row.addWidget(validate)
        controls.body.addLayout(row)
        presets = QHBoxLayout()
        presets.addWidget(QLabel("Presets:"))
        for label, value in (
            ("XOR", "2, 4, 1"),
            ("MNIST MLP", "784, 128, 64, 10"),
            ("Tiny language block", "256, 512, 256"),
        ):
            button = QPushButton(label)
            button.setAccessibleName(f"Load {label} architecture preset")
            button.clicked.connect(lambda _checked=False, text=value: self._load_preset(text))
            presets.addWidget(button)
        presets.addStretch(1)
        controls.body.addLayout(presets)
        self.workspace_layout.addWidget(controls)

        result = Card("Validated shape ledger", "Bias parameters are included in every dense layer.")
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["Layer", "Input shape", "Weights", "Output shape", "Parameters"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        result.add_widget(self.table, 1)
        self.summary = QLabel()
        self.summary.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        result.add_widget(self.summary)
        self.workspace_layout.addWidget(result, 1)
        self._model_3d_viewer: Any | None = None
        self._pending_architecture = (784, 128, 64, 10)
        self._reduced_3d_motion = reduced_motion()
        self._model_3d_placeholder = _deferred_tab(
            "3D Model",
            "The interactive renderer loads when this tab is opened, keeping the design ledger "
            "immediately responsive on systems that do not need the 3D view.",
        )
        self.tabs.insertTab(self.tabs.count() - 1, self._model_3d_placeholder, "3D Model")
        self.tabs.setTabToolTip(
            self.tabs.indexOf(self._model_3d_placeholder),
            "Rotate, zoom, and inspect a capability-adaptive 3D view of this architecture",
        )
        self.tabs.currentChanged.connect(self._materialize_architecture_tab)
        self.validate_architecture()

    @property
    def model_3d_viewer(self):
        """Materialize the optional renderer on explicit tab or API access."""

        return self._ensure_model_3d_viewer()

    def _materialize_architecture_tab(self, index: int) -> None:
        if self.tabs.widget(index) is self._model_3d_placeholder:
            self._ensure_model_3d_viewer()

    def _ensure_model_3d_viewer(self):
        viewer = self._model_3d_viewer
        if viewer is not None:
            return viewer
        from daedalus.gui.model_3d import Model3DViewer

        viewer = Model3DViewer(self)
        viewer.set_architecture(self._pending_architecture)
        viewer.set_reduced_motion(self._reduced_3d_motion)
        self._model_3d_viewer = viewer
        _replace_deferred_tab(
            self.tabs,
            self._model_3d_placeholder,
            viewer,
            "3D Model",
            "Rotate, zoom, and inspect a capability-adaptive 3D view of this architecture",
        )
        return viewer

    def set_reduced_motion(self, reduced: bool) -> None:
        self._reduced_3d_motion = bool(reduced)
        if self._model_3d_viewer is not None:
            self._model_3d_viewer.set_reduced_motion(reduced)

    def _load_preset(self, text: str) -> None:
        self.sizes.setText(text)
        self.validate_architecture()

    def validate_architecture(self) -> bool:
        try:
            sizes = _parse_sizes(self.sizes.text())
            counts, total = _dense_parameters(sizes)
        except (TypeError, ValueError) as exc:
            self.summary.setObjectName("Danger")
            repolish(self.summary)
            self.summary.setText(f"Validation failed: {exc}")
            self.table.setRowCount(0)
            return False
        self.table.setRowCount(len(counts))
        for row, (left, right, params) in enumerate(zip(sizes, sizes[1:], counts)):
            values = (
                f"Dense {row + 1}",
                f"[batch, {left:,}]",
                f"[{left:,}, {right:,}] + [{right:,}]",
                f"[batch, {right:,}]",
                f"{params:,}",
            )
            for column, value in enumerate(values):
                self.table.setItem(row, column, QTableWidgetItem(value))
        self.summary.setObjectName("Success")
        repolish(self.summary)
        self.summary.setText(
            f"Valid feed-forward chain · {len(counts)} dense layers · {total:,} trainable parameters"
        )
        self._pending_architecture = sizes
        if self._model_3d_viewer is not None:
            self._model_3d_viewer.set_architecture(sizes)
        return True


class CalculatorLabPage(WorkspacePage):
    status_changed = Signal(str, str)
    open_in_workshop_requested = Signal(object)

    FORMULA = '''# Transparent dense-network estimate
parameters = sum((n_in * n_out) + n_out for n_in, n_out in layers)
parameter_bytes = parameters * bytes_per_value
gradient_bytes = parameter_bytes
optimizer_bytes = parameter_bytes * optimizer_state_copies
activation_bytes = batch_size * sum([input_features, *layer_outputs]) * bytes_per_value
estimated_training_bytes = (
    parameter_bytes + gradient_bytes + optimizer_bytes + activation_bytes
)
'''

    def __init__(self, manager, parent=None) -> None:
        from daedalus.gui.editor import CodeEditorPanel

        super().__init__(
            manager,
            "Calculator Lab",
            "Estimate parameters, tensor shapes, and training-state memory with visible formulas.",
            "calculator",
            (
                (
                    "What this estimate includes",
                    "Weights, biases, one gradient buffer, optimizer-state copies, and retained dense "
                    "activations. Framework workspaces, temporary kernels, data batches, and OS overhead are not included.",
                ),
                (
                    "Precision",
                    "FP64 uses 8 bytes per scalar, FP32 uses 4, and FP16/BF16 use 2. Some training "
                    "systems keep master FP32 weights even when activations use lower precision.",
                ),
            ),
            parent,
            scroll_workspace=False,
        )
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.setOpaqueResize(False)

        inputs = Card("Inputs", "Dense feed-forward estimate", accent=True)
        form = QFormLayout()
        self.layer_sizes = QLineEdit("784, 128, 64, 10")
        self.layer_sizes.setAccessibleName("Calculator layer widths")
        self.batch = QSpinBox()
        self.batch.setRange(1, 1_000_000)
        self.batch.setValue(64)
        self.batch.setAccessibleName("Training batch size")
        self.precision = QComboBox()
        self.precision.addItem("FP64 · 8 bytes", 8)
        self.precision.addItem("FP32 · 4 bytes", 4)
        self.precision.addItem("FP16 / BF16 · 2 bytes", 2)
        self.precision.setCurrentIndex(1)
        self.precision.setAccessibleName("Scalar precision")
        self.optimizer = QComboBox()
        self.optimizer.addItem("SGD · no state copy", 0)
        self.optimizer.addItem("Momentum · 1 state copy", 1)
        self.optimizer.addItem("Adam · 2 state copies", 2)
        self.optimizer.setCurrentIndex(2)
        self.optimizer.setAccessibleName("Optimizer state model")
        form.addRow("Layer widths", self.layer_sizes)
        form.addRow("Batch size", self.batch)
        form.addRow("Precision", self.precision)
        form.addRow("Optimizer", self.optimizer)
        inputs.body.addLayout(form)
        calculate = QPushButton("Calculate")
        calculate.setObjectName("Primary")
        calculate.setAccessibleName("Calculate architecture parameters and memory")
        calculate.clicked.connect(self.calculate)
        inputs.add_widget(calculate)
        inputs.body.addStretch(1)
        splitter.addWidget(inputs)

        formula = Card("Formula editor", "Read-only implementation used for the visible estimate.")
        self.formula_editor = CodeEditorPanel()
        self.formula_editor.setPlainText(self.FORMULA)
        self.formula_editor.editor.setReadOnly(True)
        self.formula_editor.editor.setAccessibleName("Calculator formula source")
        formula.add_widget(self.formula_editor, 1)
        splitter.addWidget(formula)

        result = Card("Results", "Estimates update only when Calculate is activated.")
        self.results = QPlainTextEdit()
        self.results.setReadOnly(True)
        self.results.setAccessibleName("Calculator results")
        result.add_widget(self.results, 1)
        splitter.addWidget(result)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)
        splitter.setStretchFactor(2, 2)
        splitter.setSizes([300, 430, 330])
        self.tool_splitter = splitter
        self._calculator_panes = (inputs, formula, result)
        self._compact_tool_tabs: QTabWidget | None = None
        self.workspace_layout.addWidget(splitter, 1)
        self._weight_lab: Any | None = None
        self._advanced_panel: Any | None = None
        self._pending_weight_project: str | Path | None = None
        self._reduced_weight_motion = reduced_motion()
        self._weight_lab_placeholder = _deferred_tab(
            "Weight Lab",
            "Six bounded synthesis and experimentation tools load when this tab is opened. "
            "Your calculator inputs remain available while the toolset starts.",
        )
        self._advanced_placeholder = _deferred_tab(
            "Advanced calculators",
            "Convolution, Transformer, quantization, and training-plan calculators load only "
            "when this tab is opened.",
        )
        self.tabs.insertTab(1, self._weight_lab_placeholder, "Weight Lab")
        self.tabs.setTabToolTip(
            1,
            "Six bounded weight synthesis, logic, recurrence, constraint, ELM, and active-learning tools",
        )
        self.tabs.insertTab(2, self._advanced_placeholder, "Advanced")
        self.tabs.setTabToolTip(
            2,
            "Convolution, Transformer, quantization, and training-plan calculators",
        )
        self.tabs.currentChanged.connect(self._materialize_calculator_tab)
        self.calculate()

    @property
    def weight_lab(self):
        """Materialize Weight Lab only when the user or API requests it."""

        return self._ensure_weight_lab()

    @property
    def advanced_panel(self):
        return self._ensure_advanced_panel()

    def _materialize_calculator_tab(self, index: int) -> None:
        widget = self.tabs.widget(index)
        if widget is self._weight_lab_placeholder:
            self._ensure_weight_lab()
        elif widget is self._advanced_placeholder:
            self._ensure_advanced_panel()

    def _ensure_weight_lab(self):
        panel = self._weight_lab
        if panel is not None:
            return panel
        from daedalus.gui.weight_lab_panel import WeightLabPanel

        panel = WeightLabPanel(self.manager, self)
        panel.set_reduced_motion(self._reduced_weight_motion)
        if self._pending_weight_project is not None:
            panel.set_project(self._pending_weight_project)
        panel.status_changed.connect(self.status_changed.emit)
        panel.open_in_workshop_requested.connect(self.open_in_workshop_requested.emit)
        self._weight_lab = panel
        _replace_deferred_tab(
            self.tabs,
            self._weight_lab_placeholder,
            panel,
            "Weight Lab",
            "Six bounded weight synthesis, logic, recurrence, constraint, ELM, and active-learning tools",
        )
        return panel

    def _ensure_advanced_panel(self):
        panel = self._advanced_panel
        if panel is not None:
            return panel
        from daedalus.gui.advanced_calculator_panel import AdvancedCalculatorPanel

        panel = AdvancedCalculatorPanel(self)
        self._advanced_panel = panel
        _replace_deferred_tab(
            self.tabs,
            self._advanced_placeholder,
            panel,
            "Advanced",
            "Convolution, Transformer, quantization, and training-plan calculators",
        )
        return panel

    def set_compact_layout(self, compact: bool) -> None:
        compact = bool(compact)
        if compact and self._compact_tool_tabs is None:
            tabs = QTabWidget(self.workspace_widget)
            tabs.setAccessibleName("Compact calculator work areas")
            scrolls: list[QScrollArea] = []
            for pane, label, minimum_height in zip(
                self._calculator_panes,
                ("Inputs", "Formula", "Results"),
                (360, 380, 380),
            ):
                pane.setMinimumHeight(minimum_height)
                area = _scroll(pane)
                area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
                tabs.addTab(area, label)
                scrolls.append(area)
            self.workspace_layout.removeWidget(self.tool_splitter)
            self.tool_splitter.hide()
            self.workspace_layout.addWidget(tabs, 1)
            self._compact_tool_tabs = tabs
            self._compact_tool_scrolls = scrolls
            return
        if not compact and self._compact_tool_tabs is not None:
            tabs = self._compact_tool_tabs
            self.workspace_layout.removeWidget(tabs)
            for area, pane in zip(self._compact_tool_scrolls, self._calculator_panes):
                area.takeWidget()
                pane.setMinimumHeight(0)
                self.tool_splitter.addWidget(pane)
            self.tool_splitter.setStretchFactor(0, 2)
            self.tool_splitter.setStretchFactor(1, 3)
            self.tool_splitter.setStretchFactor(2, 2)
            self.tool_splitter.setSizes([300, 430, 330])
            self.workspace_layout.addWidget(self.tool_splitter, 1)
            self.tool_splitter.show()
            tabs.deleteLater()
            self._compact_tool_tabs = None
            self._compact_tool_scrolls = []

    def set_project(self, project: str | Path | None) -> bool:
        self._pending_weight_project = project
        if self._weight_lab is not None:
            return bool(self._weight_lab.set_project(project))
        return True

    def set_reduced_motion(self, reduced: bool) -> None:
        self._reduced_weight_motion = bool(reduced)
        if self._weight_lab is not None:
            self._weight_lab.set_reduced_motion(reduced)

    def calculate(self) -> dict[str, int] | None:
        try:
            sizes = _parse_sizes(self.layer_sizes.text())
            _per_layer, parameters = _dense_parameters(sizes)
            scalar_bytes = int(self.precision.currentData())
            states = int(self.optimizer.currentData())
            # A what-if calculator must not allocate the weights it is trying
            # to estimate. These shape-only formulas match the engine ledger
            # while keeping both memory and runtime constant with model size.
            parameter_bytes = parameters * scalar_bytes
            gradient_bytes = parameter_bytes
            optimizer_bytes = parameter_bytes * states
            activation_scalars = self.batch.value() * sum(sizes)
            activation_bytes = activation_scalars * scalar_bytes
            training_bytes = parameter_bytes + gradient_bytes + optimizer_bytes + activation_bytes
            used_engine = False
        except (TypeError, ValueError) as exc:
            self.results.setPlainText(f"Input error\n===========\n{exc}")
            return None
        report = {
            "parameters": parameters,
            "parameter_bytes": parameter_bytes,
            "gradient_bytes": gradient_bytes,
            "optimizer_bytes": optimizer_bytes,
            "activation_bytes": activation_bytes,
            "training_bytes": training_bytes,
            "used_engine": int(used_engine),
        }
        self.results.setPlainText(
            "ARCHITECTURE\n"
            "============\n"
            f"Chain: {' → '.join(f'{value:,}' for value in sizes)}\n"
            f"Dense layers: {len(sizes) - 1}\n"
            f"Trainable parameters: {parameters:,}\n\n"
            "MEMORY LEDGER\n"
            "=============\n"
            f"Weights + biases: {human_bytes(parameter_bytes)}\n"
            f"Gradients: {human_bytes(gradient_bytes)}\n"
            f"Optimizer state: {human_bytes(optimizer_bytes)}\n"
            f"Retained activations: {human_bytes(activation_bytes)}\n"
            f"Estimated training state: {human_bytes(training_bytes)}\n\n"
            f"Calculator path: {'Daedalus engine' if used_engine else 'bounded arithmetic-only'}\n"
            "Boundary: excludes temporary kernels, dataset batches, Python/NumPy overhead, "
            "allocator fragmentation, and device-specific workspaces."
        )
        return report


class TrainingLabPage(WorkspacePage):
    def __init__(self, manager, parent=None) -> None:
        super().__init__(
            manager,
            "Data & Training Lab",
            "Inspect examples, train with guarded holdouts, and keep every metric and artifact visible.",
            "training",
            (
                (
                    "What this stage builds",
                    "Training Lab is the local numeric-tabular path: import examples, choose the "
                    "target, inspect quality, reserve validation and final-test rows, fit a small "
                    "from-scratch network, and save a reproducible checkpoint. It is not yet an "
                    "LLM, image, audio, or foundation-model fine-tuning system.",
                ),
                (
                    "Data readiness",
                    "Rows are examples, feature columns are information available at prediction "
                    "time, and the target column is what the model should learn. Analysis flags "
                    "constant inputs, duplicate/conflicting examples, weak class coverage, scale "
                    "differences, and datasets too small for a trustworthy split.",
                ),
                (
                    "Holdout discipline",
                    "Training rows update weights. Validation rows guide early stopping. Final-test "
                    "rows are evaluated only after fitting. Feature means and scales are learned "
                    "from training rows alone so information cannot leak backward from held-out data.",
                ),
                (
                    "Determinism",
                    "A fixed seed and identical data/order should reproduce a teaching run within "
                    "the numerical limits of the platform. Record the seed with every checkpoint.",
                ),
                (
                    "Gradient health",
                    "Track loss together with gradient norm and finite values. Falling loss alone "
                    "does not prove that every layer is learning or that the model will generalize.",
                ),
                (
                    "Saved training contract",
                    "A useful checkpoint needs more than weights. Daedalus records the data hash, "
                    "split identity, label mapping, train-only preprocessing, complete layer shapes, "
                    "optimizer settings, seed, stop reason, and held-out metrics alongside the arrays.",
                ),
            ),
            parent,
        )
        config = Card(
            "Experiment configuration",
            "Choose data and a bounded run. Daedalus keeps validation and test rows out of optimization.",
            accent=True,
        )
        form = QGridLayout()
        self.project = QComboBox()
        self.project.setAccessibleName("Training project")
        self.task = QComboBox()
        self.task.addItems(
            [
                "Auto-detect data",
                "Classification / built-in XOR",
                "Regression / built-in regression",
            ]
        )
        self.task.setAccessibleName("Training objective")
        self.dataset = QComboBox()
        self.dataset.setAccessibleName("Training dataset")
        self.dataset.addItem("Built-in teaching data", None)
        self.dataset.currentIndexChanged.connect(self._show_dataset_metadata)
        self.epochs = QSpinBox()
        self.epochs.setRange(1, 5_000)
        self.epochs.setValue(500)
        self.epochs.setAccessibleName("Training epochs")
        self.seed = QSpinBox()
        self.seed.setRange(0, 2_147_483_647)
        self.seed.setValue(47)
        self.seed.setAccessibleName("Random seed")
        self.batch_size = QSpinBox()
        self.batch_size.setRange(1, 1_000_000)
        self.batch_size.setValue(32)
        self.batch_size.setAccessibleName("Training batch size")
        self.learning_rate = QDoubleSpinBox()
        self.learning_rate.setRange(0.000001, 1.0)
        self.learning_rate.setDecimals(6)
        self.learning_rate.setSingleStep(0.001)
        self.learning_rate.setValue(0.03)
        self.learning_rate.setAccessibleName("Optimizer learning rate")
        self.validation_percent = QSpinBox()
        self.validation_percent.setRange(5, 40)
        self.validation_percent.setValue(20)
        self.validation_percent.setSuffix(" %")
        self.validation_percent.setAccessibleName("Validation split percentage")
        self.test_percent = QSpinBox()
        self.test_percent.setRange(5, 40)
        self.test_percent.setValue(10)
        self.test_percent.setSuffix(" %")
        self.test_percent.setAccessibleName("Final test split percentage")
        self.standardize = QCheckBox("Fit feature standardization on training rows only")
        self.standardize.setChecked(True)
        self.standardize.setAccessibleName("Leakage-safe feature standardization")
        self.early_stopping = QSpinBox()
        self.early_stopping.setRange(0, 5_000)
        self.early_stopping.setValue(25)
        self.early_stopping.setSpecialValueText("Off")
        self.early_stopping.setAccessibleName("Early stopping patience")
        form.addWidget(QLabel("Project"), 0, 0)
        form.addWidget(self.project, 0, 1)
        form.addWidget(QLabel("Task"), 0, 2)
        form.addWidget(self.task, 0, 3)
        form.addWidget(QLabel("Epochs"), 1, 0)
        form.addWidget(self.epochs, 1, 1)
        form.addWidget(QLabel("Seed"), 1, 2)
        form.addWidget(self.seed, 1, 3)
        form.addWidget(QLabel("Batch size"), 2, 0)
        form.addWidget(self.batch_size, 2, 1)
        form.addWidget(QLabel("Learning rate"), 2, 2)
        form.addWidget(self.learning_rate, 2, 3)
        form.addWidget(QLabel("Validation"), 3, 0)
        form.addWidget(self.validation_percent, 3, 1)
        form.addWidget(QLabel("Final test"), 3, 2)
        form.addWidget(self.test_percent, 3, 3)
        form.addWidget(QLabel("Dataset"), 4, 0)
        form.addWidget(self.dataset, 4, 1, 1, 3)
        form.addWidget(self.standardize, 5, 0, 1, 2)
        form.addWidget(QLabel("Early-stop patience"), 5, 2)
        form.addWidget(self.early_stopping, 5, 3)
        form.setColumnStretch(1, 1)
        form.setColumnStretch(3, 1)
        config.body.addLayout(form)
        row = QHBoxLayout()
        refresh = QPushButton("Refresh projects")
        refresh.setAccessibleName("Refresh training project list")
        refresh.clicked.connect(self.refresh_projects)
        self.run_button = QPushButton("Train with held-out evaluation")
        self.run_button.setObjectName("Primary")
        self.run_button.setIcon(semantic_icon("training", size=18))
        self.run_button.setAccessibleName("Run bounded validation-aware training")
        self.run_button.clicked.connect(self.start_training)
        row.addWidget(refresh)
        row.addWidget(self.run_button)
        row.addStretch(1)
        config.body.addLayout(row)
        self.workspace_layout.addWidget(config)

        csv_card = Card(
            "Custom numeric CSV",
            "Import copies a validated CSV into the private dataset workspace; source files are never executed.",
        )
        self.csv_path = PathField(
            "CSV source",
            "",
            manager=manager,
            git_excluded=True,
            mode="file",
            file_filter="CSV datasets (*.csv);;All files (*)",
        )
        csv_card.add_widget(self.csv_path)
        csv_options = QGridLayout()
        self.csv_name = QLineEdit()
        self.csv_name.setPlaceholderText("Optional private dataset name")
        self.csv_name.setAccessibleName("Imported dataset name")
        self.csv_target = QLineEdit()
        self.csv_target.setPlaceholderText("Defaults to the final CSV column")
        self.csv_target.setAccessibleName("CSV target column")
        self.csv_delimiter = QLineEdit(",")
        self.csv_delimiter.setMaxLength(1)
        self.csv_delimiter.setMaximumWidth(70)
        self.csv_delimiter.setAccessibleName("CSV delimiter")
        csv_options.addWidget(QLabel("Dataset name"), 0, 0)
        csv_options.addWidget(self.csv_name, 0, 1)
        csv_options.addWidget(QLabel("Target column"), 0, 2)
        csv_options.addWidget(self.csv_target, 0, 3)
        csv_options.addWidget(QLabel("Delimiter"), 0, 4)
        csv_options.addWidget(self.csv_delimiter, 0, 5)
        csv_options.setColumnStretch(1, 1)
        csv_options.setColumnStretch(3, 1)
        csv_card.body.addLayout(csv_options)
        csv_actions = QHBoxLayout()
        self.import_csv_button = QPushButton("Import CSV")
        self.import_csv_button.setAccessibleName("Import the selected numeric CSV")
        self.import_csv_button.clicked.connect(self.start_csv_import)
        self.analyze_data_button = QPushButton("Analyze selected data")
        self.analyze_data_button.setAccessibleName("Analyze selected training data")
        self.analyze_data_button.clicked.connect(self.start_dataset_analysis)
        refresh_datasets = QPushButton("Refresh datasets")
        refresh_datasets.setAccessibleName("Refresh imported datasets")
        refresh_datasets.clicked.connect(self.refresh_datasets)
        csv_actions.addWidget(self.import_csv_button)
        csv_actions.addWidget(self.analyze_data_button)
        csv_actions.addWidget(refresh_datasets)
        csv_actions.addStretch(1)
        csv_card.body.addLayout(csv_actions)
        self.dataset_metadata = QPlainTextEdit()
        self.dataset_metadata.setReadOnly(True)
        self.dataset_metadata.setMaximumHeight(240)
        self.dataset_metadata.setAccessibleName("Imported dataset metadata")
        csv_card.add_widget(self.dataset_metadata)
        self.workspace_layout.addWidget(csv_card)

        output = Card("Run transcript", "Training executes away from the GUI thread when an engine is available.")
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setAccessibleName("Training activity")
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setAccessibleName("Training transcript")
        self.log.setPlainText(
            "Ready. If the from-scratch training engine is not installed yet, this page "
            "will report the missing capability without blocking the rest of the suite."
        )
        output.add_widget(self.progress)
        output.add_widget(self.log, 1)
        self.workspace_layout.addWidget(output, 1)
        self.refresh_projects()
        self.refresh_datasets()

    def refresh_projects(self) -> None:
        selected = self.project.currentText()
        self.project.clear()
        for project in _project_items(self.manager):
            self.project.addItem(_project_name(project), project)
        if not self.project.count():
            self.project.addItem("No project selected", None)
        index = self.project.findText(selected)
        if index >= 0:
            self.project.setCurrentIndex(index)

    @staticmethod
    def _metadata_text(metadata: Any) -> str:
        if metadata is None:
            return (
                "Built-in teaching data is selected. XOR classification or the small "
                "regression fixture will be generated deterministically from the run seed."
            )
        value = asdict(metadata) if is_dataclass(metadata) else metadata
        return _pretty(value)

    def _dataset_service(self):
        from daedalus.workspace.datasets import DatasetService

        return DatasetService(self.manager)

    def import_csv_path(
        self,
        source: str | Path,
        *,
        name: str = "",
        target_column: str = "",
        delimiter: str = ",",
    ) -> Any:
        """Import a CSV without invoking a native dialog; useful to UI and tests."""

        return self._dataset_service().import_csv(
            source,
            name=name.strip() or None,
            target_column=target_column.strip() or None,
            delimiter=delimiter,
        )

    def load_dataset(self, name: str) -> tuple[Any, Any, Any]:
        """Load a previously imported dataset and verify its stored checksum."""

        return self._dataset_service().load(name)

    def refresh_datasets(self, select_name: str | None = None) -> None:
        selected = select_name or self.dataset.currentData()
        self.dataset.blockSignals(True)
        self.dataset.clear()
        self.dataset.addItem("Built-in teaching data", None)
        try:
            datasets = self._dataset_service().list_datasets()
        except Exception as exc:
            datasets = []
            self.dataset_metadata.setPlainText(
                "Imported datasets are unavailable in this environment.\n" + str(exc)
            )
        for metadata in datasets:
            label = f"{metadata.name} · {metadata.rows:,} rows · {len(metadata.feature_columns)} features"
            self.dataset.addItem(label, metadata.name)
            index = self.dataset.count() - 1
            self.dataset.setItemData(index, metadata, Qt.ItemDataRole.UserRole + 1)
        index = self.dataset.findData(selected) if selected else 0
        self.dataset.setCurrentIndex(max(0, index))
        self.dataset.blockSignals(False)
        self._show_dataset_metadata(self.dataset.currentIndex())

    def _show_dataset_metadata(self, index: int) -> None:
        metadata = self.dataset.itemData(index, Qt.ItemDataRole.UserRole + 1)
        self.dataset_metadata.setPlainText(self._metadata_text(metadata))

    def start_csv_import(self) -> None:
        if not self.import_csv_button.isEnabled():
            return
        source = self.csv_path.path
        delimiter = self.csv_delimiter.text()
        if not self.csv_path.is_valid() or source.suffix.casefold() != ".csv":
            self.dataset_metadata.setPlainText(
                "CSV import has not started. Select an existing regular .csv file."
            )
            return
        if len(delimiter) != 1:
            self.dataset_metadata.setPlainText(
                "CSV import has not started. The delimiter must be exactly one character."
            )
            return
        pending = {
            "source": source,
            "name": self.csv_name.text(),
            "target_column": self.csv_target.text(),
            "delimiter": delimiter,
        }
        self.import_csv_button.setEnabled(False)
        self.dataset_metadata.setPlainText("Validating and importing the private CSV…")
        run_in_background(
            self,
            lambda: self.import_csv_path(**pending),
            self._csv_import_finished,
            self._csv_import_failed,
        )

    def _csv_import_finished(self, metadata: Any) -> None:
        self.import_csv_button.setEnabled(True)
        self.refresh_datasets(str(getattr(metadata, "name", "")))
        self.dataset_metadata.setPlainText(self._metadata_text(metadata))
        self.log.appendPlainText(
            f"\nImported private dataset {getattr(metadata, 'name', 'dataset')!r}."
        )

    def _csv_import_failed(self, error: str) -> None:
        self.import_csv_button.setEnabled(True)
        summary = error.strip().splitlines()[-1] if error.strip() else "Unknown import error"
        self.dataset_metadata.setPlainText("CSV import failed safely:\n" + summary)

    def _selected_task(self) -> str:
        tasks = ("auto", "classification", "regression")
        return tasks[max(0, min(self.task.currentIndex(), len(tasks) - 1))]

    def analyze_dataset(
        self,
        name: str | None,
        *,
        task: str = "auto",
        seed: int = 47,
    ) -> Any:
        """Assess registered or built-in data without starting a training run."""

        from daedalus.services.training import TrainingService

        return TrainingService(self.manager).analyze(dataset=name, task=task, seed=seed)

    def _dataset_analysis_call(self) -> dict[str, Any]:
        pending = getattr(self, "_pending_analysis", None) or {
            "dataset": self.dataset.currentData(),
            "task": self._selected_task(),
            "seed": self.seed.value(),
            "metadata": None,
        }
        assessment = self.analyze_dataset(
            pending["dataset"],
            task=str(pending["task"]),
            seed=int(pending["seed"]),
        )
        metadata = pending.get("metadata")
        return {
            "dataset": (
                asdict(metadata)
                if is_dataclass(metadata)
                else metadata or {"name": "built-in teaching data"}
            ),
            "assessment": assessment.to_dict(),
        }

    def start_dataset_analysis(self) -> None:
        if not self.analyze_data_button.isEnabled():
            return
        self._pending_analysis = {
            "dataset": self.dataset.currentData(),
            "task": self._selected_task(),
            "seed": self.seed.value(),
            "metadata": self.dataset.itemData(
                self.dataset.currentIndex(), Qt.ItemDataRole.UserRole + 1
            ),
        }
        self.analyze_data_button.setEnabled(False)
        self.dataset_metadata.setPlainText(
            "Analyzing data quality, label balance, duplicates, and safe split capacity…"
        )
        run_in_background(
            self,
            self._dataset_analysis_call,
            self._dataset_analysis_finished,
            self._dataset_analysis_failed,
        )

    def _dataset_analysis_finished(self, result: Any) -> None:
        self.analyze_data_button.setEnabled(True)
        self.dataset_metadata.setPlainText(_pretty(result))

    def _dataset_analysis_failed(self, error: str) -> None:
        self.analyze_data_button.setEnabled(True)
        summary = error.strip().splitlines()[-1] if error.strip() else "Unknown analysis error"
        self.dataset_metadata.setPlainText("Dataset analysis failed safely:\n" + summary)

    def _training_values(self) -> dict[str, Any]:
        return {
            "manager": self.manager,
            "workspace_root": _path_attr(self.manager, "workspace_root"),
            "project": self.project.currentData() or self.project.currentText(),
            "task": self._selected_task(),
            "dataset": self.dataset.currentData(),
            "epochs": self.epochs.value(),
            "batch_size": self.batch_size.value(),
            "learning_rate": self.learning_rate.value(),
            "seed": self.seed.value(),
            "validation_fraction": self.validation_percent.value() / 100.0,
            "test_fraction": self.test_percent.value() / 100.0,
            "standardize": self.standardize.isChecked(),
            "early_stopping_patience": self.early_stopping.value(),
        }

    def _training_call(self) -> Any:
        values = getattr(self, "_pending_training", None) or self._training_values()
        try:
            from daedalus.services.training import TrainingRequest, TrainingService

            project_name = _project_name(values.get("project") or "Teaching Lab")
            if project_name == "No project selected":
                project_name = "Teaching Lab"
            request = TrainingRequest(
                project=project_name,
                dataset=(str(values["dataset"]) if values.get("dataset") else None),
                task=str(values["task"]),
                epochs=int(values["epochs"]),
                batch_size=int(values["batch_size"]),
                learning_rate=float(values["learning_rate"]),
                seed=int(values["seed"]),
                validation_fraction=float(values["validation_fraction"]),
                test_fraction=float(values["test_fraction"]),
                standardize=bool(values["standardize"]),
                early_stopping_patience=int(values["early_stopping_patience"]),
            )
            return TrainingService(self.manager).run(request).to_dict()
        except ImportError:
            return {
                "available": False,
                "message": "The training engine is not available in this build yet.",
                "requested": {
                    key: str(value) for key, value in values.items() if key != "manager"
                },
            }

    def start_training(self) -> None:
        if self.run_button.isEnabled() is False:
            return
        self._pending_training = self._training_values()
        self.run_button.setEnabled(False)
        self.progress.setRange(0, 0)
        self.log.appendPlainText("\nStarting bounded training request…")
        run_in_background(self, self._training_call, self._training_finished, self._training_failed)

    def _training_finished(self, result: Any) -> None:
        self.progress.setRange(0, 100)
        self.progress.setValue(100)
        self.run_button.setEnabled(True)
        self.log.appendPlainText(_pretty(result))

    def _training_failed(self, error: str) -> None:
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.run_button.setEnabled(True)
        self.log.appendPlainText("Training request failed safely:\n" + error)


class CodeWorkshopPage(WorkspacePage):
    project_selected = Signal(object)
    TREE_SCAN_LIMIT = 600
    TREE_ITEM_LIMIT = 300
    TREE_LOADED_ROLE = Qt.ItemDataRole.UserRole + 1
    TEXT_SUFFIXES = {
        ".err",
        ".json",
        ".log",
        ".md",
        ".out",
        ".py",
        ".toml",
        ".txt",
        ".yaml",
        ".yml",
    }

    def __init__(self, manager, parent=None) -> None:
        from daedalus.gui.editor import CodeEditorPanel

        super().__init__(
            manager,
            "Code Workshop",
            "Edit private project files and run Python only through the constrained sandbox service.",
            "workshop",
            (
                (
                    "Execution boundary",
                    "The editor does not execute text. Run asks SandboxRunner to start a separate, "
                    "time- and path-confined process. It is not a virtual machine and must not be used for hostile code.",
                ),
                (
                    "Project custody",
                    "Only text files inside the configured projects directory are opened or saved. "
                    "Public application source is outside this editor's writable boundary.",
                ),
            ),
            parent,
            scroll_workspace=False,
        )
        self.current_file: Path | None = None
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.setOpaqueResize(False)

        browser = Card("Private projects", str(_path_attr(manager, "projects_dir")), accent=True)
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Project file", "Type"])
        self.tree.setAccessibleName("Private project file tree")
        self.tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.tree.itemDoubleClicked.connect(self._open_tree_item)
        self.tree.itemExpanded.connect(self._populate_tree_item)
        self.tree.currentItemChanged.connect(self._tree_project_changed)
        browser.add_widget(self.tree, 1)
        browser_buttons = QHBoxLayout()
        create = QPushButton("New project")
        create.setAccessibleName("Create private project")
        create.clicked.connect(self.create_project)
        refresh = QPushButton("Refresh")
        refresh.setAccessibleName("Refresh private project tree")
        refresh.clicked.connect(self.refresh_tree)
        browser_buttons.addWidget(create)
        browser_buttons.addWidget(refresh)
        browser.body.addLayout(browser_buttons)
        splitter.addWidget(browser)

        editor_card = Card("Editor", "No file open")
        self.file_label = editor_card.root.itemAt(1).widget()
        self.editor = CodeEditorPanel()
        self.editor.editor.setAccessibleName("Private project Python editor")
        editor_card.add_widget(self.editor, 1)
        editor_actions = QHBoxLayout()
        save = QPushButton("Save working file")
        save.setObjectName("Success")
        save.setAccessibleName("Save current private project file")
        save.clicked.connect(self.save_current)
        run = QPushButton("Run in sandbox")
        run.setObjectName("Primary")
        run.setIcon(semantic_icon("training", size=18))
        run.setAccessibleName("Run current Python file in constrained sandbox")
        run.clicked.connect(self.run_current)
        find = QPushButton("Find")
        find.setAccessibleName("Find text in current file")
        find.clicked.connect(self.editor.show_find)
        editor_actions.addWidget(save)
        editor_actions.addWidget(run)
        editor_actions.addWidget(find)
        editor_actions.addStretch(1)
        editor_card.body.addLayout(editor_actions)
        splitter.addWidget(editor_card)

        results = Card("Sandbox results", "Output, errors, and service boundaries appear here.")
        self.console = QPlainTextEdit()
        self.console.setReadOnly(True)
        self.console.setAccessibleName("Sandbox execution results")
        self.console.setPlainText("Select a .py file inside a private project to begin.")
        results.add_widget(self.console, 1)
        splitter.addWidget(results)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 5)
        splitter.setStretchFactor(2, 3)
        splitter.setSizes([260, 620, 360])
        self.tool_splitter = splitter
        self.workspace_layout.addWidget(splitter, 1)
        self.refresh_tree()

    def set_compact_layout(self, compact: bool) -> None:
        self.tool_splitter.setOrientation(
            Qt.Orientation.Vertical if compact else Qt.Orientation.Horizontal
        )
        self.tool_splitter.setSizes([150, 300, 180] if compact else [260, 620, 360])

    @property
    def projects_root(self) -> Path:
        return _path_attr(self.manager, "projects_dir")

    def _inside_projects(self, path: Path) -> bool:
        try:
            path.resolve(strict=False).relative_to(self.projects_root.resolve(strict=False))
            return True
        except (OSError, ValueError):
            return False

    def refresh_tree(self) -> None:
        self.tree.clear()
        root = self.projects_root
        if not root.is_dir():
            placeholder = QTreeWidgetItem(["Projects directory is not available", "status"])
            self.tree.addTopLevelItem(placeholder)
            return
        projects = [path for path in root.iterdir() if path.is_dir() and not path.name.startswith(".")]
        for project in sorted(projects, key=lambda item: item.name.casefold()):
            project_item = QTreeWidgetItem([project.name, "project"])
            project_item.setIcon(0, semantic_icon("architecture", size=17))
            project_item.setData(0, Qt.ItemDataRole.UserRole, str(project))
            self.tree.addTopLevelItem(project_item)
            self._prepare_directory_item(project_item)
        if not projects:
            self.tree.addTopLevelItem(QTreeWidgetItem(["No projects yet", "status"]))

    def _prepare_directory_item(self, item: QTreeWidgetItem) -> None:
        item.setData(0, self.TREE_LOADED_ROLE, False)
        placeholder = QTreeWidgetItem(["Expand to load…", "status"])
        placeholder.setDisabled(True)
        item.addChild(placeholder)

    def _populate_tree_item(self, parent: QTreeWidgetItem) -> None:
        if bool(parent.data(0, self.TREE_LOADED_ROLE)):
            return
        raw = parent.data(0, Qt.ItemDataRole.UserRole)
        if not raw:
            return
        directory = Path(str(raw))
        if not directory.is_dir() or not self._inside_projects(directory):
            return
        parent.takeChildren()
        parent.setData(0, self.TREE_LOADED_ROLE, True)
        entries: list[Path] = []
        truncated = False
        try:
            for scanned, entry in enumerate(directory.iterdir(), start=1):
                if scanned > self.TREE_SCAN_LIMIT:
                    truncated = True
                    break
                if entry.name.startswith(".") or entry.name in {"__pycache__", ".venv", "venv"}:
                    continue
                try:
                    is_directory = entry.is_dir()
                except OSError:
                    continue
                if is_directory or entry.suffix.casefold() in self.TEXT_SUFFIXES:
                    entries.append(entry)
        except OSError:
            return
        entries.sort(key=lambda item: (item.is_file(), item.name.casefold()))
        if len(entries) > self.TREE_ITEM_LIMIT:
            truncated = True
        for entry in entries[: self.TREE_ITEM_LIMIT]:
            try:
                is_directory = entry.is_dir()
            except OSError:
                continue
            if is_directory:
                item = QTreeWidgetItem([entry.name, "folder"])
                item.setIcon(0, semantic_icon("folder", size=16))
                item.setData(0, Qt.ItemDataRole.UserRole, str(entry))
                parent.addChild(item)
                self._prepare_directory_item(item)
            else:
                item = QTreeWidgetItem([entry.name, entry.suffix.casefold().lstrip(".") or "text"])
                item.setIcon(0, semantic_icon("workshop", size=16))
                item.setData(0, Qt.ItemDataRole.UserRole, str(entry))
                parent.addChild(item)
        if truncated:
            parent.addChild(
                QTreeWidgetItem(
                    [
                        f"More items not shown (bounded to {self.TREE_ITEM_LIMIT})",
                        "status",
                    ]
                )
            )

    def _open_tree_item(self, item: QTreeWidgetItem, _column: int) -> None:
        raw = item.data(0, Qt.ItemDataRole.UserRole)
        if not raw:
            return
        path = Path(str(raw))
        if path.is_file():
            self.open_file(path)

    def _tree_project_changed(
        self,
        current: QTreeWidgetItem | None,
        _previous: QTreeWidgetItem | None,
    ) -> None:
        item = current
        while item is not None and item.parent() is not None:
            item = item.parent()
        if item is None:
            return
        raw = item.data(0, Qt.ItemDataRole.UserRole)
        if not raw:
            return
        project = Path(str(raw)).resolve(strict=False)
        if project.parent == self.projects_root.resolve(strict=False) and project.is_dir():
            self.project_selected.emit(project)

    def open_file(self, path: Path) -> bool:
        if not self._inside_projects(path) or path.suffix.casefold() not in self.TEXT_SUFFIXES:
            self.console.setPlainText("Blocked: file is outside the private project text boundary.")
            return False
        try:
            if path.stat().st_size > 2 * 1024 * 1024:
                raise ValueError("The editor limit is 2 MiB per text file.")
            source = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError, ValueError) as exc:
            self.console.setPlainText(f"Could not open {path}:\n{exc}")
            return False
        self.current_file = path
        self.file_label.setText(str(path))
        self.editor.setPlainText(source)
        self.editor.editor.document().setModified(False)
        highlight_note = (
            ""
            if self.editor.syntax_highlighting_enabled
            else "\nSyntax highlighting is paused for this large file to keep editing responsive."
        )
        self.console.setPlainText(f"Opened private working file:\n{path}{highlight_note}")
        return True

    def create_project(self) -> None:
        name, ok = QInputDialog.getText(self, "New private project", "Project name:")
        if not ok or not name.strip():
            return
        cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", name.strip()).strip("-.")
        if not cleaned:
            QMessageBox.information(self, "New project", "Enter a valid project name.")
            return
        try:
            project = self.manager.create_project(cleaned, template="minimal")
            self.refresh_tree()
            for index in range(self.tree.topLevelItemCount()):
                item = self.tree.topLevelItem(index)
                raw = item.data(0, Qt.ItemDataRole.UserRole)
                if raw and Path(str(raw)).resolve(strict=False) == project.resolve(strict=False):
                    self.tree.setCurrentItem(item)
                    item.setExpanded(True)
                    break
        except Exception as exc:
            QMessageBox.warning(self, "New project", f"Project creation failed:\n{exc}")

    def save_current(self) -> bool:
        path = self.current_file
        if path is None or not self._inside_projects(path):
            self.console.setPlainText("No writable private project file is open.")
            return False
        temporary = path.with_name(f".{path.name}.daedalus-{time.time_ns()}.tmp")
        try:
            temporary.write_text(self.editor.toPlainText(), encoding="utf-8", newline="\n")
            temporary.replace(path)
        except OSError as exc:
            temporary.unlink(missing_ok=True)
            self.console.setPlainText(f"Save failed:\n{exc}")
            return False
        self.editor.editor.document().setModified(False)
        self.console.setPlainText(f"Saved private working file atomically:\n{path}")
        return True

    def _sandbox_call(self) -> Any:
        runner_type = _resolve_symbol(
            (
                ("daedalus.services.sandbox_runner", "SandboxRunner"),
                ("daedalus.services.sandbox", "SandboxRunner"),
            )
        )
        if runner_type is None:
            return {
                "available": False,
                "message": "SandboxRunner is not available in this build.",
                "file": str(self.current_file),
            }
        runner = runner_type(self.manager)
        return runner.run_file(self.current_file)

    def run_current(self) -> None:
        if self.current_file is None or self.current_file.suffix.casefold() != ".py":
            self.console.setPlainText("Only a selected .py project file can be sent to SandboxRunner.")
            return
        if self.editor.editor.document().isModified() and not self.save_current():
            return
        self.console.setPlainText("Running in constrained subprocess…")
        run_in_background(
            self,
            self._sandbox_call,
            lambda result: self.console.setPlainText(_pretty(result)),
            lambda error: self.console.setPlainText("Sandbox failed safely:\n" + error),
        )


class ModelEvaluatorPage(WorkspacePage):
    def __init__(self, manager, parent=None) -> None:
        super().__init__(
            manager,
            "Model Evaluator",
            "Replay a completed checkpoint on held-out data and record immutable quality evidence.",
            "evaluate",
            (
                (
                    "Checkpoint trust",
                    "Daedalus treats .npz arrays as data and opens them with allow_pickle=False. "
                    "A checkpoint is not proof of model quality; its architecture, preprocessing, and provenance still matter.",
                ),
                (
                    "Evaluation discipline",
                    "Keep the evaluation set separate from training, record latency and output shape, "
                    "and compare against a declared baseline rather than a single attractive example.",
                ),
            ),
            parent,
        )
        checkpoint_card = Card(
            "Trusted checkpoint",
            "Choose a Daedalus checkpoint, held-out split, and optional promotion gate.",
            accent=True,
        )
        self.checkpoint = PathField(
            "Checkpoint file",
            "",
            manager=manager,
            git_excluded=True,
            mode="file",
            file_filter="Daedalus checkpoints (*.npz *.json);;All files (*)",
        )
        checkpoint_card.add_widget(self.checkpoint)
        evaluation_options = QFormLayout()
        self.evaluation_split = QComboBox()
        self.evaluation_split.addItem("Test split (final evidence)", "test")
        self.evaluation_split.addItem("Validation split (iteration)", "validation")
        self.evaluation_split.setAccessibleName("Held-out evaluation split")
        evaluation_options.addRow("Held-out split", self.evaluation_split)

        self.acceptance_metric = QComboBox()
        self.acceptance_metric.addItem("No numeric gate", None)
        self.acceptance_metric.addItem("Accuracy ≥ threshold", "accuracy")
        self.acceptance_metric.addItem("Macro F1 ≥ threshold", "macro_f1")
        self.acceptance_metric.addItem("RMSE ≤ threshold", "rmse")
        self.acceptance_metric.addItem("MAE ≤ threshold", "mae")
        self.acceptance_metric.addItem("R² ≥ threshold", "r2")
        self.acceptance_metric.setAccessibleName("Acceptance metric for model promotion")
        self.acceptance_metric.currentIndexChanged.connect(self._gate_options_changed)
        evaluation_options.addRow("Acceptance gate", self.acceptance_metric)

        self.acceptance_threshold = QDoubleSpinBox()
        self.acceptance_threshold.setRange(-1_000_000_000.0, 1_000_000_000.0)
        self.acceptance_threshold.setDecimals(6)
        self.acceptance_threshold.setValue(0.8)
        self.acceptance_threshold.setAccessibleName("Acceptance threshold value")
        evaluation_options.addRow("Threshold", self.acceptance_threshold)

        baseline_row = QHBoxLayout()
        self.compare_baseline = QCheckBox("Compare with declared baseline")
        self.compare_baseline.setAccessibleName("Compare evaluated metric with a baseline")
        self.compare_baseline.toggled.connect(self._gate_options_changed)
        self.baseline_value = QDoubleSpinBox()
        self.baseline_value.setRange(-1_000_000_000.0, 1_000_000_000.0)
        self.baseline_value.setDecimals(6)
        self.baseline_value.setValue(0.5)
        self.baseline_value.setAccessibleName("Declared baseline metric value")
        baseline_row.addWidget(self.compare_baseline)
        baseline_row.addWidget(self.baseline_value)
        baseline_row.addStretch(1)
        evaluation_options.addRow("Baseline", baseline_row)
        checkpoint_card.body.addLayout(evaluation_options)
        self._gate_options_changed()

        row = QHBoxLayout()
        inspect_button = QPushButton("Inspect checkpoint")
        inspect_button.setObjectName("Primary")
        inspect_button.setAccessibleName("Inspect trusted checkpoint metadata")
        inspect_button.clicked.connect(self.inspect_checkpoint)
        self.evaluate_button = QPushButton("Replay held-out evaluation")
        self.evaluate_button.setAccessibleName(
            "Evaluate trusted checkpoint on its held-out data"
        )
        self.evaluate_button.clicked.connect(self.run_evaluator)
        open_root = QPushButton("Open checkpoint folder")
        open_root.setAccessibleName("Open configured checkpoint folder")
        open_root.clicked.connect(self._open_checkpoint_root)
        row.addWidget(inspect_button)
        row.addWidget(self.evaluate_button)
        row.addWidget(open_root)
        row.addStretch(1)
        checkpoint_card.body.addLayout(row)
        self.workspace_layout.addWidget(checkpoint_card)
        result = Card(
            "Evaluation report",
            "Held-out metrics, slice details, baseline comparisons, gate decision, and report path.",
        )
        self.report = QPlainTextEdit()
        self.report.setReadOnly(True)
        self.report.setAccessibleName("Model evaluation report")
        result.add_widget(self.report, 1)
        self.workspace_layout.addWidget(result, 1)

    def _gate_options_changed(self, _value: object = None) -> None:
        gated = self.acceptance_metric.currentData() is not None
        self.acceptance_threshold.setEnabled(gated)
        self.compare_baseline.setEnabled(gated)
        if not gated:
            self.compare_baseline.setChecked(False)
        self.baseline_value.setEnabled(gated and self.compare_baseline.isChecked())

    def _open_checkpoint_root(self) -> None:
        root = _path_attr(self.manager, "checkpoints_dir")
        opener = getattr(self.manager, "open_in_file_manager", None)
        if callable(opener):
            try:
                opener(root)
                return
            except Exception as exc:
                self.report.setPlainText(f"Could not open checkpoint directory:\n{exc}")

    def _inspect_call(self) -> dict[str, Any]:
        path = self.checkpoint.path
        if path.suffix.casefold() == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
            return {"file": str(path), "metadata": data}
        if path.suffix.casefold() != ".npz":
            raise ValueError("Only .npz checkpoints and .json metadata are accepted.")
        import numpy as np

        arrays: list[dict[str, Any]] = []
        with np.load(path, allow_pickle=False) as checkpoint:
            for name in checkpoint.files:
                array = checkpoint[name]
                arrays.append(
                    {
                        "name": name,
                        "shape": list(array.shape),
                        "dtype": str(array.dtype),
                        "bytes": int(array.nbytes),
                    }
                )
        return {"file": str(path), "file_bytes": path.stat().st_size, "arrays": arrays}

    def inspect_checkpoint(self) -> None:
        if not self.checkpoint.is_valid():
            self.report.setPlainText("Choose an existing trusted .npz or .json checkpoint first.")
            return
        self.report.setPlainText("Inspecting checkpoint as non-executable data…")
        run_in_background(
            self,
            self._inspect_call,
            lambda result: self.report.setPlainText(_pretty(result)),
            lambda error: self.report.setPlainText("Checkpoint inspection failed safely:\n" + error),
        )

    def _evaluator_call(self) -> Any:
        path = self.checkpoint.path
        split = str(self.evaluation_split.currentData() or "test")
        metric = self.acceptance_metric.currentData()
        thresholds = (
            {str(metric): float(self.acceptance_threshold.value())}
            if metric is not None
            else None
        )
        baseline = (
            {str(metric): float(self.baseline_value.value())}
            if metric is not None and self.compare_baseline.isChecked()
            else None
        )
        function = _resolve_symbol(
            (
                ("daedalus.engine.evaluator", "evaluate_checkpoint"),
                ("daedalus.engine.evaluation", "evaluate_checkpoint"),
            )
        )
        if function is not None:
            return _call_supported(
                function,
                path=path,
                checkpoint=path,
                manager=self.manager,
                split=split,
                acceptance_thresholds=thresholds,
                baseline=baseline,
            )
        evaluator_type = _resolve_symbol(
            (
                ("daedalus.engine.evaluator", "ModelEvaluator"),
                ("daedalus.engine.evaluation", "ModelEvaluator"),
            )
        )
        if evaluator_type is not None:
            evaluator = _call_supported(evaluator_type, manager=self.manager)
            action = getattr(evaluator, "evaluate_checkpoint", None) or getattr(evaluator, "evaluate", None)
            if callable(action):
                return _call_supported(
                    action,
                    path=path,
                    checkpoint=path,
                    split=split,
                    acceptance_thresholds=thresholds,
                    baseline=baseline,
                )
        return {"available": False, "message": "The Daedalus evaluator is unavailable."}

    def run_evaluator(self) -> None:
        if not self.checkpoint.is_valid():
            self.report.setPlainText("Choose an existing trusted checkpoint first.")
            return
        self.report.setPlainText("Replaying the recorded held-out split without data leakage…")
        run_in_background(
            self,
            self._evaluator_call,
            lambda result: self.report.setPlainText(_pretty(result)),
            lambda error: self.report.setPlainText("Evaluator failed safely:\n" + error),
        )


def _guard_findings(result: Any) -> list[Any]:
    if result is None:
        return []
    if isinstance(result, dict):
        for key in ("findings", "issues", "results", "violations"):
            value = result.get(key)
            if isinstance(value, (list, tuple)):
                return list(value)
        return []
    for key in ("findings", "issues", "results", "violations"):
        value = getattr(result, key, None)
        if isinstance(value, (list, tuple)):
            return list(value)
    return list(result) if isinstance(result, (list, tuple)) else []


def _finding_value(finding: Any, *names: str, default: str = "") -> str:
    for name in names:
        if isinstance(finding, dict) and finding.get(name) is not None:
            return str(finding[name])
        value = getattr(finding, name, None)
        if value is not None:
            return str(value)
    return default


class ReleaseGuardPage(WorkspacePage):
    status_changed = Signal(str, str)

    def __init__(self, manager, parent=None) -> None:
        super().__init__(
            manager,
            "Release Guard",
            "Scan source, dependencies, privacy boundaries, and Git policy before any publication.",
            "guard",
            (
                (
                    "Fail-closed publication",
                    "Blocking findings stop Safe Push. The guard must distinguish public application "
                    "source from private projects, datasets, checkpoints, logs, credentials, and generated models.",
                ),
                (
                    "Reviewable evidence",
                    "A clean result should identify which checks ran. An unavailable scanner is not "
                    "equivalent to a pass and must remain visible to the operator.",
                ),
            ),
            parent,
        )
        source_card = Card("Public source boundary", "This is the only tree eligible for guarded publication.", accent=True)
        self.source_path = PathField(
            "Source repository",
            _path_attr(manager, "source_root"),
            manager=manager,
            git_excluded=False,
        )
        source_card.add_widget(self.source_path)
        actions = QHBoxLayout()
        actions.addWidget(QLabel("Commit message"))
        self.commit_message = QLineEdit()
        self.commit_message.setPlaceholderText("Describe the public source change")
        self.commit_message.setAccessibleName("Safe Push commit message")
        actions.addWidget(self.commit_message, 1)
        self.scan_button = QPushButton("Scan release")
        self.scan_button.setObjectName("Primary")
        self.scan_button.setIcon(semantic_icon("guard", size=18))
        self.scan_button.setAccessibleName("Scan public source with Release Guard")
        self.scan_button.clicked.connect(self.start_scan)
        self.push_button = QPushButton("Safe Push")
        self.push_button.setObjectName("Warning")
        self.push_button.setIcon(semantic_icon("push", size=18))
        self.push_button.setAccessibleName("Run guarded GitHub safe push")
        self.push_button.clicked.connect(self.start_safe_push)
        actions.addWidget(self.scan_button)
        actions.addWidget(self.push_button)
        actions.addStretch(1)
        source_card.body.addLayout(actions)
        self.workspace_layout.addWidget(source_card)

        results = Card("Guard findings", "Severity, check, location, and remediation remain reviewable.")
        self.findings = QTableWidget(0, 4)
        self.findings.setHorizontalHeaderLabels(["Severity", "Check", "Location", "Message"])
        self.findings.verticalHeader().setVisible(False)
        self.findings.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.findings.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.findings.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.findings.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.findings.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Interactive)
        self.findings.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self.output = QPlainTextEdit()
        self.output.setReadOnly(True)
        self.output.setMaximumHeight(150)
        self.output.setAccessibleName("Release Guard report")
        results.add_widget(self.findings, 1)
        results.add_widget(self.output)
        self.workspace_layout.addWidget(results, 1)

    def _guard(self):
        guard_type = _resolve_symbol((("daedalus.services.release_guard", "ReleaseGuard"),))
        if guard_type is None:
            raise RuntimeError("ReleaseGuard service is not available in this build.")
        return guard_type(self.source_path.path)

    def _scan_call(self) -> Any:
        return self._guard().scan(
            include_tests=True,
            include_dependencies=True,
            include_github=True,
        )

    def start_scan(self) -> None:
        if not self.source_path.is_valid():
            self.output.setPlainText(
                "Release scan has not started. Select an existing public source directory."
            )
            self.status_changed.emit("Release scan needs a valid source directory.", "warning")
            return
        self._begin("Scanning public source…")
        run_in_background(self, self._scan_call, self._scan_finished, self._failed)

    def _safe_push_call(self) -> Any:
        guard = self._guard()
        action = getattr(guard, "safe_push", None) or getattr(guard, "push_safe", None)
        if not callable(action):
            scan = guard.scan()
            return {
                "pushed": False,
                "scan": scan,
                "message": "The guard scan ran, but this build does not expose a Safe Push transport.",
            }
        return _call_supported(
            action,
            message=self._pending_commit_message,
            auto_fix=False,
            push=True,
        )

    def start_safe_push(self) -> None:
        if not self.source_path.is_valid():
            self.output.setPlainText(
                "Safe Push has not started. Select an existing public source directory."
            )
            self.status_changed.emit("Safe Push needs a valid source directory.", "warning")
            return
        message = self.commit_message.text().strip()
        if not message:
            self.output.setPlainText(
                "Safe Push has not started. Enter a specific commit message, then review and activate Safe Push again."
            )
            self.commit_message.setFocus()
            self.status_changed.emit("Safe Push needs a commit message.", "warning")
            return
        self._pending_commit_message = message
        self._begin("Running fail-closed Safe Push…")
        run_in_background(self, self._safe_push_call, self._push_finished, self._failed)

    def _begin(self, text: str) -> None:
        self.scan_button.setEnabled(False)
        self.push_button.setEnabled(False)
        self.output.setPlainText(text)
        self.status_changed.emit(text, "muted")

    def _restore_buttons(self) -> None:
        self.scan_button.setEnabled(True)
        self.push_button.setEnabled(True)

    def _scan_finished(self, result: Any) -> None:
        self._restore_buttons()
        self._display_result(result)
        count = len(_guard_findings(result))
        level = "success" if count == 0 else "warning"
        message = f"Release scan completed with {count} finding(s)."
        self.status_changed.emit(message, level)

    def _push_finished(self, result: Any) -> None:
        self._restore_buttons()
        self._display_result(result)
        pushed = any(
            _finding_value(finding, "check").casefold() == "push"
            and _finding_value(finding, "level", "severity").casefold() in {"pass", "success"}
            and "push" in _finding_value(finding, "message", "detail").casefold()
            for finding in _guard_findings(result)
        )
        ok = bool(getattr(result, "ok", False))
        message = (
            "Safe Push completed."
            if pushed
            else "Guard checks passed, but no remote push was confirmed."
            if ok
            else "Safe Push stopped or was unavailable; review the report."
        )
        self.status_changed.emit(message, "success" if pushed else "warning")

    def _failed(self, error: str) -> None:
        self._restore_buttons()
        self.findings.setRowCount(0)
        self.output.setPlainText("Release operation failed closed:\n" + error)
        self.status_changed.emit("Release operation failed closed.", "danger")

    def _display_result(self, result: Any) -> None:
        findings = _guard_findings(result)
        self.findings.setRowCount(len(findings))
        for row, finding in enumerate(findings):
            values = (
                _finding_value(finding, "severity", "level", default="info"),
                _finding_value(finding, "check", "code", "rule", default="guard"),
                _finding_value(finding, "path", "location", "file"),
                _finding_value(finding, "message", "detail", "description", default=str(finding)),
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if column == 0:
                    color = {
                        "error": "#fb7185",
                        "critical": "#fb7185",
                        "warning": "#fbbf24",
                        "high": "#f97316",
                    }.get(value.casefold(), "#94a3b8")
                    item.setForeground(QColor(color))
                self.findings.setItem(row, column, item)
        self.output.setPlainText(_pretty(result))


class VaultBackupPage(WorkspacePage):
    """Operator-facing backup status, execution, and restore workspace.

    The page deliberately treats configured manager paths as read-only. Editing a
    display field would not reconfigure ``BackupService`` and could otherwise
    create a dangerous mismatch between what is shown and what is copied.
    """

    status_changed = Signal(str, str)

    def __init__(self, manager, parent=None) -> None:
        super().__init__(
            manager,
            "Vault & Backup",
            "Verify custody boundaries, create non-destructive recovery copies, and restore only into a new folder.",
            "backup",
            (
                (
                    "Non-destructive backup",
                    "Daedalus copies public source and the private workspace into separate current trees. "
                    "A later run never deletes destination-only files, so an accidental local deletion does "
                    "not immediately erase its recovery copy.",
                ),
                (
                    "Destination ownership",
                    "The backup service refuses any destination that overlaps source or workspace. A non-empty "
                    "destination must already contain the Daedalus ownership marker; arbitrary folders are never reused.",
                ),
                (
                    "Restore isolation",
                    "Restore creates a new folder and refuses an existing destination. This page additionally blocks "
                    "destinations inside the source, active workspace, or backup root. It never switches the active workspace.",
                ),
            ),
            parent,
        )
        self._busy = False
        self._refresh_generation = 0
        self._refresh_running = False
        self._refresh_pending = False
        self._latest_manifest: dict[str, Any] | None = None
        self._last_restore_path: Path | None = None

        metrics = QGridLayout()
        self.metric_layout = metrics
        self.destination_metric = MetricTile("Destination", "Unchecked", "Configured backup root")
        self.latest_metric = MetricTile("Latest backup", "None", "No manifest found")
        self.files_metric = MetricTile("Files copied", "0", "Most recent run")
        self.bytes_metric = MetricTile("Data copied", "0 B", "Most recent run")
        self.metric_tiles = (
            self.destination_metric,
            self.latest_metric,
            self.files_metric,
            self.bytes_metric,
        )
        for column, tile in enumerate(self.metric_tiles):
            metrics.addWidget(tile, 0, column)
            metrics.setColumnStretch(column, 1)
        self._compact_metrics = False
        self.workspace_layout.addLayout(metrics)

        custody = Card(
            "Custody map",
            "These are the active service paths. Change them only through supported configuration or migration tooling.",
            accent=True,
        )
        self.source_path = self._display_path(
            "Public source copied to source-current",
            _path_attr(manager, "source_root"),
            git_excluded=False,
        )
        self.workspace_path = self._display_path(
            "Private workspace copied to workspace-current",
            _path_attr(manager, "workspace_root"),
            git_excluded=True,
        )
        self.backup_path = self._display_path(
            "Configured backup root",
            _path_attr(manager, "backup_root"),
            git_excluded=True,
            allow_missing=True,
        )
        custody.add_widget(self.source_path)
        custody.add_widget(self.workspace_path)
        custody.add_widget(self.backup_path)
        self.workspace_layout.addWidget(custody)

        controls = Card(
            "Backup controls",
            "Validation is read-only. Backup copies changed files in a worker thread and records a reviewable manifest.",
        )
        row = QHBoxLayout()
        self.validate_button = QPushButton("Validate destination")
        self.validate_button.setIcon(semantic_icon("guard", size=18))
        self.validate_button.setAccessibleName("Validate backup destination without copying files")
        self.validate_button.clicked.connect(self.start_validation)
        self.backup_button = QPushButton("Back up now")
        self.backup_button.setObjectName("Primary")
        self.backup_button.setIcon(semantic_icon("backup", size=18))
        self.backup_button.setAccessibleName("Run a non-destructive backup in the background")
        self.backup_button.clicked.connect(self.start_backup)
        self.refresh_button = QPushButton("Refresh manifest")
        self.refresh_button.setAccessibleName("Refresh latest backup manifest")
        self.refresh_button.clicked.connect(self.refresh)
        self.open_backup_button = QPushButton("Open backup root")
        self.open_backup_button.setIcon(semantic_icon("folder", size=18))
        self.open_backup_button.setAccessibleName("Open configured backup root in file manager")
        self.open_backup_button.clicked.connect(self._open_backup_root)
        for button in (
            self.validate_button,
            self.backup_button,
            self.refresh_button,
            self.open_backup_button,
        ):
            row.addWidget(button)
        row.addStretch(1)
        controls.body.addLayout(row)
        self.destination_state = QLabel()
        self.destination_state.setWordWrap(True)
        self.destination_state.setAccessibleName("Backup destination validation status")
        controls.add_widget(self.destination_state)
        self.activity_output = QPlainTextEdit()
        self.activity_output.setReadOnly(True)
        self.activity_output.setMaximumHeight(130)
        self.activity_output.setAccessibleName("Backup operation report")
        controls.add_widget(self.activity_output)
        self.workspace_layout.addWidget(controls)

        manifest_card = Card(
            "Latest manifest",
            "The manifest records timestamps, destination, scan/copy counts, copied bytes, skipped links, and errors.",
        )
        self.manifest_output = QPlainTextEdit()
        self.manifest_output.setReadOnly(True)
        self.manifest_output.setMaximumHeight(220)
        self.manifest_output.setAccessibleName("Latest backup manifest")
        manifest_card.add_widget(self.manifest_output)
        self.workspace_layout.addWidget(manifest_card)

        restore = Card(
            "Restore to a new folder",
            "Choose an existing parent and a new child-folder name. Existing folders and protected custody trees are blocked.",
        )
        restore_row = QHBoxLayout()
        self.restore_destination = QLineEdit(str(self._suggested_restore_target()))
        self.restore_destination.setClearButtonEnabled(True)
        self.restore_destination.setAccessibleName("New restore destination")
        self.restore_destination.setPlaceholderText("Absolute path for a new restore folder")
        self.restore_destination.textChanged.connect(self._refresh_restore_state)
        restore_row.addWidget(self.restore_destination, 1)
        self.choose_restore_parent_button = QPushButton("Choose parent…")
        self.choose_restore_parent_button.setIcon(semantic_icon("folder", size=18))
        self.choose_restore_parent_button.setAccessibleName("Choose parent folder for restored workspace")
        self.choose_restore_parent_button.clicked.connect(self._choose_restore_parent)
        restore_row.addWidget(self.choose_restore_parent_button)
        self.suggest_restore_button = QPushButton("Suggest new folder")
        self.suggest_restore_button.setAccessibleName("Suggest a new non-existing restore folder")
        self.suggest_restore_button.clicked.connect(
            lambda: self.restore_destination.setText(str(self._suggested_restore_target()))
        )
        restore_row.addWidget(self.suggest_restore_button)
        restore.body.addLayout(restore_row)
        self.restore_state = QLabel()
        self.restore_state.setWordWrap(True)
        self.restore_state.setAccessibleName("Restore destination safety status")
        restore.add_widget(self.restore_state)
        self.restore_confirmation = QCheckBox(
            "I understand this creates a separate folder and does not replace or activate it."
        )
        self.restore_confirmation.setAccessibleName("Confirm isolated restore behavior")
        self.restore_confirmation.toggled.connect(self._refresh_restore_state)
        restore.add_widget(self.restore_confirmation)
        restore_actions = QHBoxLayout()
        self.restore_button = QPushButton("Restore to new folder")
        self.restore_button.setObjectName("Warning")
        self.restore_button.setIcon(semantic_icon("backup", size=18))
        self.restore_button.setAccessibleName("Restore workspace backup into a new folder")
        self.restore_button.clicked.connect(self.start_restore)
        self.open_restore_button = QPushButton("Open restored folder")
        self.open_restore_button.setIcon(semantic_icon("folder", size=18))
        self.open_restore_button.setAccessibleName("Open the newly restored workspace folder")
        self.open_restore_button.setEnabled(False)
        self.open_restore_button.clicked.connect(self._open_restored_folder)
        restore_actions.addWidget(self.restore_button)
        restore_actions.addWidget(self.open_restore_button)
        restore_actions.addStretch(1)
        restore.body.addLayout(restore_actions)
        self.workspace_layout.addWidget(restore)
        self.refresh()

    def set_compact_layout(self, compact: bool) -> None:
        compact = bool(compact)
        if compact == self._compact_metrics:
            return
        self._compact_metrics = compact
        columns = 2 if compact else 4
        for tile in self.metric_tiles:
            self.metric_layout.removeWidget(tile)
        for column in range(4):
            self.metric_layout.setColumnStretch(column, 0)
        for index, tile in enumerate(self.metric_tiles):
            self.metric_layout.addWidget(tile, index // columns, index % columns)
        for column in range(columns):
            self.metric_layout.setColumnStretch(column, 1)

    def _display_path(
        self,
        title: str,
        path: Path,
        *,
        git_excluded: bool,
        allow_missing: bool = False,
    ) -> PathField:
        field = PathField(
            title,
            path,
            manager=self.manager,
            git_excluded=git_excluded,
            allow_missing=allow_missing,
        )
        field.line_edit.setReadOnly(True)
        field.line_edit.setToolTip("This field reflects active configuration and is read-only here.")
        field.browse_button.setVisible(False)
        return field

    def _service(self):
        from daedalus.services.backup import BackupService

        return BackupService(self.manager)

    @staticmethod
    def _is_within(candidate: Path, root: Path) -> bool:
        try:
            candidate.relative_to(root)
        except ValueError:
            return False
        return True

    def _suggested_restore_target(self, parent: Path | None = None) -> Path:
        workspace = _path_attr(self.manager, "workspace_root")
        base_parent = parent or workspace.parent
        stem = f"{workspace.name or 'Daedalus Workspace'} Restored {time.strftime('%Y%m%d-%H%M%S')}"
        candidate = base_parent / stem
        suffix = 2
        while candidate.exists():
            candidate = base_parent / f"{stem} {suffix}"
            suffix += 1
        return candidate

    def _set_state(self, label: QLabel, text: str, level: str) -> None:
        label.setText(text)
        label.setObjectName(
            "Success" if level == "success" else "Warning" if level == "warning" else "Danger"
        )
        style = label.style()
        style.unpolish(label)
        style.polish(label)

    def _validate_call(self) -> dict[str, Any]:
        service = self._service()
        service.validate_destination()
        return {
            "destination": str(service.backup_root),
            "exists": service.backup_root.exists(),
            "workspace_backup": (service.backup_root / "workspace-current").is_dir(),
            "message": (
                "Destination is owned and ready."
                if service.backup_root.exists()
                else "Destination boundary is valid and will be created on the first backup."
            ),
        }

    def refresh(self) -> None:
        if self._busy:
            return
        for field, name in (
            (self.source_path, "source_root"),
            (self.workspace_path, "workspace_root"),
            (self.backup_path, "backup_root"),
        ):
            field.set_path(_path_attr(self.manager, name))
        backup_root = _path_attr(self.manager, "backup_root")
        self.open_backup_button.setEnabled(backup_root.is_dir())
        if self._refresh_running:
            self._refresh_pending = True
            return
        self._refresh_generation += 1
        generation = self._refresh_generation
        self._refresh_running = True
        self.destination_metric.set_value("Checking…", str(backup_root))
        self.destination_state.setObjectName("Muted")
        self.destination_state.setText("Validating the configured destination in the background…")
        repolish(self.destination_state)
        run_in_background(
            self,
            self._refresh_snapshot,
            lambda result, current=generation: self._refresh_finished(current, result),
            lambda error, current=generation: self._refresh_failed(current, error),
        )
        self._refresh_restore_state()

    def _refresh_snapshot(self) -> dict[str, Any]:
        """Read destination and manifest state without touching GUI objects."""

        result: dict[str, Any] = {
            "validation": None,
            "validation_error": "",
            "manifest": None,
            "manifest_error": "",
        }
        try:
            service = self._service()
            service.validate_destination()
            result["validation"] = {
                "destination": str(service.backup_root),
                "exists": service.backup_root.exists(),
                "workspace_backup": (service.backup_root / "workspace-current").is_dir(),
                "message": (
                    "Destination is owned and ready."
                    if service.backup_root.exists()
                    else "Destination boundary is valid and will be created on the first backup."
                ),
            }
        except Exception as exc:
            result["validation_error"] = str(exc)
        try:
            service = locals().get("service") or self._service()
            result["manifest"] = service.latest_status()
        except Exception as exc:
            result["manifest_error"] = str(exc)
        return result

    def _refresh_finished(self, generation: int, result: Any) -> None:
        self._refresh_running = False
        if generation != self._refresh_generation:
            return
        payload = result if isinstance(result, dict) else {}
        backup_root = _path_attr(self.manager, "backup_root")
        validation = payload.get("validation")
        if isinstance(validation, dict):
            self._show_validation(validation)
        else:
            self.destination_metric.set_value("Blocked", str(backup_root))
            self._set_state(
                self.destination_state,
                "Destination validation failed safely: "
                + str(payload.get("validation_error") or "unknown error"),
                "danger",
            )
        manifest_error = str(payload.get("manifest_error") or "")
        if manifest_error:
            self._latest_manifest = None
            self.manifest_output.setPlainText(
                "Latest manifest could not be read safely:\n" + manifest_error
            )
            self.latest_metric.set_value("Unreadable", "Review the manifest file")
            self.files_metric.set_value("—", "Manifest unavailable")
            self.bytes_metric.set_value("—", "Manifest unavailable")
        else:
            manifest = payload.get("manifest")
            self._latest_manifest = manifest if isinstance(manifest, dict) else None
            self._render_manifest(self._latest_manifest)
        self._refresh_restore_state()
        if self._refresh_pending:
            self._refresh_pending = False
            self.refresh()

    def _refresh_failed(self, generation: int, error: str) -> None:
        self._refresh_running = False
        if generation != self._refresh_generation:
            return
        detail = error.strip().splitlines()[-1] if error.strip() else "unknown error"
        self.destination_metric.set_value("Unavailable", "Background refresh failed")
        self._set_state(
            self.destination_state,
            "Backup status refresh failed safely: " + detail,
            "danger",
        )

    def _show_validation(self, result: dict[str, Any]) -> None:
        detail = str(result["destination"])
        self.destination_metric.set_value("Ready", detail)
        self._set_state(self.destination_state, str(result["message"]), "success")

    @staticmethod
    def _manifest_count(value: Any) -> int:
        try:
            return max(0, int(value))
        except (TypeError, ValueError, OverflowError):
            return 0

    def _render_manifest(self, manifest: dict[str, Any] | None) -> None:
        if not manifest:
            self.latest_metric.set_value("None", "Run the first backup to create a manifest")
            self.files_metric.set_value("0", "No completed run")
            self.bytes_metric.set_value("0 B", "No completed run")
            self.manifest_output.setPlainText(
                "No latest.json manifest exists yet. Validate the destination, then run Back up now."
            )
            return
        if not isinstance(manifest, dict):
            self.latest_metric.set_value("Unreadable", "Manifest root must be an object")
            self.files_metric.set_value("—", "Manifest unavailable")
            self.bytes_metric.set_value("—", "Manifest unavailable")
            self.manifest_output.setPlainText(
                "Latest manifest is invalid: expected a JSON object. No backup or restore was started."
            )
            return
        finished = str(manifest.get("finished_utc") or "Unknown")
        copied = self._manifest_count(manifest.get("files_copied"))
        copied_bytes = self._manifest_count(manifest.get("bytes_copied"))
        errors = manifest.get("errors") or []
        self.latest_metric.set_value(finished.replace("T", " ")[:19], str(manifest.get("destination") or ""))
        self.files_metric.set_value(
            copied,
            f"{self._manifest_count(manifest.get('files_scanned'))} scanned",
        )
        error_count = len(errors) if isinstance(errors, (list, tuple)) else 1
        self.bytes_metric.set_value(human_bytes(copied_bytes), f"{error_count} recorded error(s)")
        self.manifest_output.setPlainText(_pretty(manifest))

    def _set_busy(self, busy: bool, message: str = "") -> None:
        self._busy = busy
        if busy and self._refresh_running:
            self._refresh_generation += 1
            self._refresh_pending = False
        for button in (
            self.validate_button,
            self.backup_button,
            self.refresh_button,
            self.choose_restore_parent_button,
            self.suggest_restore_button,
        ):
            button.setEnabled(not busy)
        self.restore_destination.setEnabled(not busy)
        self.restore_confirmation.setEnabled(not busy)
        if message:
            self.activity_output.setPlainText(message)
            self.status_changed.emit(message, "muted")
        self._refresh_restore_state()

    def start_validation(self) -> None:
        if self._busy:
            return
        self._set_busy(True, "Validating the configured destination without copying files…")
        run_in_background(self, self._validate_call, self._validation_finished, self._operation_failed)

    def _validation_finished(self, result: Any) -> None:
        self._set_busy(False)
        validation = dict(result) if isinstance(result, dict) else {"message": _pretty(result)}
        validation.setdefault("destination", str(_path_attr(self.manager, "backup_root")))
        self._show_validation(validation)
        self.activity_output.setPlainText(_pretty(validation))
        self.status_changed.emit("Backup destination is valid.", "success")

    def start_backup(self) -> None:
        if self._busy:
            return
        self._set_busy(
            True,
            "Creating a non-destructive backup. Existing recovery-only files will not be deleted…",
        )
        run_in_background(self, lambda: self._service().run(), self._backup_finished, self._operation_failed)

    def _backup_finished(self, result: Any) -> None:
        self._set_busy(False)
        payload = asdict(result) if is_dataclass(result) else result
        self.activity_output.setPlainText(_pretty(payload))
        self._latest_manifest = payload if isinstance(payload, dict) else self._service().latest_status()
        self._render_manifest(self._latest_manifest)
        self.open_backup_button.setEnabled(_path_attr(self.manager, "backup_root").is_dir())
        errors = self._latest_manifest.get("errors", []) if self._latest_manifest else []
        if errors:
            message = f"Backup completed with {len(errors)} recorded error(s); review the manifest."
            level = "warning"
        else:
            message = "Backup completed, verified, and recorded in the latest manifest."
            level = "success"
        self.status_changed.emit(message, level)
        self._refresh_restore_state()

    def _restore_validation(self) -> tuple[bool, str, Path | None]:
        raw = self.restore_destination.text().strip()
        if not raw:
            return False, "Choose an absolute destination for the new restore folder.", None
        try:
            target = Path(raw).expanduser()
        except (OSError, RuntimeError, ValueError) as exc:
            return False, f"Blocked: restore destination is invalid ({exc}).", None
        if not target.is_absolute():
            return False, "Restore destination must be an absolute path.", target
        try:
            target = target.resolve(strict=False)
        except (OSError, RuntimeError, ValueError) as exc:
            return False, f"Blocked: restore destination cannot be resolved ({exc}).", target
        if target.exists():
            return False, "Blocked: restore destination already exists. Choose a new folder name.", target
        if not target.parent.is_dir():
            return False, "Blocked: choose an existing parent folder before restoring.", target
        protected = (
            ("public source", _path_attr(self.manager, "source_root")),
            ("active workspace", _path_attr(self.manager, "workspace_root")),
            ("backup root", _path_attr(self.manager, "backup_root")),
        )
        for label, root in protected:
            root = root.resolve(strict=False)
            if target == root or self._is_within(target, root):
                return False, f"Blocked: restore destination is inside the {label} custody tree.", target
        workspace_copy = _path_attr(self.manager, "backup_root") / "workspace-current"
        if not workspace_copy.is_dir():
            return False, "No workspace-current backup is available to restore yet.", target
        return True, "Safe new-folder destination. Confirm the isolation statement to enable restore.", target

    def _refresh_restore_state(self, *_args: Any) -> None:
        valid, message, _target = self._restore_validation()
        confirmed = self.restore_confirmation.isChecked()
        if valid and confirmed:
            message = "Ready: a separate restored workspace will be created; the active workspace stays unchanged."
        level = "success" if valid else "warning"
        self._set_state(self.restore_state, message, level)
        self.restore_button.setEnabled(valid and confirmed and not self._busy)

    def _choose_restore_parent(self) -> None:
        workspace = _path_attr(self.manager, "workspace_root")
        selected = QFileDialog.getExistingDirectory(
            self,
            "Choose parent for restored workspace",
            str(workspace.parent),
        )
        if selected:
            self.restore_destination.setText(str(self._suggested_restore_target(Path(selected))))

    def start_restore(self) -> None:
        if self._busy:
            return
        valid, message, target = self._restore_validation()
        if not valid or target is None or not self.restore_confirmation.isChecked():
            self.activity_output.setPlainText("Restore has not started. " + message)
            self._refresh_restore_state()
            return
        self._pending_restore_target = target
        self._set_busy(True, f"Restoring workspace backup into new folder:\n{target}")
        run_in_background(
            self,
            lambda: self._service().restore_workspace(self._pending_restore_target),
            self._restore_finished,
            self._operation_failed,
        )

    def _restore_finished(self, result: Any) -> None:
        self._last_restore_path = Path(str(result)).resolve(strict=False)
        self._set_busy(False)
        self.restore_confirmation.setChecked(False)
        self.open_restore_button.setEnabled(self._last_restore_path.is_dir())
        self.activity_output.setPlainText(
            "Restore completed into a separate folder. The active workspace was not changed.\n"
            + str(self._last_restore_path)
        )
        self.status_changed.emit("Workspace restored into a new isolated folder.", "success")
        self._refresh_restore_state()

    def _operation_failed(self, error: str) -> None:
        self._set_busy(False)
        summary = error.strip().splitlines()[-1] if error.strip() else "unknown error"
        self.activity_output.setPlainText(
            "Operation failed safely; no existing files were replaced. A failed restore may leave "
            "a partial new folder for inspection, so choose a different new destination before retrying.\n"
            + error
        )
        self.status_changed.emit(f"Vault operation failed safely: {summary}", "danger")

    def _open_backup_root(self) -> None:
        root = _path_attr(self.manager, "backup_root")
        if root.is_dir():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(root)))

    def _open_restored_folder(self) -> None:
        if self._last_restore_path is not None and self._last_restore_path.is_dir():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._last_restore_path)))


class SettingsPage(WorkspacePage):
    appearance_changed = Signal(str, float, bool, bool)

    def __init__(self, manager, parent=None) -> None:
        super().__init__(
            manager,
            "Settings",
            "Appearance, accessibility, workspace custody, and backup destinations.",
            "settings",
            (
                (
                    "Accessible motion",
                    "Reduced motion disables optional animation while preserving every destination "
                    "and status change. The DAEDALUS_REDUCE_MOTION environment override always wins.",
                ),
                (
                    "Path custody",
                    "Changing a displayed path does not migrate data automatically. Use installer or "
                    "workspace migration tooling for real moves so backups and exclusion rules remain consistent.",
                ),
            ),
            parent,
        )
        appearance = Card("Appearance", "Theme and scale changes apply to the current session.", accent=True)
        form = QFormLayout()
        self.theme = QComboBox()
        for key, label in available_themes():
            self.theme.addItem(label, key)
        self.theme.setAccessibleName("Application theme")
        self.auto_scale = QCheckBox("Use system display scale (recommended)")
        self.auto_scale.setChecked(True)
        self.auto_scale.setAccessibleName("Use automatic system display scaling")
        self.auto_scale.setToolTip(
            "Uses Qt and the operating system's DPI-aware logical scale without restyling on resize."
        )
        self.scale = QDoubleSpinBox()
        self.scale.setRange(0.78, 1.35)
        self.scale.setSingleStep(0.05)
        self.scale.setValue(1.0)
        self.scale.setAccessibleName("Fixed interface scale")
        self.scale.setEnabled(False)
        self.auto_scale.toggled.connect(lambda checked: self.scale.setEnabled(not checked))
        self.reduce_motion = QCheckBox("Reduce non-essential motion")
        self.reduce_motion.setChecked(reduced_motion())
        self.reduce_motion.setAccessibleName("Reduced motion preference")
        form.addRow("Theme", self.theme)
        form.addRow("Display scale", self.auto_scale)
        form.addRow("Fixed scale", self.scale)
        form.addRow("Accessibility", self.reduce_motion)
        appearance.body.addLayout(form)
        apply_button = QPushButton("Apply appearance")
        apply_button.setObjectName("Primary")
        apply_button.setAccessibleName("Apply theme and accessibility preferences")
        apply_button.clicked.connect(self.apply_appearance)
        appearance.add_widget(apply_button)
        self.workspace_layout.addWidget(appearance)

        custody = Card("Workspace custody", "These paths are displayed explicitly so private artifacts are never mistaken for public source.")
        self.workspace_path = PathField(
            "External workspace",
            _path_attr(manager, "workspace_root"),
            manager=manager,
            git_excluded=True,
        )
        self.backup_path = PathField(
            "Backup root",
            _path_attr(manager, "backup_root"),
            manager=manager,
            git_excluded=True,
            allow_missing=True,
        )
        custody.add_widget(self.workspace_path)
        custody.add_widget(self.backup_path)
        for field in (self.workspace_path, self.backup_path):
            field.line_edit.setReadOnly(True)
            field.line_edit.setCursorPosition(0)
            field.line_edit.setClearButtonEnabled(False)
            field.browse_button.setVisible(False)
        note = QLabel(
            "Custody paths are read-only here so the displayed location cannot diverge from the "
            "active backup/workspace service. Change them through the installer or documented "
            "environment settings, then restart Daedalus."
        )
        note.setObjectName("Warning")
        note.setWordWrap(True)
        custody.add_widget(note)
        self.workspace_layout.addWidget(custody)

    def set_theme(self, key: str) -> None:
        index = self.theme.findData(key)
        if index >= 0:
            self.theme.setCurrentIndex(index)

    def apply_appearance(self) -> None:
        self.appearance_changed.emit(
            str(self.theme.currentData()),
            float(self.scale.value()),
            bool(self.auto_scale.isChecked()),
            bool(self.reduce_motion.isChecked()),
        )


PAGE_CLASSES = {
    "mission": MissionControlPage,
    "developer": DeveloperBotPage,
    "learn": LearningAtlasPage,
    "architecture": ArchitectureBuilderPage,
    "calculator": CalculatorLabPage,
    "training": TrainingLabPage,
    "workshop": CodeWorkshopPage,
    "evaluate": ModelEvaluatorPage,
    "guard": ReleaseGuardPage,
    "vault": VaultBackupPage,
    "settings": SettingsPage,
}
