"""Guided Weight Lab, per-tool sandbox, and current learning links."""

from __future__ import annotations

from functools import partial
from html import escape
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlencode

import numpy as np
from PySide6.QtCore import Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from daedalus.engine.weight_tools import (
    WEIGHT_TOOL_SPECS,
    WeightToolRecord,
    WeightToolSpec,
    compile_truth_table,
    fit_extreme_learning_machine,
    fit_physics_constrained_polynomial,
    get_weight_tool_spec,
    recommend_weight_tool,
    select_uncertain_candidates,
    synthesize_low_rank_adapter,
    synthesize_recurrent_kernel,
)
from daedalus.gui.editor import CodeEditorPanel
from daedalus.gui.reveal import ToolRevealHost
from daedalus.gui.widgets import run_in_background
from daedalus.services.weight_sandbox import (
    WeightSandboxService,
    sandbox_template,
)

_YOUTUBE_ROOT = "https://www.youtube.com/results"
_YOUTUBE_HOSTS = frozenset({"youtube.com", "www.youtube.com"})
_REVIEWED_DATE = "2026-08-30"

_PIPELINES = {
    "meta_weight": "NUMERIC CONTEXT  →  SEEDED CONTROLLER  →  LOW-RANK ΔW",
    "logic_compiler": "COMPLETE BINARY TABLE  →  PATTERN DETECTORS  →  EXACT OUTPUT",
    "recurrent_kernel": "SCALAR STREAM  →  SELECTIVE STABLE STATE  →  OUTPUT + KERNEL",
    "constraint_optimizer": "DATA + TYPED OPERATOR  →  BOUNDED LEAST SQUARES  →  u(x)",
    "matrix_inverter": "NUMERIC DATA  →  FROZEN RANDOM FEATURES  →  SOLVED OUTPUT WEIGHTS",
    "uncertainty_sampler": "LABELED DATA + POOL  →  RBF-GP POSTERIOR  →  NEXT ROWS TO LABEL",
}


def _scroll(widget: QWidget) -> QScrollArea:
    area = QScrollArea()
    area.setFrameShape(QFrame.Shape.NoFrame)
    area.setWidgetResizable(True)
    area.setWidget(widget)
    widget.setAutoFillBackground(False)
    return area


def _matrix(text: str, name: str) -> np.ndarray:
    """Parse newline/semicolon rows and comma/space-separated numeric cells."""

    rows: list[list[float]] = []
    normalized = str(text).replace(";", "\n")
    for raw_row in normalized.splitlines():
        row = raw_row.strip()
        if not row:
            continue
        cells = row.replace(",", " ").split()
        try:
            rows.append([float(cell) for cell in cells])
        except ValueError as exc:
            raise ValueError(f"{name} contains a non-numeric value") from exc
    if not rows:
        raise ValueError(f"{name} cannot be empty")
    width = len(rows[0])
    if width == 0 or any(len(row) != width for row in rows):
        raise ValueError(f"{name} rows must have the same number of values")
    values = np.asarray(rows, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain finite values")
    return values


def _vector(text: str, name: str) -> np.ndarray:
    values = _matrix(str(text).replace(",", "\n"), name)
    if values.shape[1] != 1:
        return values.reshape(-1)
    return values[:, 0]


def _compact_array(value: Any, *, limit: int = 16) -> str:
    array = np.asarray(value)
    flat = array.reshape(-1)
    preview = flat[:limit]
    rendered = np.array2string(
        preview,
        precision=6,
        suppress_small=False,
        separator=", ",
        max_line_width=110,
    )
    suffix = " …" if flat.size > limit else ""
    return f"shape={tuple(array.shape)}  values={rendered}{suffix}"


class WeightLabPanel(QWidget):
    """Six bounded numerical tools behind one animated, extensible shell."""

    status_changed = Signal(str, str)
    open_in_workshop_requested = Signal(object)

    def __init__(self, manager, parent=None) -> None:
        super().__init__(parent)
        self.manager = manager
        self.sandbox_service = WeightSandboxService(manager)
        self.current_project: Path | None = None
        self._pending_project: Path | None = None
        self._project_change_pending = False
        self.current_tool_key = WEIGHT_TOOL_SPECS[0].key
        self._run_generation = 0
        self._busy = False
        self._busy_mode = ""
        self._sandbox_exists = False
        self._sandbox_buffers: dict[tuple[str, str], tuple[str, bool]] = {}
        self._fields: dict[str, dict[str, QWidget]] = {}
        self.launch_buttons: dict[str, QPushButton] = {}

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(12)
        intro = QLabel(
            "Six inspectable prototypes turn small, explicit numeric contracts into weights, "
            "recurrences, constrained fits, or next-sample decisions. Every result states its "
            "assurance level and limits."
        )
        intro.setWordWrap(True)
        intro.setObjectName("Muted")
        intro.setAccessibleName("Weight Lab scope")
        root.addWidget(intro)

        chooser = QHBoxLayout()
        chooser.addWidget(QLabel("Help me choose"))
        self.goal_chooser = QComboBox()
        self.goal_chooser.setAccessibleName("Weight Lab design goal")
        self.goal_chooser.addItem("Generate a small adapter from numeric context", "context_adapter")
        self.goal_chooser.addItem("Turn complete binary rules into exact weights", "binary_rules")
        self.goal_chooser.addItem("Explore a stable streaming recurrence", "streaming_sequence")
        self.goal_chooser.addItem("Fit data while respecting a known equation", "known_equation")
        self.goal_chooser.addItem("Create a fast numeric regression baseline", "instant_numeric_fit")
        self.goal_chooser.addItem("Choose the next expensive label", "choose_next_sample")
        self.goal_chooser.currentIndexChanged.connect(self._recommend_goal)
        chooser.addWidget(self.goal_chooser, 1)
        self.recommendation = QLabel()
        self.recommendation.setObjectName("Success")
        self.recommendation.setWordWrap(True)
        self.recommendation.setAccessibleName("Recommended Weight Lab tool")
        chooser.addWidget(self.recommendation, 2)
        root.addLayout(chooser)

        gallery = QWidget()
        gallery.setAccessibleName("Weight Lab tool launchers")
        grid = QGridLayout(gallery)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(12)
        for index, spec in enumerate(WEIGHT_TOOL_SPECS):
            button = QPushButton(f"{spec.title}\n\n{spec.concept}")
            button.setObjectName("WeightToolLauncher")
            button.setMinimumHeight(118)
            button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.setToolTip(f"Use when: {spec.use_when}")
            button.setAccessibleName(f"Open {spec.title}")
            button.setAccessibleDescription(
                f"{spec.concept} Use when: {spec.use_when}"
            )
            button.clicked.connect(partial(self.open_tool, spec.key, button))
            self.launch_buttons[spec.key] = button
            grid.addWidget(button, index // 3, index % 3)
        root.addWidget(_scroll(gallery), 1)

        self.tool_workspace = self._build_tool_workspace()
        self.reveal_host = ToolRevealHost(self.tool_workspace, self)
        self.reveal_host.opened.connect(self._reveal_opened)
        self.reveal_host.closed.connect(self._reveal_closed)
        self._select_tool(self.current_tool_key)
        self._recommend_goal()

    # ---- Workspace construction -------------------------------------------------

    def _build_tool_workspace(self) -> QWidget:
        workspace = QFrame()
        workspace.setObjectName("WeightToolWorkspace")
        layout = QVBoxLayout(workspace)
        layout.setContentsMargins(12, 10, 12, 12)
        layout.setSpacing(10)

        heading = QHBoxLayout()
        titles = QVBoxLayout()
        self.tool_title = QLabel()
        self.tool_title.setObjectName("PageTitle")
        self.tool_title.setAccessibleName("Opened Weight Lab tool")
        self.tool_maturity = QLabel()
        self.tool_maturity.setObjectName("Warning")
        self.tool_maturity.setAccessibleName("Tool maturity and assurance")
        titles.addWidget(self.tool_title)
        titles.addWidget(self.tool_maturity)
        heading.addLayout(titles, 1)
        self.close_button = QPushButton("Close")
        self.close_button.setAccessibleName("Close Weight Lab tool")
        self.close_button.clicked.connect(self.close_tool)
        heading.addWidget(self.close_button)
        layout.addLayout(heading)

        self.pipeline = QLabel()
        self.pipeline.setObjectName("Success")
        self.pipeline.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.pipeline.setWordWrap(True)
        self.pipeline.setAccessibleName("Current tool data flow")
        layout.addWidget(self.pipeline)

        self.inner_tabs = QTabWidget()
        self.inner_tabs.setAccessibleName("Weight Lab tool modes")
        self.guided_tab = self._build_guided_tab()
        self.sandbox_tab = self._build_sandbox_tab()
        self.info_tab = self._build_info_tab()
        self.inner_tabs.addTab(self.guided_tab, "Guided")
        self.inner_tabs.addTab(self.sandbox_tab, "Sandbox")
        self.inner_tabs.addTab(self.info_tab, "More Info")
        layout.addWidget(self.inner_tabs, 1)
        return workspace

    def _build_guided_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        self.decision_hint = QLabel()
        self.decision_hint.setWordWrap(True)
        self.decision_hint.setObjectName("Muted")
        self.decision_hint.setAccessibleName("Current tool decision hint")
        layout.addWidget(self.decision_hint)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        self.guided_stack = QStackedWidget()
        self.guided_stack.setAccessibleName("Current Weight Lab inputs")
        for spec in WEIGHT_TOOL_SPECS:
            self.guided_stack.addWidget(self._guided_page(spec.key))
        splitter.addWidget(self.guided_stack)

        result_panel = QWidget()
        result_layout = QVBoxLayout(result_panel)
        result_layout.setContentsMargins(6, 0, 0, 0)
        result_heading = QLabel("RESULT + DESIGN HINTS")
        result_heading.setObjectName("BrandTagline")
        result_layout.addWidget(result_heading)
        self.guided_result = QPlainTextEdit()
        self.guided_result.setReadOnly(True)
        self.guided_result.setAccessibleName("Weight Lab guided result")
        self.guided_result.setPlainText(
            "Load the bounded example, review its contract, then run the tool."
        )
        result_layout.addWidget(self.guided_result, 1)
        splitter.addWidget(result_panel)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 5)
        splitter.setSizes([470, 590])
        layout.addWidget(splitter, 1)

        actions = QHBoxLayout()
        self.run_button = QPushButton("Run bounded example")
        self.run_button.setObjectName("Primary")
        self.run_button.setAccessibleName("Run current Weight Lab guided tool")
        self.run_button.clicked.connect(self.run_guided)
        actions.addWidget(self.run_button)
        actions.addStretch(1)
        layout.addLayout(actions)
        return tab

    def _guided_page(self, key: str) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)
        form.setContentsMargins(8, 4, 12, 4)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        fields: dict[str, QWidget] = {}

        def text(name: str, value: str, accessible: str) -> QLineEdit:
            widget = QLineEdit(value)
            widget.setAccessibleName(accessible)
            fields[name] = widget
            return widget

        def multi(name: str, value: str, accessible: str) -> QPlainTextEdit:
            widget = QPlainTextEdit(value)
            widget.setMaximumHeight(94)
            widget.setAccessibleName(accessible)
            fields[name] = widget
            return widget

        def integer(
            name: str,
            value: int,
            maximum: int,
            accessible: str,
            *,
            minimum: int = 1,
        ) -> QSpinBox:
            widget = QSpinBox()
            widget.setRange(minimum, maximum)
            widget.setValue(value)
            widget.setAccessibleName(accessible)
            fields[name] = widget
            return widget

        def number(
            name: str,
            value: float,
            minimum: float,
            maximum: float,
            accessible: str,
            *,
            decimals: int = 6,
        ) -> QDoubleSpinBox:
            widget = QDoubleSpinBox()
            widget.setRange(minimum, maximum)
            widget.setDecimals(decimals)
            widget.setValue(value)
            widget.setAccessibleName(accessible)
            fields[name] = widget
            return widget

        if key == "meta_weight":
            form.addRow("Context vector", text("context", "1, 0.5, -0.25", "Numeric context vector"))
            form.addRow("Input columns", integer("input_dim", 4, 4096, "Adapter input columns"))
            form.addRow("Output rows", integer("output_dim", 3, 4096, "Adapter output rows"))
            form.addRow("Low-rank width", integer("rank", 2, 256, "Adapter rank"))
            form.addRow("Controller width", integer("hidden_dim", 8, 512, "Controller hidden width"))
            form.addRow("Adapter scale", number("scale", 0.5, 0.0, 1000.0, "Adapter scale"))
            form.addRow("Maximum ΔW norm", number("max_norm", 1.0, 0.000001, 1000.0, "Maximum adapter norm"))
            form.addRow("Seed", integer("seed", 47, 2_147_483_647, "Hypernetwork seed", minimum=0))
        elif key == "logic_compiler":
            form.addRow("Binary rows", multi("rows", "0,0\n0,1\n1,0\n1,1", "Complete binary truth table inputs"))
            form.addRow("Targets", multi("targets", "0\n1\n1\n0", "Truth table targets"))
        elif key == "recurrent_kernel":
            form.addRow("Scalar stream", text("stream", "1, 0.5, 0.25, 0.125, 0", "Recurrent input stream"))
            form.addRow("State size", integer("state_size", 8, 512, "Recurrent state size"))
            form.addRow("Kernel length", integer("kernel_length", 16, 100000, "Reference kernel length"))
            form.addRow("Contraction radius", number("contraction", 0.95, 0.000001, 0.999999, "Transition contraction radius"))
            form.addRow("Seed", integer("seed", 47, 2_147_483_647, "Recurrent controller seed", minimum=0))
        elif key == "constraint_optimizer":
            form.addRow("Observed x", text("coordinates", "0, 0.5, 1", "Observed coordinates"))
            form.addRow("Observed u(x)", text("observations", "1, 0.60653066, 0.36787944", "Observed field values"))
            form.addRow("Polynomial degree", integer("degree", 5, 32, "Polynomial degree"))
            form.addRow("u'' coefficient", number("a2", 0.0, -1000.0, 1000.0, "Second derivative coefficient"))
            form.addRow("u' coefficient", number("a1", 1.0, -1000.0, 1000.0, "First derivative coefficient"))
            form.addRow("u coefficient", number("a0", 1.0, -1000.0, 1000.0, "Value coefficient"))
            form.addRow("Source (scalar)", number("source", 0.0, -1e9, 1e9, "Differential equation source"))
            form.addRow("Physics weight", number("physics", 5.0, 0.0, 1e9, "Physics residual weight"))
            form.addRow("Boundary x (optional)", text("boundary_x", "0", "Boundary coordinates"))
            form.addRow("Boundary u(x)", text("boundary_y", "1", "Boundary values"))
            form.addRow("Boundary weight", number("boundary_weight", 10.0, 0.0, 1e9, "Boundary loss weight"))
            form.addRow("Collocation rows", integer("collocation", 64, 100000, "Physics collocation rows"))
            form.addRow("Ridge", number("ridge", 1e-10, 0.0, 1e6, "Constraint fit ridge", decimals=12))
        elif key == "matrix_inverter":
            form.addRow("Feature rows", multi("features", "0\n0.5\n1\n1.5\n2", "ELM feature matrix"))
            form.addRow("Target rows", multi("targets", "0\n0.479426\n0.841471\n0.997495\n0.909297", "ELM target matrix"))
            form.addRow("Hidden units", integer("hidden", 12, 8192, "ELM hidden units"))
            activation = QComboBox()
            activation.addItems(["tanh", "relu", "sigmoid"])
            activation.setAccessibleName("ELM hidden activation")
            fields["activation"] = activation
            form.addRow("Activation", activation)
            form.addRow("Ridge", number("ridge", 1e-6, 0.0, 1e6, "ELM ridge", decimals=12))
            form.addRow("Seed", integer("seed", 47, 2_147_483_647, "ELM seed", minimum=0))
        elif key == "uncertainty_sampler":
            form.addRow("Labeled features", multi("labeled_x", "0\n1", "Gaussian process labeled features"))
            form.addRow("Labeled targets", text("labeled_y", "0, 1", "Gaussian process labeled targets"))
            form.addRow("Candidate pool", multi("candidates", "-1\n0.5\n1\n2", "Gaussian process candidate pool"))
            form.addRow("Rows to select", integer("queries", 1, 100000, "Active learning query count"))
            form.addRow("Length scale", number("length", 1.0, 0.000001, 1e6, "RBF length scale"))
            form.addRow("Signal std", number("signal", 1.0, 0.000001, 1e6, "RBF signal standard deviation"))
            form.addRow("Noise std", number("noise", 0.0001, 0.0, 1e6, "Observation noise standard deviation", decimals=8))
        else:  # pragma: no cover - catalog invariant
            raise RuntimeError(f"No guided page for {key}")

        self._fields[key] = fields
        return _scroll(page)

    def _build_sandbox_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        self.sandbox_status = QLabel()
        self.sandbox_status.setWordWrap(True)
        self.sandbox_status.setAccessibleName("Current Weight Lab sandbox draft status")
        layout.addWidget(self.sandbox_status)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        self.sandbox_editor = CodeEditorPanel()
        self.sandbox_editor.editor.setAccessibleName("Current tool private sandbox editor")
        splitter.addWidget(self.sandbox_editor)
        self.sandbox_console = QPlainTextEdit()
        self.sandbox_console.setReadOnly(True)
        self.sandbox_console.setAccessibleName("Current tool sandbox output")
        self.sandbox_console.setPlainText(
            "Choose an active private project. Drafts run only in the constrained subprocess."
        )
        splitter.addWidget(self.sandbox_console)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([650, 420])
        layout.addWidget(splitter, 1)

        actions = QHBoxLayout()
        self.create_draft_button = QPushButton("Create private draft")
        self.create_draft_button.setAccessibleName("Create current tool sandbox draft")
        self.create_draft_button.clicked.connect(self.create_sandbox_draft)
        self.save_draft_button = QPushButton("Save")
        self.save_draft_button.setAccessibleName("Save current tool sandbox draft")
        self.save_draft_button.clicked.connect(self.save_sandbox_draft)
        self.run_sandbox_button = QPushButton("Run in sandbox")
        self.run_sandbox_button.setObjectName("Primary")
        self.run_sandbox_button.setAccessibleName("Run current tool private sandbox draft")
        self.run_sandbox_button.clicked.connect(self.run_sandbox)
        self.find_button = QPushButton("Find")
        self.find_button.setAccessibleName("Find text in current tool sandbox")
        self.find_button.clicked.connect(self.sandbox_editor.show_find)
        self.load_starter_button = QPushButton("Load starter (unsaved)")
        self.load_starter_button.setAccessibleName("Load current tool starter without saving")
        self.load_starter_button.clicked.connect(self.load_sandbox_starter)
        self.open_workshop_button = QPushButton("Open in Code Workshop")
        self.open_workshop_button.setAccessibleName("Open current tool draft in Code Workshop")
        self.open_workshop_button.clicked.connect(self.open_sandbox_in_workshop)
        for button in (
            self.create_draft_button,
            self.save_draft_button,
            self.run_sandbox_button,
            self.find_button,
            self.load_starter_button,
            self.open_workshop_button,
        ):
            actions.addWidget(button)
        actions.addStretch(1)
        layout.addLayout(actions)

        boundary = QLabel(
            "Sandbox boundary: a time-, path-, syntax-, and import-constrained subprocess for "
            "trusted learning code. It is not a VM and is not safe for hostile code. Built-ins "
            "remain immutable."
        )
        boundary.setWordWrap(True)
        boundary.setObjectName("Warning")
        boundary.setAccessibleName("Weight Lab sandbox security boundary")
        layout.addWidget(boundary)
        return tab

    def _build_info_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(10, 10, 10, 10)
        self.info_browser = QTextBrowser()
        self.info_browser.setOpenExternalLinks(False)
        self.info_browser.setReadOnly(True)
        self.info_browser.setAccessibleName("Current Weight Lab tool information")
        layout.addWidget(self.info_browser, 1)
        actions = QHBoxLayout()
        self.primary_button = QPushButton("Open primary research")
        self.primary_button.setAccessibleName("Open current tool primary research")
        self.primary_button.clicked.connect(self.open_primary_source)
        self.youtube_button = QPushButton("Search YouTube")
        self.youtube_button.setAccessibleName("Search YouTube for current tool")
        self.youtube_button.clicked.connect(self.open_youtube_search)
        actions.addWidget(self.primary_button)
        actions.addWidget(self.youtube_button)
        actions.addStretch(1)
        layout.addLayout(actions)
        return tab

    # ---- Tool state and reveal --------------------------------------------------

    def open_tool(self, tool_key: str, origin: QPushButton | None = None) -> None:
        if self._busy:
            self.status_changed.emit("Wait for the current Weight Lab run to finish.", "warning")
            return
        key = get_weight_tool_spec(tool_key).key
        self._select_tool(key)
        button = origin or self.launch_buttons[key]
        self.reveal_host.open_from(button, key, focus_target=self.run_button)

    def _recommend_goal(self, _index: int = 0) -> None:
        spec = recommend_weight_tool(str(self.goal_chooser.currentData()))
        self.recommendation.setText(f"Recommended: {spec.title} · {spec.use_when}")
        self.recommendation.setAccessibleDescription(
            f"{spec.title} is recommended. {spec.use_when}"
        )

    def _select_tool(self, tool_key: str) -> None:
        self._store_sandbox_buffer()
        spec = get_weight_tool_spec(tool_key)
        self.current_tool_key = spec.key
        self.guided_stack.setCurrentIndex(
            next(index for index, item in enumerate(WEIGHT_TOOL_SPECS) if item.key == spec.key)
        )
        self.tool_title.setText(spec.title)
        self.tool_maturity.setText(
            f"{spec.maturity} · assurance: {spec.assurance.replace('_', ' ')}"
        )
        self.close_button.setAccessibleName(
            f"Close {spec.title} and return to Weight Lab tools"
        )
        self.pipeline.setText(_PIPELINES[spec.key])
        self.pipeline.setAccessibleDescription(_PIPELINES[spec.key].replace("→", "then"))
        self.decision_hint.setText(
            f"Use when: {spec.use_when}\nAvoid when: {spec.avoid_when}"
        )
        self._update_more_info(spec)
        self._load_sandbox()

    def close_tool(self) -> None:
        if self._busy:
            self.status_changed.emit(
                "The tool will remain open until its bounded subprocess or calculation finishes.",
                "warning",
            )
            return
        self.reveal_host.close_reveal()

    def _reveal_opened(self, _tool_key: str) -> None:
        spec = get_weight_tool_spec(self.current_tool_key)
        self.status_changed.emit(f"{spec.title} opened.", "success")

    def _reveal_closed(self, _tool_key: str) -> None:
        self.status_changed.emit("Weight Lab tool gallery ready.", "muted")

    def set_reduced_motion(self, reduced: bool) -> None:
        self.reveal_host.set_reduced_motion(reduced)

    def _set_busy(self, busy: bool, mode: str = "") -> None:
        self._busy = bool(busy)
        self._busy_mode = mode if busy else ""
        self.run_button.setEnabled(not busy)
        self.run_sandbox_button.setEnabled(not busy and self._sandbox_exists)
        self.close_button.setEnabled(not busy)
        self.sandbox_editor.setEnabled(not busy)
        self.reveal_host.set_close_enabled(not busy)
        self._refresh_sandbox_actions()
        if not busy and self._project_change_pending:
            self._store_sandbox_buffer()
            self.current_project = self._pending_project
            self._pending_project = None
            self._project_change_pending = False
            self._load_sandbox()

    # ---- Guided execution -------------------------------------------------------

    def _guided_callable(self) -> Callable[[], Any]:
        key = self.current_tool_key
        fields = self._fields[key]

        def value(name: str) -> Any:
            widget = fields[name]
            if isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                return widget.value()
            if isinstance(widget, QComboBox):
                return widget.currentText()
            if isinstance(widget, (QLineEdit, QPlainTextEdit)):
                return widget.text() if isinstance(widget, QLineEdit) else widget.toPlainText()
            raise TypeError(f"Unsupported field {name}")

        if key == "meta_weight":
            arguments = dict(
                context=_vector(value("context"), "context vector"),
                input_dim=value("input_dim"),
                output_dim=value("output_dim"),
                rank=value("rank"),
                hidden_dim=value("hidden_dim"),
                scale=value("scale"),
                max_delta_norm=value("max_norm"),
                seed=value("seed"),
            )
            return partial(synthesize_low_rank_adapter, **arguments)
        if key == "logic_compiler":
            rows = _matrix(value("rows"), "binary rows")
            targets = _matrix(value("targets"), "targets")
            target_value = targets[:, 0] if targets.shape[1] == 1 else targets
            return partial(compile_truth_table, rows, target_value)
        if key == "recurrent_kernel":
            return partial(
                synthesize_recurrent_kernel,
                _vector(value("stream"), "scalar stream"),
                state_size=value("state_size"),
                kernel_length=value("kernel_length"),
                contraction=value("contraction"),
                seed=value("seed"),
            )
        if key == "constraint_optimizer":
            boundary_x_text = str(value("boundary_x")).strip()
            boundary_y_text = str(value("boundary_y")).strip()
            if bool(boundary_x_text) != bool(boundary_y_text):
                raise ValueError("Provide both boundary coordinates and boundary values, or neither")
            boundary_x = _vector(boundary_x_text, "boundary coordinates") if boundary_x_text else None
            boundary_y = _vector(boundary_y_text, "boundary values") if boundary_y_text else None
            return partial(
                fit_physics_constrained_polynomial,
                _vector(value("coordinates"), "observed coordinates"),
                _vector(value("observations"), "observations"),
                degree=value("degree"),
                second_derivative=value("a2"),
                first_derivative=value("a1"),
                value_coefficient=value("a0"),
                source=value("source"),
                physics_weight=value("physics"),
                boundary_coordinates=boundary_x,
                boundary_values=boundary_y,
                boundary_weight=value("boundary_weight"),
                collocation_count=value("collocation"),
                ridge=value("ridge"),
            )
        if key == "matrix_inverter":
            targets = _matrix(value("targets"), "target rows")
            target_value = targets[:, 0] if targets.shape[1] == 1 else targets
            return partial(
                fit_extreme_learning_machine,
                _matrix(value("features"), "feature rows"),
                target_value,
                hidden_units=value("hidden"),
                activation=value("activation"),
                ridge=value("ridge"),
                seed=value("seed"),
            )
        if key == "uncertainty_sampler":
            return partial(
                select_uncertain_candidates,
                _matrix(value("labeled_x"), "labeled features"),
                _vector(value("labeled_y"), "labeled targets"),
                _matrix(value("candidates"), "candidate pool"),
                query_count=value("queries"),
                length_scale=value("length"),
                signal_std=value("signal"),
                noise_std=value("noise"),
            )
        raise RuntimeError(f"Unsupported Weight Lab tool: {key}")

    def run_guided(self) -> None:
        if self._busy:
            return
        try:
            operation = self._guided_callable()
        except (TypeError, ValueError) as exc:
            self.guided_result.setPlainText(f"INPUT NEEDS ATTENTION\n=====================\n{exc}")
            self.status_changed.emit(f"Weight Lab input needs attention: {exc}", "warning")
            return
        self._run_generation += 1
        generation = self._run_generation
        key = self.current_tool_key
        self.guided_result.setPlainText("Running the bounded numerical contract…")
        self._set_busy(True, "guided")

        def success(result: Any) -> None:
            self._set_busy(False)
            if generation != self._run_generation or key != self.current_tool_key:
                return
            self.guided_result.setPlainText(self._format_result(result))
            self.status_changed.emit(
                f"{get_weight_tool_spec(key).title} completed with explicit assurance.",
                "success",
            )

        def failure(error: str) -> None:
            self._set_busy(False)
            summary = error.strip().splitlines()[-1] if error.strip() else "unknown error"
            self.guided_result.setPlainText(
                "RUN STOPPED SAFELY\n==================\n" + summary
            )
            self.status_changed.emit(f"Weight Lab run stopped safely: {summary}", "danger")

        run_in_background(self, operation, success, failure)

    @staticmethod
    def _format_result(result: Any) -> str:
        record: WeightToolRecord = result.record
        diagnostics = dict(record.diagnostics)
        lines = [
            "RESULT",
            "======",
            f"Algorithm: {record.algorithm}",
            f"Assurance: {record.assurance.replace('_', ' ')}",
            f"Input digest: {record.input_sha256[:16]}…",
        ]
        if record.seed is not None:
            lines.append(f"Seed: {record.seed}")
        lines.extend(["", "DIAGNOSTICS", "==========="])
        for name, value in diagnostics.items():
            label = name.replace("_", " ").capitalize()
            if isinstance(value, float):
                shown = f"{value:.8g}"
            else:
                shown = str(value)
            lines.append(f"{label}: {shown}")
        lines.extend(["", "OUTPUT ARRAYS", "============="])
        for descriptor in record.arrays:
            lines.append(
                f"{descriptor.name}: shape={descriptor.shape}, dtype={descriptor.dtype}, "
                f"sha256={descriptor.sha256[:12]}…"
            )
        previews: list[tuple[str, Any]] = []
        for name in (
            "delta",
            "output",
            "reference_kernel",
            "coefficients",
            "predictions",
            "selected_indices",
            "selected_candidates",
            "posterior_std",
            "output_weights",
        ):
            if hasattr(result, name):
                previews.append((name, getattr(result, name)))
        if previews:
            lines.extend(["", "PREVIEW", "======="])
            for name, value in previews[:3]:
                lines.append(f"{name}: {_compact_array(value)}")
        lines.extend(["", "HELPFUL HINTS", "============="])
        for hint in record.hints:
            lines.append(f"[{hint.severity.upper()}] {hint.message}")
        return "\n".join(lines)

    # ---- Private sandbox --------------------------------------------------------

    def set_project(self, project: str | Path | None) -> bool:
        candidate = Path(project).resolve(strict=False) if project else None
        if self._busy:
            self._pending_project = candidate
            self._project_change_pending = True
            return True
        self._store_sandbox_buffer()
        self.current_project = candidate
        self._load_sandbox()
        return True

    def _load_sandbox(self) -> None:
        project = self.current_project
        buffer_key = self._sandbox_buffer_key()
        if project is None:
            self._sandbox_exists = False
            buffered = self._sandbox_buffers.get(buffer_key)
            source, modified = buffered or (sandbox_template(self.current_tool_key), False)
            self.sandbox_editor.setPlainText(source)
            self.sandbox_editor.editor.document().setModified(modified)
            self.sandbox_status.setText(
                "No active private project. The starter is displayed in memory only; choose a "
                "project before creating a draft."
            )
            self.sandbox_console.setPlainText(
                "No project file has been created or executed."
            )
            self._refresh_sandbox_actions()
            return
        try:
            draft = self.sandbox_service.load(project, self.current_tool_key)
        except Exception as exc:
            self._sandbox_exists = False
            self.sandbox_status.setText(f"Sandbox unavailable: {exc}")
            self.sandbox_console.setPlainText("The sandbox path failed closed.")
            self._refresh_sandbox_actions()
            return
        self._sandbox_exists = draft.exists
        buffered = self._sandbox_buffers.get(buffer_key)
        source, modified = buffered or (draft.source, False)
        self.sandbox_editor.setPlainText(source)
        self.sandbox_editor.editor.document().setModified(modified)
        if modified:
            state = "Unsaved in-memory edits preserved"
        else:
            state = "Existing private draft" if draft.exists else "Starter preview; not written yet"
        self.sandbox_status.setText(
            f"{state}\n{draft.path}\nStarter SHA-256: {draft.template_sha256}"
        )
        self.sandbox_console.setPlainText(
            "Edit visible code, explicitly create/save it, then run the saved file through the "
            "constrained project subprocess."
        )
        self._refresh_sandbox_actions()

    def _sandbox_buffer_key(self) -> tuple[str, str]:
        project = str(self.current_project) if self.current_project is not None else "<no-project>"
        return (project.casefold(), self.current_tool_key)

    def _store_sandbox_buffer(self) -> None:
        if not hasattr(self, "sandbox_editor"):
            return
        self._sandbox_buffers[self._sandbox_buffer_key()] = (
            self.sandbox_editor.toPlainText(),
            self.sandbox_editor.editor.document().isModified(),
        )

    def _refresh_sandbox_actions(self) -> None:
        has_project = self.current_project is not None
        enabled = has_project and not self._busy
        self.create_draft_button.setEnabled(enabled and not self._sandbox_exists)
        self.save_draft_button.setEnabled(enabled and self._sandbox_exists)
        self.run_sandbox_button.setEnabled(enabled and self._sandbox_exists)
        self.open_workshop_button.setEnabled(enabled and self._sandbox_exists)
        self.load_starter_button.setEnabled(not self._busy)
        self.find_button.setEnabled(not self._busy)

    def create_sandbox_draft(self) -> bool:
        if self.current_project is None or self._busy:
            return False
        try:
            path = self.sandbox_service.create(
                self.current_project,
                self.current_tool_key,
                self.sandbox_editor.toPlainText(),
            )
        except Exception as exc:
            self.sandbox_console.setPlainText(f"Draft was not created:\n{exc}")
            self.status_changed.emit(f"Sandbox draft was not created: {exc}", "warning")
            self._load_sandbox()
            return False
        self._sandbox_exists = True
        self.sandbox_editor.editor.document().setModified(False)
        self._store_sandbox_buffer()
        self.sandbox_status.setText(f"Private draft created without overwriting anything:\n{path}")
        self.sandbox_console.setPlainText("Draft created. It has not been run.")
        self._refresh_sandbox_actions()
        self.status_changed.emit("Private Weight Lab sandbox draft created.", "success")
        return True

    def save_sandbox_draft(self) -> bool:
        if self.current_project is None or not self._sandbox_exists or self._busy:
            return False
        try:
            path = self.sandbox_service.save(
                self.current_project,
                self.current_tool_key,
                self.sandbox_editor.toPlainText(),
            )
        except Exception as exc:
            self.sandbox_console.setPlainText(f"Save failed safely:\n{exc}")
            self.status_changed.emit(f"Sandbox save failed: {exc}", "danger")
            return False
        self.sandbox_editor.editor.document().setModified(False)
        self._store_sandbox_buffer()
        self.sandbox_status.setText(f"Saved atomically:\n{path}")
        self.status_changed.emit("Weight Lab sandbox draft saved.", "success")
        return True

    def load_sandbox_starter(self) -> None:
        if self._busy:
            return
        self.sandbox_editor.setPlainText(sandbox_template(self.current_tool_key))
        self.sandbox_editor.editor.document().setModified(True)
        self._store_sandbox_buffer()
        self.sandbox_console.setPlainText(
            "Starter loaded into the editor only. The existing file, if any, was not changed."
        )

    def run_sandbox(self) -> None:
        if self.current_project is None or not self._sandbox_exists or self._busy:
            return
        if self.sandbox_editor.editor.document().isModified():
            self.sandbox_console.setPlainText(
                "Save the visible draft explicitly before running it. No automatic write occurred."
            )
            return
        project = self.current_project
        key = self.current_tool_key
        self._run_generation += 1
        generation = self._run_generation
        self._set_busy(True, "sandbox")
        self.sandbox_console.setPlainText("Running saved draft in constrained subprocess…")

        def success(result: Any) -> None:
            self._set_busy(False)
            if generation != self._run_generation or key != self.current_tool_key:
                return
            text = (
                f"Return code: {result.return_code}\n"
                f"Elapsed: {result.elapsed_seconds:.3f} seconds\n"
                f"Timed out: {result.timed_out}\n\n"
                f"STDOUT\n======\n{result.stdout or '(empty)'}\n\n"
                f"STDERR\n======\n{result.stderr or '(empty)'}"
            )
            self.sandbox_console.setPlainText(text)
            level = "success" if result.ok else "warning"
            self.status_changed.emit("Weight Lab sandbox run finished.", level)

        def failure(error: str) -> None:
            self._set_busy(False)
            summary = error.strip().splitlines()[-1] if error.strip() else "unknown error"
            self.sandbox_console.setPlainText("Sandbox stopped safely:\n" + summary)
            self.status_changed.emit(f"Sandbox stopped safely: {summary}", "danger")

        run_in_background(
            self,
            partial(self.sandbox_service.run, project, key),
            success,
            failure,
        )

    def open_sandbox_in_workshop(self) -> bool:
        if self.current_project is None or not self._sandbox_exists:
            return False
        try:
            path = self.sandbox_service.draft_path(
                self.current_project, self.current_tool_key
            )
        except Exception as exc:
            self.sandbox_console.setPlainText(f"Workshop handoff blocked:\n{exc}")
            return False
        if self.sandbox_editor.editor.document().isModified():
            self.sandbox_console.setPlainText(
                "Save the visible changes before opening the saved draft in Code Workshop."
            )
            return False
        self.open_in_workshop_requested.emit(path)
        return True

    # ---- Current information and exact links -----------------------------------

    @property
    def current_spec(self) -> WeightToolSpec:
        return get_weight_tool_spec(self.current_tool_key)

    @property
    def youtube_search_url(self) -> QUrl:
        spec = self.current_spec
        return QUrl(f"{_YOUTUBE_ROOT}?{urlencode({'search_query': spec.youtube_query})}")

    @property
    def primary_source_url(self) -> QUrl:
        return QUrl(self.current_spec.primary_source_url)

    def _update_more_info(self, spec: WeightToolSpec) -> None:
        project_boundary = (
            "A custom version can be edited as a visible file under the active private project. "
            "It never replaces the verified built-in and runs only through the constrained subprocess."
        )
        html = (
            f"<h2>{escape(spec.title)}</h2>"
            f"<p><b>What it does:</b> {escape(spec.concept)}</p>"
            f"<p><b>Current implementation:</b> {escape(spec.maturity)}. "
            f"Assurance is <code>{escape(spec.assurance)}</code>.</p>"
            f"<p><b>Formula:</b><br><code>{escape(spec.formula)}</code></p>"
            f"<p><b>Use when:</b> {escape(spec.use_when)}</p>"
            f"<p><b>Do not use when:</b> {escape(spec.avoid_when)}</p>"
            f"<p><b>Output:</b> {escape(spec.output)}</p>"
            f"<p><b>Sandbox extension boundary:</b> {escape(project_boundary)}</p>"
            f"<h3>Current learning links</h3>"
            f"<p><b>Primary source:</b> {escape(spec.primary_source_title)}<br>"
            "The YouTube button opens a live, pre-filled search so the available videos are current "
            "when you press it. Links open only after exact HTTPS validation.</p>"
            f"<p><small>Daedalus scope and links last reviewed {_REVIEWED_DATE}. "
            "A review date is not a claim that every new publication has been indexed.</small></p>"
        )
        self.info_browser.setHtml(html)
        self.info_browser.setAccessibleDescription(
            f"Explanation, formula, limits, primary source, and current video search for {spec.title}."
        )
        self.primary_button.setToolTip(spec.primary_source_title)
        self.youtube_button.setToolTip(f"Search YouTube for: {spec.youtube_query}")

    @staticmethod
    def _open_exact_url(candidate: QUrl, expected: QUrl, *, youtube: bool) -> bool:
        if (
            not candidate.isValid()
            or candidate.toString() != expected.toString()
            or candidate.scheme().casefold() != "https"
        ):
            return False
        if youtube and (
            candidate.host().casefold() not in _YOUTUBE_HOSTS
            or candidate.path() != "/results"
        ):
            return False
        if not youtube and candidate.host().strip() != expected.host().strip():
            return False
        return bool(QDesktopServices.openUrl(candidate))

    def open_primary_source(self) -> bool:
        expected = QUrl(self.current_spec.primary_source_url)
        return self._open_exact_url(self.primary_source_url, expected, youtube=False)

    def open_youtube_search(self) -> bool:
        expected = QUrl(
            f"{_YOUTUBE_ROOT}?"
            f"{urlencode({'search_query': self.current_spec.youtube_query})}"
        )
        return self._open_exact_url(self.youtube_search_url, expected, youtube=True)


__all__ = ["WeightLabPanel"]
