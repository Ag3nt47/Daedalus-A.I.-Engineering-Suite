"""Reusable native widgets shared by Daedalus workbench pages."""

from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any, Callable

from PySide6.QtCore import Qt, QThread, QTimer, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QGuiApplication
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)
from shiboken6 import isValid as is_qt_object_valid

from daedalus.gui.icons import semantic_icon


def repolish(widget: QWidget) -> None:
    """Re-evaluate object-name/property selectors after a semantic state change."""

    style = widget.style()
    style.unpolish(widget)
    style.polish(widget)
    widget.update()


def clear_layout(layout) -> None:
    while layout.count():
        item = layout.takeAt(0)
        child = item.widget()
        if child is not None:
            child.deleteLater()
        nested = item.layout()
        if nested is not None:
            clear_layout(nested)


class PageHeader(QWidget):
    def __init__(self, title: str, subtitle: str, icon: str, parent=None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        icon_label = QLabel()
        icon_label.setPixmap(semantic_icon(icon, size=30).pixmap(30, 30))
        icon_label.setAccessibleName(f"{title} icon")
        layout.addWidget(icon_label, 0, Qt.AlignmentFlag.AlignTop)
        labels = QVBoxLayout()
        labels.setContentsMargins(0, 0, 0, 0)
        labels.setSpacing(2)
        heading = QLabel(title)
        heading.setObjectName("PageTitle")
        heading.setAccessibleName(f"{title} page title")
        detail = QLabel(subtitle)
        detail.setObjectName("PageSubtitle")
        detail.setWordWrap(True)
        labels.addWidget(heading)
        labels.addWidget(detail)
        layout.addLayout(labels, 1)


class Card(QFrame):
    """A titled surface with a public body layout for page composition."""

    def __init__(
        self,
        title: str,
        subtitle: str = "",
        parent=None,
        *,
        accent: bool = False,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("Card")
        self.setProperty("accent", bool(accent))
        self.setAccessibleName(title)
        self.root = QVBoxLayout(self)
        self.root.setContentsMargins(14, 12, 14, 12)
        self.root.setSpacing(8)
        heading = QLabel(title)
        heading.setObjectName("SectionTitle")
        self.root.addWidget(heading)
        if subtitle:
            detail = QLabel(subtitle)
            detail.setObjectName("Muted")
            detail.setWordWrap(True)
            self.root.addWidget(detail)
        self.body = QVBoxLayout()
        self.body.setSpacing(8)
        self.root.addLayout(self.body, 1)

    def add_widget(self, widget: QWidget, stretch: int = 0) -> QWidget:
        self.body.addWidget(widget, stretch)
        return widget


class MetricTile(Card):
    def __init__(self, title: str, value: str = "—", detail: str = "", parent=None) -> None:
        super().__init__(title, parent=parent, accent=True)
        self.value_label = QLabel(value)
        self.value_label.setObjectName("MetricValue")
        self.detail_label = QLabel(detail)
        self.detail_label.setObjectName("Muted")
        self.detail_label.setWordWrap(True)
        self.body.addWidget(self.value_label)
        self.body.addWidget(self.detail_label)

    def set_value(self, value: Any, detail: str | None = None) -> None:
        self.value_label.setText(str(value))
        if detail is not None:
            self.detail_label.setText(detail)


class InfoPanel(QFrame):
    """Expandable theory/meaning panel with an accessible disclosure control."""

    def __init__(
        self,
        title: str,
        text: str,
        parent=None,
        *,
        expanded: bool = True,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("InfoPanel")
        self.setAccessibleName(f"Information: {title}")
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 10, 12, 10)
        root.setSpacing(6)
        self.toggle = QToolButton()
        self.toggle.setObjectName("NavButton")
        self.toggle.setText(title)
        self.toggle.setCheckable(True)
        self.toggle.setChecked(expanded)
        self.toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.toggle.setAccessibleName(f"Toggle {title} information")
        self.toggle.toggled.connect(self._set_expanded)
        root.addWidget(self.toggle)
        self.content = QLabel(text)
        self.content.setWordWrap(True)
        self.content.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.content.setAccessibleName(f"{title} information")
        root.addWidget(self.content)
        self._set_expanded(expanded)

    def _set_expanded(self, expanded: bool) -> None:
        self.content.setVisible(expanded)
        self.toggle.setArrowType(
            Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow
        )
        self.toggle.setAccessibleDescription(
            "Expanded" if expanded else "Collapsed; activate to show details"
        )


class PathField(QFrame):
    """Path editor with browse, copy, reveal, validity, and Git-boundary state."""

    path_changed = Signal(str)

    def __init__(
        self,
        title: str,
        path: str | Path = "",
        parent=None,
        *,
        manager=None,
        git_excluded: bool = True,
        mode: str = "directory",
        file_filter: str = "All files (*)",
        allow_missing: bool = False,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("PathField")
        self.title = str(title)
        self.manager = manager
        self.git_excluded = bool(git_excluded)
        self.mode = "file" if mode == "file" else "directory"
        self.file_filter = file_filter
        self.allow_missing = bool(allow_missing)
        self.setAccessibleName(f"{self.title} path field")

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 8, 10, 8)
        root.setSpacing(6)
        header = QHBoxLayout()
        label = QLabel(self.title)
        label.setObjectName("SectionTitle")
        header.addWidget(label)
        header.addStretch(1)
        self.state_label = QLabel()
        self.state_label.setAccessibleName(f"{self.title} Git and path state")
        header.addWidget(self.state_label)
        root.addLayout(header)

        row = QHBoxLayout()
        self.line_edit = QLineEdit(str(path))
        self._validation_timer = QTimer(self)
        self._validation_timer.setSingleShot(True)
        self._validation_timer.setInterval(180)
        self._validation_timer.timeout.connect(self.refresh_state)
        self._last_state_signature: tuple[bool, bool, bool, bool] | None = None
        self.line_edit.setClearButtonEnabled(True)
        self.line_edit.setAccessibleName(f"{self.title} path")
        self.line_edit.setPlaceholderText("Choose a file" if self.mode == "file" else "Choose a folder")
        self.line_edit.textChanged.connect(self._on_text_changed)
        row.addWidget(self.line_edit, 1)

        self.browse_button = QPushButton("Browse…")
        self.browse_button.setIcon(semantic_icon("folder", size=18))
        self.browse_button.setAccessibleName(f"Browse for {self.title}")
        self.browse_button.clicked.connect(self.browse)
        row.addWidget(self.browse_button)

        self.copy_button = QPushButton("Copy")
        self.copy_button.setIcon(semantic_icon("copy", size=18))
        self.copy_button.setAccessibleName(f"Copy {self.title} path")
        self.copy_button.clicked.connect(self.copy_path)
        row.addWidget(self.copy_button)

        self.reveal_button = QPushButton("Reveal")
        self.reveal_button.setIcon(semantic_icon("folder", size=18))
        self.reveal_button.setAccessibleName(f"Reveal {self.title} in file manager")
        self.reveal_button.clicked.connect(self.reveal_path)
        row.addWidget(self.reveal_button)
        root.addLayout(row)
        self.refresh_state()

    @property
    def path(self) -> Path:
        raw = self.line_edit.text().strip()
        return Path(raw).expanduser() if raw else Path()

    def set_path(self, value: str | Path) -> None:
        self.line_edit.setText(str(value))
        self._validation_timer.stop()
        self.refresh_state()
        if self.line_edit.isReadOnly():
            self.line_edit.setCursorPosition(0)

    def set_git_excluded(self, excluded: bool) -> None:
        self.git_excluded = bool(excluded)
        self.refresh_state()

    def is_valid(self) -> bool:
        raw = self.line_edit.text().strip()
        if not raw:
            return False
        candidate = Path(raw).expanduser()
        if self.mode == "file":
            return candidate.is_file()
        if candidate.is_dir():
            return True
        if not self.allow_missing or not candidate.is_absolute():
            return False
        anchor = Path(candidate.anchor)
        return bool(candidate.anchor) and anchor.is_dir()

    def _on_text_changed(self, text: str) -> None:
        # Interactive path checks may touch removable or network storage.
        # Coalesce typing bursts; programmatic set_path() remains immediate.
        self._validation_timer.start()
        self.path_changed.emit(text)

    def refresh_state(self) -> None:
        valid = self.is_valid()
        boundary = "GIT EXCLUDED" if self.git_excluded else "GIT TRACKED"
        location = "EXTERNAL" if self.git_excluded else "PUBLIC SOURCE"
        creatable = self.allow_missing and valid and not self.path.exists()
        exists = "READY TO CREATE" if creatable else "READY" if valid else "PATH NOT FOUND"
        has_text = bool(self.line_edit.text().strip())
        signature = (valid, self.git_excluded, creatable, has_text)
        if signature != self._last_state_signature:
            validity_changed = (
                self._last_state_signature is None
                or self._last_state_signature[0] != valid
            )
            self.setProperty("valid", valid)
            self.line_edit.setProperty("valid", valid)
            if validity_changed:
                for widget in (self, self.line_edit):
                    style = widget.style()
                    style.unpolish(widget)
                    style.polish(widget)
            object_name = "Success" if valid and self.git_excluded else "Warning"
            object_name_changed = self.state_label.objectName() != object_name
            self.state_label.setText(f"● {location} · {boundary} · {exists}")
            self.state_label.setObjectName(object_name)
            if object_name_changed:
                repolish(self.state_label)
            self.copy_button.setEnabled(has_text)
            self.reveal_button.setEnabled(valid)
            self._last_state_signature = signature
        self.state_label.setToolTip(
            "This location is outside the public source repository and must not be staged."
            if self.git_excluded
            else "This location may be included in source-control scans."
        )

    def browse(self) -> None:
        start = str(self.path if self.line_edit.text().strip() else Path.home())
        if self.mode == "file":
            selected, _ = QFileDialog.getOpenFileName(
                self, f"Choose {self.title}", start, self.file_filter
            )
        else:
            selected = QFileDialog.getExistingDirectory(self, f"Choose {self.title}", start)
        if selected:
            self.set_path(selected)

    def copy_path(self) -> None:
        QGuiApplication.clipboard().setText(self.line_edit.text().strip())
        self.state_label.setToolTip("Path copied to the clipboard.")

    def reveal_path(self) -> None:
        if not self.is_valid():
            self.refresh_state()
            return
        target = self.path
        if self.mode == "file":
            target = target.parent
        opener = getattr(self.manager, "open_in_file_manager", None)
        if callable(opener):
            try:
                opener(target)
                return
            except Exception:
                pass
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(target)))


class StatusStrip(QFrame):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._workspace_path = ""
        self._workspace_ready = True
        self.setObjectName("StatusStrip")
        self.setAccessibleName("Application status")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 5, 12, 5)
        layout.setSpacing(12)
        self.workspace = QLabel("● WORKSPACE —")
        self.workspace.setObjectName("Success")
        self.workspace.setAccessibleName("Workspace status")
        self.service = QLabel("SERVICES READY")
        self.service.setObjectName("Muted")
        self.service.setAccessibleName("Service status")
        self.motion = QLabel("MOTION ON")
        self.motion.setObjectName("Muted")
        self.motion.setAccessibleName("Motion preference")
        self.message = QLabel("Ready")
        self.message.setObjectName("Muted")
        self.message.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.message.setAccessibleName("Current operation status")
        layout.addWidget(self.workspace)
        layout.addWidget(self.service)
        layout.addWidget(self.motion)
        layout.addWidget(self.message, 1)

    def set_workspace(self, path: str | Path, ready: bool = True) -> None:
        self._workspace_path = str(path)
        self._workspace_ready = bool(ready)
        self._update_workspace_text()
        self.workspace.setObjectName("Success" if ready else "Warning")
        self._refresh_style(self.workspace)

    def _update_workspace_text(self) -> None:
        full = (
            f"● WORKSPACE {'READY' if self._workspace_ready else 'CHECK'} · "
            f"{self._workspace_path}"
        )
        available = max(170, round(max(1, self.width()) * 0.48))
        self.workspace.setMaximumWidth(available)
        self.workspace.setText(
            self.workspace.fontMetrics().elidedText(
                full,
                Qt.TextElideMode.ElideMiddle,
                available,
            )
        )
        self.workspace.setToolTip(full)
        self.workspace.setAccessibleDescription(full)

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt API
        super().resizeEvent(event)
        self._update_workspace_text()

    def set_message(self, message: str, level: str = "muted") -> None:
        self.message.setText(str(message))
        self.message.setObjectName(
            {"success": "Success", "warning": "Warning", "danger": "Danger"}.get(
                level, "Muted"
            )
        )
        self._refresh_style(self.message)

    def set_motion_reduced(self, reduced: bool) -> None:
        self.motion.setText("REDUCED MOTION" if reduced else "MOTION ON")

    @staticmethod
    def _refresh_style(widget: QWidget) -> None:
        repolish(widget)


class BackgroundTask(QThread):
    """Execute one blocking service call without touching widgets from the worker."""

    succeeded = Signal(object)
    failed = Signal(str)

    def __init__(self, function: Callable[[], Any], parent=None) -> None:
        super().__init__(parent)
        self._function = function

    def run(self) -> None:
        try:
            self.succeeded.emit(self._function())
        except Exception:
            self.failed.emit(traceback.format_exc())


def run_in_background(
    owner: QWidget,
    function: Callable[[], Any],
    on_success: Callable[[Any], None],
    on_failure: Callable[[str], None] | None = None,
) -> BackgroundTask:
    tasks: set[BackgroundTask] = getattr(owner, "_background_tasks", set())
    owner._background_tasks = tasks  # type: ignore[attr-defined]
    # The task deliberately has no QObject parent. A caller can force-delete a
    # window even after its normal close event has deferred shutdown; parenting
    # a live QThread to that window would then destroy the thread prematurely.
    task = BackgroundTask(function)
    tasks.add(task)

    def _owner_is_alive() -> bool:
        try:
            return is_qt_object_valid(owner)
        except RuntimeError:
            return False

    def _succeeded(result: Any) -> None:
        if _owner_is_alive():
            on_success(result)

    failure_callback = on_failure or (lambda _error: None)

    def _failed(error: str) -> None:
        if _owner_is_alive():
            failure_callback(error)

    task.succeeded.connect(_succeeded)
    task.failed.connect(_failed)

    def _finished() -> None:
        tasks.discard(task)
        task.deleteLater()

    task.finished.connect(_finished)
    task.start()
    return task


def human_bytes(value: int | float) -> str:
    amount = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(amount) < 1024.0 or unit == "TiB":
            return f"{amount:.1f} {unit}"
        amount /= 1024.0
    return f"{amount:.1f} TiB"
