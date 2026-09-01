"""Angerona-inspired point-to-line-to-window reveal for tool workspaces."""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import (
    QAbstractAnimation,
    QEasingCurve,
    QEvent,
    QPointF,
    QRectF,
    Qt,
    QVariantAnimation,
    Signal,
)
from PySide6.QtGui import QColor, QKeyEvent, QPainter, QPalette, QPen
from PySide6.QtWidgets import QVBoxLayout, QWidget


@dataclass(frozen=True, slots=True)
class RevealGeometry:
    stage: str
    point: QPointF
    rect: QRectF


def calculate_reveal_geometry(
    progress: float,
    bounds: QRectF,
    normalized_origin: tuple[float, float],
) -> RevealGeometry:
    """Return deterministic point/line/box geometry for a reveal frame."""

    value = max(0.0, min(1.0, float(progress)))
    target = bounds.adjusted(14.0, 14.0, -14.0, -14.0)
    origin = QPointF(
        bounds.left() + bounds.width() * max(0.0, min(1.0, normalized_origin[0])),
        bounds.top() + bounds.height() * max(0.0, min(1.0, normalized_origin[1])),
    )
    if value <= 0.18:
        local = value / 0.18 if value else 0.0
        radius = 1.5 + 2.5 * local
        return RevealGeometry(
            "point",
            origin,
            QRectF(origin.x() - radius, origin.y() - radius, radius * 2.0, radius * 2.0),
        )
    line_width = max(2.0, target.width())
    if value <= 0.42:
        local = (value - 0.18) / 0.24
        width = 4.0 + (line_width - 4.0) * local
        return RevealGeometry(
            "line",
            origin,
            QRectF(origin.x() - width / 2.0, origin.y() - 1.5, width, 3.0),
        )

    local = (value - 0.42) / 0.58
    eased = 1.0 - (1.0 - local) ** 3
    line = QRectF(origin.x() - line_width / 2.0, origin.y() - 1.5, line_width, 3.0)
    left = line.left() + (target.left() - line.left()) * eased
    top = line.top() + (target.top() - line.top()) * eased
    width = line.width() + (target.width() - line.width()) * eased
    height = line.height() + (target.height() - line.height()) * eased
    return RevealGeometry("box", origin, QRectF(left, top, width, height))


class ToolRevealHost(QWidget):
    """Overlay that reveals one real tool workspace after a painted transition."""

    opened = Signal(str)
    closed = Signal(str)

    def __init__(self, content: QWidget, parent=None, *, reduced_motion: bool = False) -> None:
        super().__init__(parent)
        self.setObjectName("ToolRevealHost")
        self.setAccessibleName("Weight Lab revealed tool window")
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, False)
        self._content = content
        self._content.setParent(self)
        self._content.setVisible(False)
        self._content.setEnabled(False)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 26, 28, 26)
        layout.addWidget(self._content)
        self._progress = 0.0
        self._normalized_origin = (0.5, 0.5)
        self._origin_widget: QWidget | None = None
        self._focus_target: QWidget | None = None
        self._tool_key = ""
        self._opening = False
        self._close_enabled = True
        self._reduced_motion = bool(reduced_motion)
        self._animation = QVariantAnimation(self)
        self._animation.valueChanged.connect(self._set_progress)
        self._animation.finished.connect(self._animation_finished)
        if parent is not None:
            parent.installEventFilter(self)
            self.setGeometry(parent.rect())
        self.hide()

    @property
    def progress(self) -> float:
        return self._progress

    @property
    def is_open(self) -> bool:
        return self.isVisible() and self._progress >= 1.0 and self._content.isVisible()

    @property
    def is_animating(self) -> bool:
        return self._animation.state() == QAbstractAnimation.State.Running

    @property
    def close_enabled(self) -> bool:
        return self._close_enabled

    def set_close_enabled(self, enabled: bool) -> None:
        """Prevent a busy tool from being dismissed by Close or Escape."""

        self._close_enabled = bool(enabled)

    def set_reduced_motion(self, reduced: bool) -> None:
        self._reduced_motion = bool(reduced)
        if not self.is_animating:
            return
        self._animation.stop()
        self._set_progress(1.0 if self._opening else 0.0)
        self._animation_finished()

    def open_from(
        self,
        origin_widget: QWidget,
        tool_key: str,
        *,
        focus_target: QWidget | None = None,
    ) -> None:
        parent = self.parentWidget()
        if parent is None:
            return
        self._animation.stop()
        self.setGeometry(parent.rect())
        center = origin_widget.mapTo(parent, origin_widget.rect().center())
        width = max(1, parent.width())
        height = max(1, parent.height())
        self._normalized_origin = (center.x() / width, center.y() / height)
        self._origin_widget = origin_widget
        self._focus_target = focus_target
        self._tool_key = str(tool_key)
        self.setAccessibleName(f"Revealed {self._tool_key} tool window")
        self._content.hide()
        self._content.setEnabled(False)
        self._opening = True
        self.show()
        self.raise_()
        self.setFocus(Qt.FocusReason.OtherFocusReason)
        if self._reduced_motion:
            self._set_progress(1.0)
            self._animation_finished()
            return
        self._start_animation(self._progress if self._progress > 0.0 else 0.0, 1.0)

    def close_reveal(self) -> None:
        if not self.isVisible() or not self._close_enabled:
            return
        self._animation.stop()
        self._opening = False
        self._content.hide()
        self._content.setEnabled(False)
        if self._reduced_motion:
            self._set_progress(0.0)
            self._animation_finished()
            return
        self._start_animation(self._progress if self._progress > 0.0 else 1.0, 0.0)

    def _start_animation(self, start: float, end: float) -> None:
        self._animation.setStartValue(float(start))
        self._animation.setEndValue(float(end))
        self._animation.setDuration(260 if end > start else 220)
        self._animation.setEasingCurve(
            QEasingCurve.Type.OutCubic if end > start else QEasingCurve.Type.InCubic
        )
        self._animation.start()

    def _set_progress(self, value: object) -> None:
        self._progress = max(0.0, min(1.0, float(value)))
        self.update()

    def _animation_finished(self) -> None:
        if self._opening and self._progress >= 1.0:
            self._content.setEnabled(True)
            self._content.show()
            target = self._focus_target
            if target is not None and target.isEnabled():
                target.setFocus(Qt.FocusReason.OtherFocusReason)
            self.opened.emit(self._tool_key)
            return
        if not self._opening and self._progress <= 0.0:
            key = self._tool_key
            self.hide()
            origin = self._origin_widget
            if origin is not None and origin.isEnabled():
                origin.setFocus(Qt.FocusReason.OtherFocusReason)
            self.closed.emit(key)

    def eventFilter(self, watched: object, event: QEvent) -> bool:  # noqa: N802 - Qt API
        if watched is self.parentWidget() and event.type() in {
            QEvent.Type.Resize,
            QEvent.Type.Show,
        }:
            parent = self.parentWidget()
            if parent is not None:
                self.setGeometry(parent.rect())
                self.raise_()
        return super().eventFilter(watched, event)

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802 - Qt API
        if event.key() == Qt.Key.Key_Escape:
            self.close_reveal()
            event.accept()
            return
        super().keyPressEvent(event)

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt API
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        geometry = calculate_reveal_geometry(
            self._progress,
            QRectF(self.rect()),
            self._normalized_origin,
        )
        palette = self.palette()
        accent = QColor(palette.color(QPalette.ColorRole.Highlight))
        surface = QColor(palette.color(QPalette.ColorRole.Window))
        backdrop = QColor(palette.color(QPalette.ColorRole.Window))
        backdrop.setAlpha(218 if self._progress > 0.42 else 118)
        painter.fillRect(self.rect(), backdrop)
        if geometry.stage == "point":
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(accent)
            painter.drawEllipse(geometry.rect)
            return
        if geometry.stage == "line":
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(accent)
            painter.drawRoundedRect(geometry.rect, 1.5, 1.5)
            return
        painter.setPen(QPen(accent, 2.0))
        painter.setBrush(surface)
        painter.drawRoundedRect(geometry.rect, 10.0, 10.0)


__all__ = ["RevealGeometry", "ToolRevealHost", "calculate_reveal_geometry"]
