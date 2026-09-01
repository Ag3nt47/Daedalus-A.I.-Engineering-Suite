"""Capability-adaptive interactive 3D neural-network architecture viewer."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

from PySide6.QtCore import QLineF, QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QGuiApplication, QLinearGradient, QPainter, QPen
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from daedalus.gui.theme import reduced_motion


@dataclass(frozen=True, slots=True)
class RenderCapability:
    mode: str
    description: str
    recommended_quality: str
    animation_available: bool
    platform: str


def detect_render_capability() -> RenderCapability:
    """Choose a conservative visual tier without probing or allocating GPU resources."""

    application = QGuiApplication.instance()
    platform = application.platformName().casefold() if application is not None else "unknown"
    forced_software = os.getenv("QT_OPENGL", "").casefold() in {"software", "angle"}
    headless = platform in {"minimal", "offscreen", "vnc"}
    cpu_count = os.cpu_count() or 1
    available_memory = 0
    try:
        import psutil

        available_memory = int(psutil.virtual_memory().available)
    except (ImportError, OSError, ValueError):
        pass
    low_resources = cpu_count <= 2 or (available_memory and available_memory < 2 * 1024**3)
    if headless:
        return RenderCapability(
            "static-fallback",
            "Static software renderer · animation disabled for this display session",
            "low",
            False,
            platform,
        )
    if forced_software or low_resources:
        return RenderCapability(
            "software-compatible",
            "Interactive software-compatible renderer · adaptive detail",
            "medium" if not low_resources else "low",
            not reduced_motion(),
            platform,
        )
    return RenderCapability(
        "interactive",
        "Interactive native renderer · adaptive high detail",
        "high",
        not reduced_motion(),
        platform,
    )


@dataclass(frozen=True, slots=True)
class RenderStatistics:
    layer_count: int
    represented_neurons: int
    visible_nodes: int
    visible_connections: int
    quality: str


_QUALITY_LIMITS = {
    "low": (7, 90),
    "medium": (12, 220),
    "high": (18, 480),
}
_SCENE_CONNECTION_BUDGETS = {
    "low": 600,
    "medium": 1_600,
    "high": 3_200,
}


class NeuralNetwork3DCanvas(QWidget):
    """Project dense networks into an interactive perspective scene with QPainter."""

    layer_selected = Signal(int)

    def __init__(self, parent=None, *, capability: RenderCapability | None = None) -> None:
        super().__init__(parent)
        self.capability = capability or detect_render_capability()
        self._sizes = (2, 4, 1)
        self._quality = self.capability.recommended_quality
        self._yaw = -0.28
        self._pitch = 0.14
        self._zoom = 1.0
        self._selected_layer = 0
        self._drag_origin: QPointF | None = None
        self._dragged = False
        self._interaction_active = False
        self._animation_requested = False
        self._last_nodes: list[tuple[QPointF, float, int]] = []
        self._node_cache_key: tuple[tuple[int, ...], str] | None = None
        self._node_cache: list[list[tuple[float, float, float]]] = []
        self._animation = QTimer(self)
        # 20 fps keeps the optional ambient rotation smooth without competing
        # with editors and training updates for the GUI thread.
        self._animation.setInterval(50)
        self._animation.timeout.connect(self._animate)
        # Let compact windows allocate a smaller canvas without forcing the
        # tab contents beyond their parent; expanding layouts still give the
        # renderer all remaining space.
        self.setMinimumHeight(180)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)
        self.setAccessibleName("Interactive 3D neural network model")
        self.setAccessibleDescription(
            "Rotate with drag or arrow keys, zoom with the wheel or plus and minus, "
            "select a layer, and press R to reset the camera."
        )
        self.setToolTip(
            "Drag to rotate · wheel to zoom · select a layer · double-click or press R to reset"
        )
        self._update_accessible_summary()

    @property
    def architecture(self) -> tuple[int, ...]:
        return self._sizes

    @property
    def selected_layer(self) -> int:
        return self._selected_layer

    @property
    def quality(self) -> str:
        return self._quality

    def set_architecture(self, sizes: tuple[int, ...] | list[int]) -> None:
        values = tuple(int(value) for value in sizes)
        if len(values) < 2 or any(value <= 0 for value in values):
            raise ValueError("3D architecture requires at least two positive layer widths")
        if len(values) > 64:
            raise ValueError("3D architecture is limited to 64 layers")
        self._sizes = values
        self._selected_layer = min(self._selected_layer, len(values) - 1)
        self._node_cache_key = None
        self._update_accessible_summary()
        self.update()

    def set_quality(self, quality: str) -> None:
        value = str(quality).casefold()
        if value == "auto":
            value = self.capability.recommended_quality
        if value not in _QUALITY_LIMITS:
            raise ValueError(f"Unknown 3D render quality: {quality}")
        self._quality = value
        self._node_cache_key = None
        self._update_accessible_summary()
        self.update()

    def set_animation_enabled(self, enabled: bool) -> None:
        self._animation_requested = bool(enabled) and self.capability.animation_available
        self._sync_animation()

    def _sync_animation(self) -> None:
        should_run = self._animation_requested and self.isVisible()
        was_active = self._animation.isActive()
        if should_run:
            self._animation.start()
        else:
            self._animation.stop()
        if was_active != self._animation.isActive():
            # Motion uses a bounded scene; stopping restores the requested
            # static detail for one crisp final frame.
            self._node_cache_key = None
            self._update_accessible_summary()
            self.update()

    def showEvent(self, event) -> None:  # noqa: N802 - Qt API
        super().showEvent(event)
        self._sync_animation()

    def hideEvent(self, event) -> None:  # noqa: N802 - Qt API
        self._animation.stop()
        self._interaction_active = False
        self._node_cache_key = None
        super().hideEvent(event)

    def reset_camera(self) -> None:
        self._yaw = -0.28
        self._pitch = 0.14
        self._zoom = 1.0
        self.update()

    def render_statistics(self) -> RenderStatistics:
        quality = self._render_quality()
        node_limit, _connection_limit = _QUALITY_LIMITS[quality]
        connection_limit = self._connection_limit_per_transition(quality)
        visible = sum(min(width, node_limit) for width in self._sizes)
        connections = sum(
            min(min(left, node_limit) * min(right, node_limit), connection_limit)
            for left, right in zip(self._sizes, self._sizes[1:])
        )
        return RenderStatistics(
            len(self._sizes),
            sum(self._sizes),
            visible,
            connections,
            quality,
        )

    def _render_quality(self) -> str:
        if self._animation.isActive() or self._interaction_active:
            return "low"
        return self._quality

    def _connection_limit_per_transition(self, quality: str | None = None) -> int:
        quality = quality or self._render_quality()
        transitions = max(1, len(self._sizes) - 1)
        per_transition = _QUALITY_LIMITS[quality][1]
        scene_budget = _SCENE_CONNECTION_BUDGETS[quality]
        return min(per_transition, max(1, scene_budget // transitions))

    def _update_accessible_summary(self) -> None:
        stats = self.render_statistics()
        self.setAccessibleDescription(
            f"{stats.layer_count} layers representing {stats.represented_neurons:,} neurons; "
            f"showing {stats.visible_nodes} sampled nodes at {stats.quality} detail. "
            "Rotate with drag or arrow keys, zoom with the wheel or plus and minus, "
            "and press R to reset."
        )

    def _nodes(self) -> list[list[tuple[float, float, float]]]:
        quality = self._render_quality()
        cache_key = (self._sizes, quality)
        if self._node_cache_key == cache_key:
            return self._node_cache
        node_limit, _connection_limit = _QUALITY_LIMITS[quality]
        layer_spacing = 2.65
        center = (len(self._sizes) - 1) * layer_spacing / 2.0
        result: list[list[tuple[float, float, float]]] = []
        for layer_index, width in enumerate(self._sizes):
            count = min(width, node_limit)
            columns = max(1, math.ceil(math.sqrt(count)))
            rows = math.ceil(count / columns)
            points: list[tuple[float, float, float]] = []
            for index in range(count):
                row, column = divmod(index, columns)
                items_in_row = min(columns, count - row * columns)
                y = (row - (rows - 1) / 2.0) * 0.78
                z = (column - (items_in_row - 1) / 2.0) * 0.78
                points.append((layer_index * layer_spacing - center, y, z))
            result.append(points)
        self._node_cache_key = cache_key
        self._node_cache = result
        return self._node_cache

    def _projector(self):
        """Precompute the camera transform once for an entire paint pass."""

        cos_yaw, sin_yaw = math.cos(self._yaw), math.sin(self._yaw)
        cos_pitch, sin_pitch = math.cos(self._pitch), math.sin(self._pitch)
        camera = 13.0
        center_x = self.width() / 2.0
        center_y = self.height() / 2.0
        base_scale = min(max(1, self.width()) / 15.5, max(1, self.height()) / 9.0)
        base_scale *= self._zoom

        def project(point: tuple[float, float, float]) -> tuple[QPointF, float, float]:
            x, y, z = point
            x1 = x * cos_yaw + z * sin_yaw
            z1 = -x * sin_yaw + z * cos_yaw
            y2 = y * cos_pitch - z1 * sin_pitch
            z2 = y * sin_pitch + z1 * cos_pitch
            perspective = min(2.2, camera / max(3.0, camera + z2))
            scale = base_scale * perspective
            return QPointF(center_x + x1 * scale, center_y - y2 * scale), z2, perspective

        return project

    def _project(self, point: tuple[float, float, float]) -> tuple[QPointF, float, float]:
        return self._projector()(point)

    @staticmethod
    def _sample_connections(
        left: list[tuple[float, float, float]],
        right: list[tuple[float, float, float]],
        limit: int,
    ) -> list[tuple[tuple[float, float, float], tuple[float, float, float]]]:
        total = len(left) * len(right)
        if limit <= 0 or total == 0:
            return []
        if total <= limit:
            return [(source, target) for source in left for target in right]
        if limit == 1:
            indices = (total // 2,)
        else:
            indices = tuple(
                round(index * (total - 1) / (limit - 1)) for index in range(limit)
            )
        return [(left[index // len(right)], right[index % len(right)]) for index in indices]

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt API
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        background = QLinearGradient(0, 0, self.width(), self.height())
        background.setColorAt(0.0, QColor("#08121d"))
        background.setColorAt(0.55, QColor("#101b2a"))
        background.setColorAt(1.0, QColor("#071019"))
        painter.fillRect(self.rect(), background)

        layers = self._nodes()
        project = self._projector()
        projected = [[project(point) for point in layer] for layer in layers]
        projection_lookup = {
            point: projection
            for layer, projected_layer in zip(layers, projected)
            for point, projection in zip(layer, projected_layer)
        }
        connection_limit = self._connection_limit_per_transition()
        for layer_index, (left, right) in enumerate(zip(layers, layers[1:])):
            color = QColor("#45d4c7" if layer_index == self._selected_layer else "#557491")
            color.setAlpha(90 if layer_index == self._selected_layer else 38)
            painter.setPen(QPen(color, 1.25 if layer_index == self._selected_layer else 0.75))
            segments = [
                QLineF(projection_lookup[source][0], projection_lookup[target][0])
                for source, target in self._sample_connections(left, right, connection_limit)
            ]
            if segments:
                painter.drawLines(segments)

        nodes: list[tuple[float, QPointF, float, int]] = []
        for layer_index, layer in enumerate(projected):
            for screen, depth, perspective in layer:
                nodes.append((depth, screen, perspective, layer_index))
        self._last_nodes = [
            (screen, max(4.0, 7.5 * perspective), index)
            for _, screen, perspective, index in nodes
        ]
        for depth, screen, perspective, layer_index in sorted(
            nodes, key=lambda value: value[0], reverse=True
        ):
            del depth
            radius = max(4.0, 7.5 * perspective)
            selected = layer_index == self._selected_layer
            glow = QColor("#62f2dc" if selected else "#5da6d8")
            glow.setAlpha(48 if selected else 24)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(glow)
            painter.drawEllipse(screen, radius * 1.8, radius * 1.8)
            painter.setBrush(QColor("#4dd9c6" if selected else "#3f88b6"))
            painter.setPen(QPen(QColor("#d9fffa" if selected else "#a8d7f3"), 1.0))
            painter.drawEllipse(screen, radius, radius)
            painter.setBrush(QColor(255, 255, 255, 115))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(
                QPointF(screen.x() - radius * 0.27, screen.y() - radius * 0.31),
                radius * 0.22,
                radius * 0.22,
            )

        painter.setPen(QColor("#d8e8f2"))
        for index, layer in enumerate(layers):
            anchor, _depth, _perspective = project((layer[0][0], -3.1, 0.0))
            label = (
                "INPUT" if index == 0 else "OUTPUT" if index == len(layers) - 1 else f"HIDDEN {index}"
            )
            width = self._sizes[index]
            text = f"{label}\n{width:,} unit{'s' if width != 1 else ''}"
            rect = QRectF(anchor.x() - 70, anchor.y(), 140, 42)
            if index == self._selected_layer:
                painter.fillRect(rect.adjusted(-4, -2, 4, 2), QColor(38, 104, 116, 110))
            painter.drawText(rect, Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop, text)

        stats = self.render_statistics()
        painter.setPen(QColor("#8fa9ba"))
        painter.drawText(
            QRectF(12, 10, max(0, self.width() - 24), 24),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            (
                f"3D tensor flow · {stats.visible_nodes}/{stats.represented_neurons:,} sampled "
                f"neurons · {stats.visible_connections:,} visible connections"
            ),
        )
        painter.end()

    def _animate(self) -> None:
        if not self.isVisible():
            self._sync_animation()
            return
        self._yaw = (self._yaw + 0.006) % (math.pi * 2)
        self.update()

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt API
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_origin = event.position()
            self._dragged = False
            self._interaction_active = True
            self._node_cache_key = None
            self.setFocus(Qt.FocusReason.MouseFocusReason)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 - Qt API
        if self._drag_origin is not None and event.buttons() & Qt.MouseButton.LeftButton:
            delta = event.position() - self._drag_origin
            if abs(delta.x()) + abs(delta.y()) > 1:
                self._dragged = True
            self._yaw += delta.x() * 0.009
            self._pitch = max(-1.15, min(1.15, self._pitch + delta.y() * 0.007))
            self._drag_origin = event.position()
            self.update()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 - Qt API
        if event.button() == Qt.MouseButton.LeftButton and self._drag_origin is not None:
            if not self._dragged:
                self._select_nearest(event.position())
            self._drag_origin = None
            self._interaction_active = False
            self._node_cache_key = None
            self.update()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802 - Qt API
        if event.button() == Qt.MouseButton.LeftButton:
            self.reset_camera()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def wheelEvent(self, event) -> None:  # noqa: N802 - Qt API
        factor = 1.12 if event.angleDelta().y() > 0 else 1 / 1.12
        self._zoom = max(0.5, min(2.5, self._zoom * factor))
        self.update()
        event.accept()

    def keyPressEvent(self, event) -> None:  # noqa: N802 - Qt API
        key = event.key()
        if key in {Qt.Key.Key_Left, Qt.Key.Key_Right}:
            self._yaw += -0.10 if key == Qt.Key.Key_Left else 0.10
        elif key in {Qt.Key.Key_Up, Qt.Key.Key_Down}:
            self._pitch = max(
                -1.15,
                min(1.15, self._pitch + (-0.08 if key == Qt.Key.Key_Up else 0.08)),
            )
        elif key in {Qt.Key.Key_Plus, Qt.Key.Key_Equal}:
            self._zoom = min(2.5, self._zoom * 1.12)
        elif key == Qt.Key.Key_Minus:
            self._zoom = max(0.5, self._zoom / 1.12)
        elif key == Qt.Key.Key_R:
            self.reset_camera()
            event.accept()
            return
        else:
            super().keyPressEvent(event)
            return
        self.update()
        event.accept()

    def _select_nearest(self, position: QPointF) -> None:
        candidates = [
            (math.hypot(position.x() - point.x(), position.y() - point.y()), layer)
            for point, radius, layer in self._last_nodes
            if math.hypot(position.x() - point.x(), position.y() - point.y()) <= radius * 2
        ]
        if not candidates:
            return
        _distance, layer = min(candidates, key=lambda value: value[0])
        if layer != self._selected_layer:
            self._selected_layer = layer
            self.layer_selected.emit(layer)
            self.update()


class Model3DViewer(QWidget):
    """Controls and explanations surrounding the interactive 3D canvas."""

    def __init__(self, parent=None, *, capability: RenderCapability | None = None) -> None:
        super().__init__(parent)
        self.capability = capability or detect_render_capability()
        self.setAccessibleName("3D neural network architecture viewer")
        self.setAccessibleDescription(
            "Capability-adaptive model visualization with rendering controls and layer details."
        )
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(8)

        toolbar = QHBoxLayout()
        self.capability_label = QLabel(self.capability.description)
        self.capability_label.setObjectName(
            "Warning" if self.capability.mode == "static-fallback" else "Success"
        )
        self.capability_label.setAccessibleName("3D rendering capability")
        toolbar.addWidget(self.capability_label, 1)
        toolbar.addWidget(QLabel("Detail:"))
        self.quality = QComboBox()
        self.quality.addItem("Auto", "auto")
        self.quality.addItem("Low", "low")
        self.quality.addItem("Medium", "medium")
        self.quality.addItem("High", "high")
        self.quality.setAccessibleName("3D model rendering detail")
        self.quality.setAccessibleDescription(
            "Auto follows the detected rendering capability; fixed levels override visual detail."
        )
        toolbar.addWidget(self.quality)
        self.rotate = QCheckBox("Auto-rotate")
        self.rotate.setAccessibleName("Automatically rotate the 3D model")
        self.rotate.setAccessibleDescription(
            "Continuously rotates the model when animation is available."
        )
        self.rotate.setEnabled(self.capability.animation_available)
        self.rotate.setToolTip(
            "Animation is disabled when reduced-motion or static rendering is active."
        )
        toolbar.addWidget(self.rotate)
        self.reset_button = QPushButton("Reset view")
        self.reset_button.setAccessibleName("Reset 3D model camera")
        self.reset_button.setToolTip("Restore the default 3D rotation and zoom")
        toolbar.addWidget(self.reset_button)
        root.addLayout(toolbar)

        self.canvas = NeuralNetwork3DCanvas(self, capability=self.capability)
        root.addWidget(self.canvas, 1)
        self.layer_detail = QLabel()
        self.layer_detail.setWordWrap(True)
        self.layer_detail.setAccessibleName("Selected 3D architecture layer details")
        root.addWidget(self.layer_detail)
        self.guidance = QLabel(
            "Drag or use arrow keys to rotate · wheel or +/− to zoom · select a node to inspect "
            "its layer. Large layers are sampled visually; parameter totals remain exact."
        )
        self.guidance.setObjectName("Muted")
        self.guidance.setWordWrap(True)
        self.guidance.setAccessibleName("3D model interaction guidance")
        root.addWidget(self.guidance)

        self.quality.currentIndexChanged.connect(self._quality_changed)
        self.rotate.toggled.connect(self.canvas.set_animation_enabled)
        self.reset_button.clicked.connect(self.canvas.reset_camera)
        self.canvas.layer_selected.connect(self._show_layer)
        self._show_layer(0)

    def set_reduced_motion(self, reduced: bool) -> None:
        reduced = bool(reduced)
        if reduced:
            self.rotate.setChecked(False)
            self.canvas.set_animation_enabled(False)
        self.rotate.setEnabled(self.capability.animation_available and not reduced)
        self.rotate.setToolTip(
            "Animation is disabled by the reduced-motion preference."
            if reduced
            else "Continuously rotate the model while this control is enabled."
        )

    def set_architecture(self, sizes: tuple[int, ...] | list[int]) -> None:
        self.canvas.set_architecture(sizes)
        self._show_layer(self.canvas.selected_layer)

    def _quality_changed(self) -> None:
        self.canvas.set_quality(str(self.quality.currentData()))

    def _show_layer(self, index: int) -> None:
        sizes = self.canvas.architecture
        if not 0 <= index < len(sizes):
            return
        role = "Input" if index == 0 else "Output" if index == len(sizes) - 1 else "Hidden"
        incoming = "none" if index == 0 else f"{sizes[index - 1]:,} × {sizes[index]:,} weights"
        self.layer_detail.setText(
            f"Selected layer {index + 1}: {role} · {sizes[index]:,} units · incoming {incoming}"
        )


__all__ = [
    "Model3DViewer",
    "NeuralNetwork3DCanvas",
    "RenderCapability",
    "RenderStatistics",
    "detect_render_capability",
]
