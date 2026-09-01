"""Focused offscreen coverage for the capability-adaptive architecture viewer."""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QImage, QPainter
from PySide6.QtWidgets import QApplication

from daedalus.gui.model_3d import (
    Model3DViewer,
    NeuralNetwork3DCanvas,
    RenderCapability,
    detect_render_capability,
)
from daedalus.gui.pages import ArchitectureBuilderPage


@pytest.fixture(scope="module")
def app() -> QApplication:
    application = QApplication.instance() or QApplication(["daedalus-model-3d-tests"])
    application.setApplicationName("Daedalus Model 3D Tests")
    return application


def _capability(
    *,
    mode: str = "interactive",
    quality: str = "high",
    animation: bool = True,
) -> RenderCapability:
    return RenderCapability(
        mode=mode,
        description=f"{mode} test renderer",
        recommended_quality=quality,
        animation_available=animation,
        platform="test",
    )


def test_offscreen_capability_uses_static_fallback(app: QApplication) -> None:
    capability = detect_render_capability()

    assert capability.platform == "offscreen"
    assert capability.mode == "static-fallback"
    assert capability.recommended_quality == "low"
    assert not capability.animation_available


def test_canvas_renders_to_an_offscreen_image(app: QApplication) -> None:
    canvas = NeuralNetwork3DCanvas(
        capability=_capability(mode="static-fallback", quality="low", animation=False)
    )
    canvas.resize(720, 420)
    canvas.set_architecture((64, 128, 32, 4))
    image = QImage(canvas.size(), QImage.Format.Format_ARGB32)
    image.fill(Qt.GlobalColor.transparent)
    painter = QPainter(image)
    try:
        assert painter.isActive()
        canvas.render(painter, QPoint())
    finally:
        painter.end()

    try:
        statistics = canvas.render_statistics()
        assert not image.isNull()
        assert image.pixelColor(2, 2).alpha() == 255
        assert image.pixelColor(2, 2) != image.pixelColor(image.width() - 3, image.height() - 3)
        assert len(canvas._last_nodes) == statistics.visible_nodes
        assert statistics.visible_nodes < statistics.represented_neurons
    finally:
        canvas.deleteLater()
        app.processEvents()


def test_node_and_connection_sampling_remain_bounded(app: QApplication) -> None:
    canvas = NeuralNetwork3DCanvas(capability=_capability())
    canvas.set_quality("high")
    canvas.set_architecture([10_000_000] * 64)
    statistics = canvas.render_statistics()
    layers = canvas._nodes()

    assert statistics.layer_count == 64
    assert statistics.represented_neurons == 640_000_000
    assert statistics.visible_nodes == 64 * 18
    assert all(len(layer) == 18 for layer in layers)
    assert statistics.visible_connections <= 3_200
    assert canvas._nodes() is layers

    left = [(float(index), 0.0, 0.0) for index in range(25)]
    right = [(float(index), 1.0, 0.0) for index in range(31)]
    sampled = canvas._sample_connections(left, right, 37)
    assert len(sampled) == 37
    assert len(set(sampled)) == 37
    assert sampled[0] == (left[0], right[0])
    assert sampled[-1] == (left[-1], right[-1])
    assert canvas._sample_connections(left, right, 0) == []

    with pytest.raises(ValueError, match="64 layers"):
        canvas.set_architecture([1] * 65)
    with pytest.raises(ValueError, match="positive layer widths"):
        canvas.set_architecture([2, 0, 1])
    canvas.deleteLater()
    app.processEvents()


def test_viewer_controls_follow_capability_and_are_accessible(app: QApplication) -> None:
    static_viewer = Model3DViewer(
        capability=_capability(mode="static-fallback", quality="low", animation=False)
    )
    try:
        assert static_viewer.accessibleName() == "3D neural network architecture viewer"
        assert static_viewer.capability_label.accessibleName() == "3D rendering capability"
        assert static_viewer.quality.accessibleName() == "3D model rendering detail"
        assert static_viewer.rotate.accessibleName() == "Automatically rotate the 3D model"
        assert static_viewer.reset_button.accessibleName() == "Reset 3D model camera"
        assert static_viewer.canvas.focusPolicy() == Qt.FocusPolicy.StrongFocus
        assert "sampled nodes" in static_viewer.canvas.accessibleDescription()
        assert static_viewer.canvas.quality == "low"
        assert not static_viewer.rotate.isEnabled()
        assert not static_viewer.canvas._animation.isActive()

        high_index = static_viewer.quality.findData("high")
        static_viewer.quality.setCurrentIndex(high_index)
        assert static_viewer.canvas.quality == "high"

        static_viewer.canvas._yaw = 1.2
        static_viewer.canvas._pitch = -0.7
        static_viewer.canvas._zoom = 2.0
        static_viewer.reset_button.click()
        assert static_viewer.canvas._yaw == pytest.approx(-0.28)
        assert static_viewer.canvas._pitch == pytest.approx(0.14)
        assert static_viewer.canvas._zoom == pytest.approx(1.0)
    finally:
        static_viewer.deleteLater()
        app.processEvents()

    animated_viewer = Model3DViewer(capability=_capability(animation=True))
    try:
        assert animated_viewer.rotate.isEnabled()
        animated_viewer.rotate.setChecked(True)
        assert animated_viewer.canvas._animation_requested
        assert not animated_viewer.canvas._animation.isActive()

        animated_viewer.show()
        app.processEvents()
        assert animated_viewer.canvas._animation.isActive()
        assert animated_viewer.canvas.quality == "high"
        assert animated_viewer.canvas.render_statistics().quality == "low"

        animated_viewer.hide()
        app.processEvents()
        assert not animated_viewer.canvas._animation.isActive()
        assert animated_viewer.canvas.render_statistics().quality == "high"

        animated_viewer.show()
        app.processEvents()
        assert animated_viewer.canvas._animation.isActive()
        animated_viewer.set_reduced_motion(True)
        assert not animated_viewer.rotate.isChecked()
        assert not animated_viewer.rotate.isEnabled()
        assert not animated_viewer.canvas._animation.isActive()
        animated_viewer.set_reduced_motion(False)
        assert animated_viewer.rotate.isEnabled()
        animated_viewer.rotate.setChecked(False)
        assert not animated_viewer.canvas._animation.isActive()
    finally:
        animated_viewer.canvas.set_animation_enabled(False)
        animated_viewer.deleteLater()
        app.processEvents()


def test_architecture_builder_updates_3d_model_and_keeps_info_last(
    app: QApplication,
) -> None:
    page = ArchitectureBuilderPage(None)
    try:
        assert [page.tabs.tabText(index) for index in range(page.tabs.count())] == [
            "Tools",
            "3D Model",
            "Info",
        ]
        assert page.tabs.tabText(page.tabs.count() - 1) == "Info"
        assert page.model_3d_viewer.canvas.architecture == (784, 128, 64, 10)

        page.sizes.setText("12, 6, 3")
        assert page.validate_architecture()
        assert page.table.rowCount() == 2
        assert page.model_3d_viewer.canvas.architecture == (12, 6, 3)
        assert "12 units" in page.model_3d_viewer.layer_detail.text()

        page.sizes.setText("12, 0, 3")
        assert not page.validate_architecture()
        assert page.model_3d_viewer.canvas.architecture == (12, 6, 3)
        assert page.tabs.tabText(page.tabs.count() - 1) == "Info"
    finally:
        page.deleteLater()
        app.processEvents()
