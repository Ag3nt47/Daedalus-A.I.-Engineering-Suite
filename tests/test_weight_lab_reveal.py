from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import QRectF
from PySide6.QtTest import QSignalSpy
from PySide6.QtWidgets import QApplication, QLineEdit, QPushButton, QVBoxLayout, QWidget

from daedalus.gui.reveal import ToolRevealHost, calculate_reveal_geometry


@pytest.fixture(scope="module")
def app() -> QApplication:
    application = QApplication.instance() or QApplication(["daedalus-reveal-tests"])
    application.setApplicationName("Daedalus Reveal Tests")
    return application


def test_reveal_geometry_has_deterministic_point_line_and_box_phases() -> None:
    bounds = QRectF(0.0, 0.0, 400.0, 240.0)
    origin = (0.25, 0.75)
    point = calculate_reveal_geometry(0.0, bounds, origin)
    point_end = calculate_reveal_geometry(0.18, bounds, origin)
    line = calculate_reveal_geometry(0.30, bounds, origin)
    line_end = calculate_reveal_geometry(0.42, bounds, origin)
    box = calculate_reveal_geometry(0.75, bounds, origin)
    box_end = calculate_reveal_geometry(1.0, bounds, origin)

    assert [item.stage for item in (point, point_end, line, line_end, box, box_end)] == [
        "point",
        "point",
        "line",
        "line",
        "box",
        "box",
    ]
    assert point.point.x() == pytest.approx(100.0)
    assert point.point.y() == pytest.approx(180.0)
    assert point.rect.width() == pytest.approx(3.0)
    assert point_end.rect.width() == pytest.approx(8.0)
    assert line.rect.height() == pytest.approx(3.0)
    assert 4.0 < line.rect.width() < line_end.rect.width()
    assert line_end.rect.width() == pytest.approx(372.0)
    assert box.rect.height() > line_end.rect.height()
    target = bounds.adjusted(14.0, 14.0, -14.0, -14.0)
    assert box_end.rect.left() == pytest.approx(target.left())
    assert box_end.rect.top() == pytest.approx(target.top())
    assert box_end.rect.width() == pytest.approx(target.width())
    assert box_end.rect.height() == pytest.approx(target.height())


def test_closing_uses_the_same_geometry_stages_in_reverse() -> None:
    bounds = QRectF(0.0, 0.0, 300.0, 180.0)
    forward = [
        calculate_reveal_geometry(progress, bounds, (0.5, 0.5)).stage
        for progress in (0.1, 0.3, 0.8)
    ]
    reverse = [
        calculate_reveal_geometry(progress, bounds, (0.5, 0.5)).stage
        for progress in (0.8, 0.3, 0.1)
    ]

    assert forward == ["point", "line", "box"]
    assert reverse == ["box", "line", "point"]
    clamped_start = calculate_reveal_geometry(-5.0, bounds, (-1.0, 2.0))
    clamped_end = calculate_reveal_geometry(5.0, bounds, (-1.0, 2.0))
    assert clamped_start.stage == "point"
    assert clamped_start.point.x() == pytest.approx(bounds.left())
    assert clamped_start.point.y() == pytest.approx(bounds.bottom())
    assert clamped_end.stage == "box"


def test_reduced_motion_open_close_is_synchronous_and_restores_focus(
    app: QApplication,
) -> None:
    parent = QWidget()
    parent.resize(520, 320)
    origin = QPushButton("Open tool", parent)
    origin.setGeometry(20, 20, 120, 36)
    content = QWidget()
    content_layout = QVBoxLayout(content)
    focus_target = QLineEdit()
    focus_target.setAccessibleName("First revealed control")
    content_layout.addWidget(focus_target)
    host = ToolRevealHost(content, parent, reduced_motion=True)
    opened: list[str] = []
    closed: list[str] = []
    host.opened.connect(opened.append)
    host.closed.connect(closed.append)

    parent.show()
    parent.activateWindow()
    origin.setFocus()
    app.processEvents()
    try:
        assert not host.isVisible()
        assert not content.isVisible()
        assert not content.isEnabled()

        host.open_from(origin, "meta_weight", focus_target=focus_target)
        app.processEvents()
        assert opened == ["meta_weight"]
        assert host.progress == pytest.approx(1.0)
        assert host.is_open
        assert content.isVisible()
        assert content.isEnabled()
        assert focus_target.hasFocus()

        host.close_reveal()
        app.processEvents()
        assert closed == ["meta_weight"]
        assert host.progress == pytest.approx(0.0)
        assert not host.isVisible()
        assert not content.isVisible()
        assert not content.isEnabled()
        assert origin.hasFocus()
    finally:
        parent.close()
        parent.deleteLater()
        app.processEvents()


def test_animated_open_and_reverse_close_reach_the_same_terminal_states(
    app: QApplication,
) -> None:
    parent = QWidget()
    parent.resize(520, 320)
    origin = QPushButton("Open animated tool", parent)
    origin.setGeometry(20, 20, 140, 36)
    content = QWidget()
    focus_target = QLineEdit(content)
    host = ToolRevealHost(content, parent, reduced_motion=False)
    opened: list[str] = []
    closed: list[str] = []
    opened_spy = QSignalSpy(host.opened)
    closed_spy = QSignalSpy(host.closed)
    host.opened.connect(opened.append)
    host.closed.connect(closed.append)

    parent.show()
    app.processEvents()
    try:
        host.open_from(origin, "logic_compiler", focus_target=focus_target)
        assert host.isVisible()
        assert not content.isVisible()
        assert opened_spy.wait(1_500)
        assert opened == ["logic_compiler"]
        assert host.is_open
        assert content.isVisible()

        host.close_reveal()
        assert host.isVisible()
        assert not content.isVisible()
        assert closed_spy.wait(1_500)
        assert closed == ["logic_compiler"]
        assert not host.isVisible()
        assert host.progress == pytest.approx(0.0)
    finally:
        parent.close()
        parent.deleteLater()
        app.processEvents()
