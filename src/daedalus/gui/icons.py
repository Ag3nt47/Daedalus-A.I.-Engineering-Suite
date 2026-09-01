"""Original, dependency-free semantic icons drawn with Qt primitives."""

from __future__ import annotations

import math
from functools import lru_cache

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPainterPath, QPen, QPixmap

ICON_COLORS = {
    "mission": "#38bdf8",
    "developer": "#2dd4bf",
    "learn": "#a78bfa",
    "architecture": "#22d3ee",
    "calculator": "#fbbf24",
    "training": "#34d399",
    "workshop": "#60a5fa",
    "evaluate": "#c084fc",
    "guard": "#fb7185",
    "settings": "#94a3b8",
    "backup": "#2dd4bf",
    "push": "#38bdf8",
    "folder": "#f59e0b",
    "copy": "#a3e635",
    "wing": "#22d3a6",
}


def _line(painter: QPainter, *points: QPointF) -> None:
    for left, right in zip(points, points[1:]):
        painter.drawLine(left, right)


def _draw(kind: str, color: str, size: int) -> QPixmap:
    pixmap = QPixmap(size * 2, size * 2)
    pixmap.setDevicePixelRatio(2.0)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    pen = QPen(QColor(color), 1.8)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    painter.setPen(pen)
    painter.setBrush(Qt.BrushStyle.NoBrush)
    scale = size / 24.0
    painter.scale(scale, scale)

    if kind == "mission":
        painter.drawEllipse(QRectF(3, 3, 18, 18))
        painter.drawEllipse(QRectF(7, 7, 10, 10))
        _line(painter, QPointF(12, 12), QPointF(18, 7))
        painter.setBrush(QColor(color))
        painter.drawEllipse(QRectF(16.5, 5.5, 3, 3))
    elif kind == "developer":
        painter.drawRoundedRect(QRectF(4, 6, 16, 14), 3, 3)
        _line(painter, QPointF(12, 3), QPointF(12, 6))
        painter.drawEllipse(QRectF(10.5, 1.5, 3, 3))
        painter.setBrush(QColor(color))
        painter.drawEllipse(QRectF(7.5, 10, 2.5, 2.5))
        painter.drawEllipse(QRectF(14, 10, 2.5, 2.5))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        _line(painter, QPointF(8, 16), QPointF(16, 16))
        _line(painter, QPointF(2, 11), QPointF(4, 11))
        _line(painter, QPointF(20, 11), QPointF(22, 11))
    elif kind == "learn":
        left = QPainterPath(QPointF(12, 6))
        left.cubicTo(9, 3.8, 5.5, 4.3, 3.5, 6)
        left.lineTo(3.5, 18)
        left.cubicTo(6.5, 16.5, 9.5, 17, 12, 19)
        right = QPainterPath(QPointF(12, 6))
        right.cubicTo(15, 3.8, 18.5, 4.3, 20.5, 6)
        right.lineTo(20.5, 18)
        right.cubicTo(17.5, 16.5, 14.5, 17, 12, 19)
        painter.drawPath(left)
        painter.drawPath(right)
        _line(painter, QPointF(12, 6), QPointF(12, 19))
    elif kind == "architecture":
        nodes = (QPointF(4, 12), QPointF(12, 5), QPointF(12, 19), QPointF(20, 12))
        _line(painter, nodes[0], nodes[1], nodes[3], nodes[2], nodes[0])
        _line(painter, nodes[1], nodes[2])
        for node in nodes:
            painter.drawEllipse(QRectF(node.x() - 2, node.y() - 2, 4, 4))
    elif kind == "calculator":
        painter.drawRoundedRect(QRectF(4, 2.5, 16, 19), 2.5, 2.5)
        painter.drawRoundedRect(QRectF(7, 5, 10, 4), 1, 1)
        for x in (8, 12, 16):
            for y in (13, 17):
                painter.drawEllipse(QRectF(x - 1, y - 1, 2, 2))
    elif kind == "training":
        _line(painter, QPointF(4, 19), QPointF(4, 5))
        _line(painter, QPointF(4, 19), QPointF(21, 19))
        curve = QPainterPath(QPointF(5, 17))
        curve.cubicTo(8, 16, 9, 11, 12, 12)
        curve.cubicTo(15, 13, 16, 6, 20, 5)
        painter.drawPath(curve)
        _line(painter, QPointF(17.5, 5), QPointF(20, 5), QPointF(20, 7.5))
    elif kind == "workshop":
        _line(painter, QPointF(9, 5), QPointF(4, 12), QPointF(9, 19))
        _line(painter, QPointF(15, 5), QPointF(20, 12), QPointF(15, 19))
        _line(painter, QPointF(13.5, 4), QPointF(10.5, 20))
    elif kind == "evaluate":
        painter.drawEllipse(QRectF(3, 3, 18, 18))
        _line(painter, QPointF(7, 13), QPointF(10.5, 16.5), QPointF(17.5, 8.5))
    elif kind == "guard":
        gate = QPainterPath(QPointF(12, 2.5))
        gate.lineTo(20, 6)
        gate.lineTo(18.5, 15.5)
        gate.cubicTo(17.5, 19, 14.5, 21, 12, 22)
        gate.cubicTo(9.5, 21, 6.5, 19, 5.5, 15.5)
        gate.lineTo(4, 6)
        gate.closeSubpath()
        painter.drawPath(gate)
        _line(painter, QPointF(8, 12), QPointF(11, 15), QPointF(16, 9))
    elif kind == "settings":
        for y, knob in ((6, 9), (12, 15), (18, 7)):
            _line(painter, QPointF(4, y), QPointF(20, y))
            painter.setBrush(QColor(color))
            painter.drawEllipse(QRectF(knob - 2, y - 2, 4, 4))
            painter.setBrush(Qt.BrushStyle.NoBrush)
    elif kind == "backup":
        arc = QRectF(3.5, 4, 17, 16)
        painter.drawArc(arc, 20 * 16, 270 * 16)
        _line(painter, QPointF(5, 4), QPointF(5, 9), QPointF(10, 8))
        _line(painter, QPointF(12, 7), QPointF(12, 16))
        _line(painter, QPointF(8.5, 12.5), QPointF(12, 16), QPointF(15.5, 12.5))
    elif kind == "push":
        _line(painter, QPointF(12, 20), QPointF(12, 5))
        _line(painter, QPointF(7, 10), QPointF(12, 5), QPointF(17, 10))
        painter.drawRoundedRect(QRectF(4, 15, 16, 6), 2, 2)
    elif kind == "folder":
        path = QPainterPath(QPointF(3, 7))
        path.lineTo(9, 7)
        path.lineTo(11, 9)
        path.lineTo(21, 9)
        path.lineTo(19, 19)
        path.lineTo(3, 19)
        path.closeSubpath()
        painter.drawPath(path)
    elif kind == "copy":
        painter.drawRoundedRect(QRectF(7, 7, 13, 13), 2, 2)
        painter.drawRoundedRect(QRectF(4, 4, 13, 13), 2, 2)
    elif kind == "wing":
        for direction in (-1, 1):
            wing = QPainterPath(QPointF(12, 5))
            wing.cubicTo(12 + direction * 2, 9, 12 + direction * 5, 12, 12 + direction * 9, 13)
            wing.cubicTo(12 + direction * 6, 17, 12 + direction * 3, 19, 12, 21)
            painter.drawPath(wing)
        painter.drawEllipse(QRectF(10, 2, 4, 4))
        _line(painter, QPointF(12, 6), QPointF(12, 18))
    else:
        painter.drawEllipse(QRectF(5, 5, 14, 14))
        for angle in range(0, 360, 90):
            rad = math.radians(angle)
            _line(
                painter,
                QPointF(12 + 4 * math.cos(rad), 12 + 4 * math.sin(rad)),
                QPointF(12 + 8 * math.cos(rad), 12 + 8 * math.sin(rad)),
            )

    painter.end()
    return pixmap


@lru_cache(maxsize=128)
def semantic_icon(kind: str, color: str | None = None, size: int = 22) -> QIcon:
    tone = color or ICON_COLORS.get(kind, "#e2e8f0")
    return QIcon(_draw(kind, tone, max(12, int(size))))
