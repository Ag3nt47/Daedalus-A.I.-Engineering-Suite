"""Run a repeatable offscreen Daedalus GUI performance smoke benchmark.

This is intentionally opt-in rather than a timing-sensitive unit test. It
prints JSON so local runs and CI artifacts can compare medians over time.
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

from PySide6.QtCore import QPoint, Qt  # noqa: E402
from PySide6.QtGui import QImage, QPainter  # noqa: E402
from PySide6.QtWidgets import QApplication, QWidget  # noqa: E402

from daedalus.gui.model_3d import (  # noqa: E402
    NeuralNetwork3DCanvas,
    RenderCapability,
)
from daedalus.workspace.manager import WorkspaceManager  # noqa: E402


def _milliseconds(function: Callable[[], object]) -> tuple[object, float]:
    started = time.perf_counter()
    result = function()
    return result, (time.perf_counter() - started) * 1_000


def _render_median(canvas: NeuralNetwork3DCanvas, samples: int = 5) -> float:
    durations: list[float] = []
    for _index in range(samples):
        image = QImage(canvas.size(), QImage.Format.Format_ARGB32)
        image.fill(Qt.GlobalColor.transparent)
        painter = QPainter(image)
        started = time.perf_counter()
        canvas.render(painter, QPoint())
        painter.end()
        durations.append((time.perf_counter() - started) * 1_000)
    return statistics.median(durations)


def main() -> int:
    application = QApplication.instance() or QApplication(["daedalus-gui-benchmark"])
    application.setApplicationName("Daedalus GUI Benchmark")
    results: dict[str, object] = {}

    with tempfile.TemporaryDirectory(prefix="daedalus-gui-benchmark-") as temporary:
        temporary_root = Path(temporary)
        manager = WorkspaceManager(
            REPOSITORY,
            temporary_root / "workspace",
            temporary_root / "backup",
        )
        manager.bootstrap()

        started = time.perf_counter()
        from daedalus.gui.main_window import MainWindow

        results["main_window_import_ms"] = round(
            (time.perf_counter() - started) * 1_000, 2
        )
        window_object, construct_ms = _milliseconds(lambda: MainWindow(manager))
        window = window_object
        assert isinstance(window, MainWindow)
        results["window_construct_ms"] = round(construct_ms, 2)
        window._live_scan_timer.stop()

        _, first_paint_ms = _milliseconds(lambda: (window.show(), application.processEvents()))
        results["first_paint_ms"] = round(first_paint_ms, 2)
        results["initial_widget_count"] = len(window.findChildren(QWidget))

        page_timings: dict[str, float] = {}
        for key in window.pages:
            if key == "mission":
                continue
            _, duration = _milliseconds(
                lambda page_key=key: (window.navigate(page_key), application.processEvents())
            )
            page_timings[key] = round(duration, 2)
        results["cold_page_ms"] = page_timings
        results["deferred_widget_count"] = len(window.findChildren(QWidget))

        developer = window.pages["developer"]
        calculator = window.pages["calculator"]
        architecture = window.pages["architecture"]
        deferred_timings: dict[str, float] = {}
        for name, function in (
            ("developer_setup", lambda: developer.setup_panel),
            ("weight_lab", lambda: calculator.weight_lab),
            ("advanced_calculators", lambda: calculator.advanced_panel),
            ("architecture_3d", lambda: architecture.model_3d_viewer),
        ):
            _, duration = _milliseconds(function)
            application.processEvents()
            deferred_timings[name] = round(duration, 2)
        results["deferred_tool_ms"] = deferred_timings
        results["fully_loaded_widget_count"] = len(window.findChildren(QWidget))

        from daedalus.gui.editor import CodeEditorPanel

        editor = CodeEditorPanel()
        large_source = "value = 47\n" * (2 * 1024 * 1024 // 11)
        _, editor_ms = _milliseconds(lambda: editor.setPlainText(large_source))
        results["editor_2mib_ms"] = round(editor_ms, 2)
        results["editor_large_highlighting_enabled"] = bool(
            editor.syntax_highlighting_enabled
        )
        editor.deleteLater()

        capability = RenderCapability(
            mode="benchmark",
            description="Offscreen benchmark renderer",
            recommended_quality="high",
            animation_available=True,
            platform="offscreen-benchmark",
        )
        canvas = NeuralNetwork3DCanvas(capability=capability)
        canvas.resize(960, 540)
        canvas.set_quality("high")
        canvas.set_architecture([64] * 64)
        canvas.show()
        application.processEvents()
        results["3d_static_high_median_ms"] = round(_render_median(canvas), 2)
        canvas.set_animation_enabled(True)
        application.processEvents()
        results["3d_motion_median_ms"] = round(_render_median(canvas), 2)
        results["3d_motion_quality"] = canvas.render_statistics().quality
        canvas.set_animation_enabled(False)
        canvas.hide()
        canvas.deleteLater()

        _, appearance_ms = _milliseconds(
            lambda: window.apply_appearance(
                window._theme_name,
                window._fixed_scale,
                window._auto_scale,
                window._reduced_motion,
            )
        )
        results["unchanged_appearance_ms"] = round(appearance_ms, 2)
        _, resize_ms = _milliseconds(
            lambda: (window.resize(1600, 1000), application.processEvents())
        )
        results["resize_loaded_window_ms"] = round(resize_ms, 2)

        deadline = time.monotonic() + 10.0
        while window._running_background_tasks() and time.monotonic() < deadline:
            time.sleep(0.005)
            application.processEvents()
        results["background_tasks_drained"] = not window._running_background_tasks()
        window.close()
        window.deleteLater()
        application.processEvents()

    print(json.dumps(results, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
