"""GUI wiring coverage for held-out evaluation gates and baseline comparisons."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from daedalus.gui.pages import ModelEvaluatorPage
from daedalus.services.training import TrainingRequest, TrainingService
from daedalus.workspace.manager import WorkspaceManager


@pytest.fixture(scope="module")
def app() -> QApplication:
    application = QApplication.instance() or QApplication(["daedalus-evaluation-ui-test"])
    application.setApplicationName("Daedalus Evaluation UI Tests")
    return application


def test_evaluator_page_replays_heldout_data_with_explicit_gate(
    app: QApplication,
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    manager = WorkspaceManager(source, tmp_path / "workspace", tmp_path / "backup")
    manager.bootstrap()
    project = manager.create_project("evaluation-ui")
    training = TrainingService(manager).run(
        TrainingRequest(
            project=project.name,
            task="classification",
            epochs=1,
            seed=41,
            early_stopping_patience=0,
        )
    )

    page = ModelEvaluatorPage(manager)
    page.show()
    app.processEvents()
    try:
        assert page.evaluate_button.text() == "Replay held-out evaluation"
        assert page.evaluation_split.currentData() == "test"
        assert not page.acceptance_threshold.isEnabled()
        assert not page.compare_baseline.isEnabled()

        page.checkpoint.set_path(training.checkpoint["metadata"])
        page.acceptance_metric.setCurrentIndex(1)
        page.acceptance_threshold.setValue(0.0)
        page.compare_baseline.setChecked(True)
        page.baseline_value.setValue(0.0)
        assert page.acceptance_threshold.isEnabled()
        assert page.baseline_value.isEnabled()

        report = page._evaluator_call()
        assert report["split"] == "test"
        assert report["accepted"] is True
        comparison = next(
            item for item in report["comparisons"] if item["metric"] == "accuracy"
        )
        assert comparison["threshold"] == 0.0
        assert comparison["baseline"] == 0.0
        assert Path(report["report_path"]).parent == project / "reports"
    finally:
        page.close()
        page.deleteLater()
        app.processEvents()
