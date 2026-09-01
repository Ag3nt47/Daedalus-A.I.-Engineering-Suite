from __future__ import annotations

import json
from pathlib import Path

import pytest

from daedalus.engine import ModelEvaluator, evaluate_checkpoint
from daedalus.services.training import TrainingRequest, TrainingService
from daedalus.workspace.manager import WorkspaceManager


def _manager(tmp_path: Path, project: str) -> WorkspaceManager:
    source = tmp_path / "source"
    source.mkdir()
    manager = WorkspaceManager(source, tmp_path / "workspace", tmp_path / "backup")
    manager.bootstrap()
    manager.create_project(project)
    return manager


def test_classification_evaluation_is_rich_bounded_and_immutable(tmp_path: Path) -> None:
    project = "classification-project"
    manager = _manager(tmp_path, project)
    training = TrainingService(manager).run(
        TrainingRequest(
            project=project,
            task="classification",
            epochs=2,
            batch_size=32,
            seed=19,
            early_stopping_patience=0,
        )
    )

    report = evaluate_checkpoint(
        checkpoint=training.checkpoint["metadata"],
        manager=manager,
        acceptance_thresholds={"accuracy": 0.0, "macro_f1": 0.0},
        baseline={"accuracy": 0.0},
    )

    assert report["available"] is True
    assert report["task"] == "classification"
    assert report["split"] == "test"
    assert report["accepted"] is True
    assert {"accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"} <= set(report["metrics"])
    confusion = report["details"]["confusion_matrix"]
    assert len(confusion) == 2
    assert all(len(row) == 2 for row in confusion)
    assert sum(sum(row) for row in confusion) == report["sample_count"]
    per_class = report["details"]["per_class"]
    assert len(per_class) == 2
    assert {
        "original_label",
        "encoded_label",
        "support",
        "predicted_count",
        "precision",
        "recall",
        "f1",
    } <= set(per_class[0])

    first_path = Path(report["report_path"])
    assert first_path.parent == manager.projects_dir / project / "reports"
    first_bytes = first_path.read_bytes()
    stored = json.loads(first_bytes)
    assert stored["run_id"] == training.run_id
    assert "report_path" not in stored

    second = ModelEvaluator(manager).evaluate_checkpoint(
        training.checkpoint["arrays"],
        acceptance_thresholds={"accuracy": 1.1},
    )
    assert second["accepted"] is False
    assert Path(second["report_path"]) != first_path
    assert first_path.read_bytes() == first_bytes

    arrays_path = Path(training.checkpoint["arrays"])
    tampered = bytearray(arrays_path.read_bytes())
    tampered[-1] ^= 0x01
    arrays_path.write_bytes(tampered)
    report_count = len(list(first_path.parent.glob("evaluation-*.json")))
    with pytest.raises(ValueError, match="checksum"):
        ModelEvaluator(manager).evaluate_checkpoint(training.checkpoint["metadata"])
    assert len(list(first_path.parent.glob("evaluation-*.json"))) == report_count


def test_regression_evaluation_reports_residuals_and_lower_is_better_gates(
    tmp_path: Path,
) -> None:
    project = "regression-project"
    manager = _manager(tmp_path, project)
    training = TrainingService(manager).run(
        TrainingRequest(
            project=project,
            task="regression",
            epochs=2,
            batch_size=32,
            seed=23,
            early_stopping_patience=0,
        )
    )

    report = ModelEvaluator(manager).evaluate_checkpoint(
        training.checkpoint["metadata"],
        split="validation",
        acceptance_thresholds={"rmse": 1_000_000.0, "mae": 1_000_000.0},
        baseline={"rmse": 1_000_000.0},
    )

    assert report["task"] == "regression"
    assert report["split"] == "validation"
    assert report["accepted"] is True
    assert {"mse", "rmse", "mae", "r2"} == set(report["metrics"])
    residuals = report["details"]["residual_summary"]
    assert residuals["count"] == report["sample_count"]
    assert {
        "minimum",
        "maximum",
        "mean",
        "standard_deviation",
        "q05",
        "q25",
        "median",
        "q75",
        "q95",
    } <= set(residuals)
    rmse_comparison = next(item for item in report["comparisons"] if item["metric"] == "rmse")
    assert rmse_comparison["direction"] == "lower"
    assert rmse_comparison["threshold_passed"] is True
    assert rmse_comparison["baseline_passed"] is True


def test_evaluator_rejects_executable_or_untracked_artifacts(tmp_path: Path) -> None:
    project = "safe-project"
    manager = _manager(tmp_path, project)
    checkpoint_dir = manager.checkpoints_dir / project
    checkpoint_dir.mkdir()
    executable = checkpoint_dir / "unsafe.pkl"
    executable.write_bytes(b"not a checkpoint")

    with pytest.raises(ValueError, match="Only Daedalus JSON metadata or NPZ"):
        ModelEvaluator(manager).evaluate_checkpoint(executable)
