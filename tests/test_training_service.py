from __future__ import annotations

import json
import platform
from pathlib import Path

import numpy as np

from daedalus.services.training import TrainingRequest, TrainingService
from daedalus.workspace.manager import WorkspaceManager
from daedalus.workspace.run_registry import RunRegistry


def test_training_service_records_holdouts_progress_and_contract(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    manager = WorkspaceManager(source, tmp_path / "workspace", tmp_path / "backup")
    manager.bootstrap()
    progress = []
    result = TrainingService(manager).run(
        TrainingRequest(
            project="guided-run",
            task="classification",
            epochs=3,
            batch_size=32,
            seed=17,
            early_stopping_patience=0,
        ),
        on_progress=progress.append,
    )

    assert result.status == "completed"
    assert result.epochs_completed == 3
    assert len(progress) == 3
    assert {"loss", "val_loss", "test_loss", "accuracy", "val_accuracy"} <= set(
        result.final_metrics
    )
    metadata_path = Path(result.checkpoint["metadata"])
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    contract = metadata["training_contract"]
    assert contract["task"] == "classification"
    assert contract["training"]["seed"] == 17
    assert contract["preparation"]["split_manifest"]["test_rows"] > 0
    assert contract["preparation"]["label_mapping"] == [
        {"original_label": 0, "encoded_label": 0},
        {"original_label": 1, "encoded_label": 1},
    ]
    assert contract["runtime"]["python"]["version"] == platform.python_version()
    assert contract["runtime"]["packages"]["numpy"] == np.__version__
    assert contract["runtime"]["project"]["captured"] is False
    registry = RunRegistry(manager.runs_dir / "runs.sqlite3")
    record = registry.get(result.run_id)
    assert record.status == "completed"
    assert record.config["runtime"] == contract["runtime"]
    metric_events = [event for event in registry.events(result.run_id) if event["event"] == "metrics"]
    assert len(metric_events) == 3


def test_training_request_rejects_split_without_training_rows() -> None:
    try:
        TrainingRequest(project="bad", validation_fraction=0.6, test_fraction=0.4)
    except ValueError as exc:
        assert "leave training rows" in str(exc)
    else:
        raise AssertionError("invalid held-out fractions were accepted")


def test_training_service_writes_immutable_project_run_evidence(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    manager = WorkspaceManager(source, tmp_path / "workspace", tmp_path / "backup")
    manager.bootstrap()
    project = manager.create_project("evidence-run")

    result = TrainingService(manager).run(
        TrainingRequest(
            project=project.name,
            task="regression",
            epochs=2,
            seed=23,
            early_stopping_patience=0,
        )
    )

    manifest_path = Path(result.checkpoint["run_manifest"])
    assert manifest_path == project / "runs" / f"{result.run_id}.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["kind"] == "daedalus-run-manifest"
    assert manifest["environment"]["source_sha256"]
    assert manifest["record"]["checkpoint"]["arrays_sha256"]
    assert manifest["record"]["checkpoint"]["metadata_sha256"]
    assert manifest["record"]["checkpoint"]["training_contract_sha256"]
    assert manifest["record"]["configuration"]["runtime"]["project"]["captured"] is True
    assert str(manager.workspace_root) not in manifest_path.read_text(encoding="utf-8")
