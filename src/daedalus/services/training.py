"""Application service for reproducible, validation-aware teaching runs.

The GUI is deliberately kept out of this module.  A caller supplies a frozen
request and receives JSON-friendly result metadata while raw examples remain in
the private dataset workspace.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import sys
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np

from daedalus import __version__ as DAEDALUS_VERSION
from daedalus.engine.datasets import make_regression, make_xor
from daedalus.engine.trainer import Callback, EarlyStopping, Trainer
from daedalus.engine.training_data import (
    DatasetAssessment,
    PreparedTrainingData,
    assess_training_data,
    evaluate_predictions,
    prepare_training_data,
)
from daedalus.layers import Linear, ReLU, Sequential, Tanh
from daedalus.losses import CrossEntropyLoss, MSELoss
from daedalus.optim import Adam
from daedalus.workspace.checkpoints import save_checkpoint
from daedalus.workspace.datasets import DatasetMetadata, DatasetService
from daedalus.workspace.manager import WorkspaceManager, safe_project_name
from daedalus.workspace.run_registry import RunRegistry

RequestedTask = Literal["auto", "classification", "regression"]
ProgressHandler = Callable[["EpochProgress"], None]


@dataclass(frozen=True, slots=True)
class TrainingRequest:
    """Complete, bounded configuration for one local training run."""

    project: str
    dataset: str | None = None
    task: RequestedTask = "auto"
    epochs: int = 500
    batch_size: int = 32
    learning_rate: float = 0.03
    seed: int = 47
    validation_fraction: float = 0.2
    test_fraction: float = 0.1
    standardize: bool = True
    early_stopping_patience: int = 25
    hidden_width: int | None = None

    def __post_init__(self) -> None:
        project = str(self.project).strip()
        dataset = None if self.dataset is None else str(self.dataset).strip() or None
        if not project:
            raise ValueError("A training project name is required.")
        if self.task not in {"auto", "classification", "regression"}:
            raise ValueError("task must be auto, classification, or regression")
        if not 1 <= int(self.epochs) <= 5_000:
            raise ValueError("epochs must be between 1 and 5,000")
        if not 1 <= int(self.batch_size) <= 1_000_000:
            raise ValueError("batch_size must be between 1 and 1,000,000")
        if not math.isfinite(float(self.learning_rate)) or not 0 < self.learning_rate <= 1:
            raise ValueError("learning_rate must be finite and in (0, 1]")
        if not 0 <= int(self.seed) <= 2_147_483_647:
            raise ValueError("seed must be between 0 and 2,147,483,647")
        if not 0 < float(self.validation_fraction) < 1:
            raise ValueError("validation_fraction must be between zero and one")
        if not 0 < float(self.test_fraction) < 1:
            raise ValueError("test_fraction must be between zero and one")
        if self.validation_fraction + self.test_fraction >= 1:
            raise ValueError("validation and test fractions must leave training rows")
        if not 0 <= int(self.early_stopping_patience) <= 5_000:
            raise ValueError("early_stopping_patience must be between 0 and 5,000")
        if self.hidden_width is not None and not 1 <= int(self.hidden_width) <= 4_096:
            raise ValueError("hidden_width must be between 1 and 4,096")
        object.__setattr__(self, "project", project)
        object.__setattr__(self, "dataset", dataset)
        object.__setattr__(self, "epochs", int(self.epochs))
        object.__setattr__(self, "batch_size", int(self.batch_size))
        object.__setattr__(self, "learning_rate", float(self.learning_rate))
        object.__setattr__(self, "seed", int(self.seed))
        object.__setattr__(self, "validation_fraction", float(self.validation_fraction))
        object.__setattr__(self, "test_fraction", float(self.test_fraction))
        object.__setattr__(self, "standardize", bool(self.standardize))
        object.__setattr__(self, "early_stopping_patience", int(self.early_stopping_patience))
        if self.hidden_width is not None:
            object.__setattr__(self, "hidden_width", int(self.hidden_width))


@dataclass(frozen=True, slots=True)
class EpochProgress:
    """One immutable epoch update suitable for journals or UI signals."""

    epoch: int
    total_epochs: int
    metrics: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TrainingResult:
    """Serializable summary of a completed local training run."""

    run_id: str
    status: str
    task: str
    dataset: str
    seed: int
    epochs_requested: int
    epochs_completed: int
    stop_reason: str
    final_metrics: dict[str, float]
    parameter_shapes: dict[str, list[int]]
    checkpoint: dict[str, str]
    training_data: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"available": True, **asdict(self)}


class _JournalCallback(Callback):
    def __init__(
        self,
        registry: RunRegistry,
        run_id: str,
        total_epochs: int,
        on_progress: ProgressHandler | None,
    ) -> None:
        self.registry = registry
        self.run_id = run_id
        self.total_epochs = total_epochs
        self.on_progress = on_progress

    def on_epoch_end(self, trainer: Trainer, logs: Mapping[str, Any]) -> None:
        epoch = int(logs["epoch"]) + 1
        metrics = {
            str(name): float(value)
            for name, value in logs.items()
            if name != "epoch" and isinstance(value, (int, float, np.number))
        }
        self.registry.record_metrics(self.run_id, {"epoch": float(epoch), **metrics})
        if self.on_progress is not None:
            self.on_progress(EpochProgress(epoch, self.total_epochs, metrics))


def _array_digest(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in arrays:
        array = np.ascontiguousarray(value)
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _contract_digest(contract: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        contract,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _layer_contract(model: Sequential) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for index, layer in enumerate(model):
        item: dict[str, Any] = {"index": index, "type": type(layer).__name__}
        if isinstance(layer, Linear):
            item.update(
                {
                    "in_features": layer.in_features,
                    "out_features": layer.out_features,
                    "bias": layer.bias is not None,
                }
            )
        result.append(item)
    return result


class TrainingService:
    """Analyze private numeric data and run a leak-free teaching experiment."""

    def __init__(self, manager: WorkspaceManager) -> None:
        self.manager = manager
        self.manager.bootstrap()

    def _runtime_evidence(self, project_name: str) -> tuple[dict[str, Any], Path | None]:
        """Return compact run provenance and the verified project, when one exists."""

        project = self.manager.projects_dir / project_name
        snapshot: dict[str, Any] | None = None
        resolved_project: Path | None = None
        if project.exists() or project.is_symlink():
            from daedalus.services.project_standards import ProjectStandardsService

            snapshot = ProjectStandardsService(self.manager).runtime_snapshot(project)
            resolved_project = project.resolve(strict=True)

        isolated = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
        evidence: dict[str, Any] = {
            "schema": 1,
            "kind": "daedalus-training-runtime",
            "captured_utc": (
                snapshot.get("captured_utc")
                if snapshot is not None
                else datetime.now(UTC).isoformat()
            ),
            "python": {
                "implementation": platform.python_implementation(),
                "version": platform.python_version(),
            },
            "packages": {
                "daedalus": DAEDALUS_VERSION,
                "numpy": np.__version__,
            },
            "platform": {
                "system": platform.system() or "unknown",
                "release": platform.release() or "unknown",
                "architecture": platform.machine() or "unknown",
            },
            "environment": {
                "isolated": (
                    bool(snapshot["isolated_environment"])
                    if snapshot is not None
                    else isolated
                ),
                "kind": (
                    str(snapshot["environment_kind"])
                    if snapshot is not None
                    else "virtual-environment" if isolated else "system-interpreter"
                ),
                "compute": (
                    str(snapshot["device_capability"])
                    if snapshot is not None
                    else "NumPy CPU"
                ),
            },
            "project": {
                "captured": snapshot is not None,
                "source_sha256": (
                    str(snapshot["source_sha256"]) if snapshot is not None else None
                ),
                "pyproject_sha256": (
                    snapshot.get("pyproject_sha256") if snapshot is not None else None
                ),
                "dependency_lock_sha256": (
                    snapshot.get("dependency_lock_sha256") if snapshot is not None else None
                ),
            },
        }
        return evidence, resolved_project

    def _source_data(
        self, request: TrainingRequest
    ) -> tuple[np.ndarray, np.ndarray, DatasetMetadata | None, tuple[str, ...], str]:
        if request.dataset:
            features, targets, metadata = DatasetService(self.manager).load(request.dataset)
            return (
                np.asarray(features, dtype=np.float64),
                np.asarray(targets, dtype=np.float64),
                metadata,
                metadata.feature_columns,
                metadata.name,
            )
        if request.task == "regression":
            features, targets = make_regression(200, 1, 1, noise=0.05, seed=request.seed)
            return features, targets, None, ("feature_0",), "built-in regression"
        features, targets = make_xor(200, noise=0.08, seed=request.seed)
        return features, targets, None, ("x1", "x2"), "built-in XOR"

    def analyze(
        self,
        *,
        dataset: str | None,
        task: RequestedTask = "auto",
        seed: int = 47,
    ) -> DatasetAssessment:
        """Inspect a registered dataset (or deterministic fixture) without training."""

        request = TrainingRequest(project="Teaching Lab", dataset=dataset, task=task, seed=seed)
        features, targets, _metadata, feature_names, _dataset_name = self._source_data(request)
        return assess_training_data(
            features,
            targets,
            task=request.task,
            feature_names=feature_names,
        )

    @staticmethod
    def _model_for(
        prepared: PreparedTrainingData, request: TrainingRequest
    ) -> tuple[Sequential, Any, int]:
        input_width = int(prepared.train_features.shape[1])
        hidden_width = request.hidden_width or min(128, max(8, input_width * 2))
        if prepared.assessment.task == "classification":
            output_width = len(prepared.label_mapping)
            model = Sequential(
                Linear(input_width, hidden_width, seed=request.seed),
                Tanh(),
                Linear(hidden_width, output_width, seed=request.seed + 1),
            )
            return model, CrossEntropyLoss(), hidden_width
        model = Sequential(
            Linear(input_width, hidden_width, seed=request.seed),
            ReLU(),
            Linear(hidden_width, 1, seed=request.seed + 1),
        )
        return model, MSELoss(), hidden_width

    @staticmethod
    def _final_metrics(
        trainer: Trainer,
        prepared: PreparedTrainingData,
    ) -> dict[str, float]:
        metrics: dict[str, float] = {}
        splits = (
            ("", prepared.train_features, prepared.train_targets),
            ("val_", prepared.validation_features, prepared.validation_targets),
            ("test_", prepared.test_features, prepared.test_targets),
        )
        for prefix, features, targets in splits:
            metrics[f"{prefix}loss"] = trainer.evaluate(features, targets)
            predictions = trainer.predict(features)
            evaluated = evaluate_predictions(
                targets,
                predictions,
                task=prepared.assessment.task,
            ).to_dict()
            metrics.update({f"{prefix}{name}": float(value) for name, value in evaluated.items()})
        if not all(math.isfinite(value) for value in metrics.values()):
            raise FloatingPointError("Training produced a non-finite final metric.")
        return metrics

    def run(
        self,
        request: TrainingRequest,
        *,
        on_progress: ProgressHandler | None = None,
    ) -> TrainingResult:
        """Run one deterministic experiment and persist its audit/checkpoint artifacts."""

        features, targets, dataset_metadata, feature_names, dataset_name = self._source_data(request)
        prepared = prepare_training_data(
            features,
            targets,
            task=request.task,
            feature_names=feature_names,
            validation_fraction=request.validation_fraction,
            test_fraction=request.test_fraction,
            seed=request.seed,
            standardize=request.standardize,
        )
        project_name = safe_project_name(request.project)
        runtime_evidence, project_path = self._runtime_evidence(project_name)
        model, loss_function, hidden_width = self._model_for(prepared, request)
        architecture = _layer_contract(model)
        source_identity = {
            "name": dataset_name,
            "sha256": (
                dataset_metadata.sha256
                if dataset_metadata is not None
                else _array_digest(features, targets)
            ),
            "feature_columns": list(feature_names),
            "target_column": (
                dataset_metadata.target_column if dataset_metadata is not None else "target"
            ),
        }
        training_data = {
            "source": source_identity,
            "preparation": prepared.to_dict(),
        }
        run_config = {
            **asdict(request),
            "dataset": dataset_name,
            "resolved_task": prepared.assessment.task,
            "hidden_width": hidden_width,
            "engine": "daedalus.services.training.TrainingService",
            "training_data": training_data,
            "runtime": runtime_evidence,
        }
        registry = RunRegistry(self.manager.runs_dir / "runs.sqlite3")
        run_id = registry.create_run(project_name, dataset_name, run_config)
        registry.transition(run_id, "running")
        try:
            journal = _JournalCallback(registry, run_id, request.epochs, on_progress)
            callbacks: list[Callback] = [journal]
            early_stopping: EarlyStopping | None = None
            if request.early_stopping_patience:
                early_stopping = EarlyStopping(
                    monitor="val_loss",
                    patience=request.early_stopping_patience,
                    min_delta=1e-7,
                    mode="min",
                    restore_best=True,
                )
                callbacks.append(early_stopping)
            metric_functions = {}
            if prepared.assessment.task == "classification":
                metric_functions["accuracy"] = lambda prediction, truth: float(
                    np.mean(np.argmax(prediction, axis=1) == truth)
                )
            trainer = Trainer(
                model,
                loss_function,
                Adam(model.parameters(), lr=request.learning_rate),
                seed=request.seed,
                callbacks=callbacks,
                metrics=metric_functions,
            )
            history = trainer.fit(
                prepared.train_features,
                prepared.train_targets,
                epochs=request.epochs,
                batch_size=min(request.batch_size, len(prepared.train_features)),
                validation_data=(
                    prepared.validation_features,
                    prepared.validation_targets,
                ),
                seed=request.seed,
            )
            final_metrics = self._final_metrics(trainer, prepared)
            stop_reason = (
                "early_stopping"
                if early_stopping is not None and early_stopping.stopped_epoch is not None
                else "epochs_completed"
            )
            contract = {
                "schema": 1,
                "task": prepared.assessment.task,
                "architecture": architecture,
                "optimizer": {"type": "Adam", "learning_rate": request.learning_rate},
                "training": {
                    "seed": request.seed,
                    "batch_size": min(request.batch_size, len(prepared.train_features)),
                    "epochs_requested": request.epochs,
                    "epochs_completed": len(history.epochs),
                    "stop_reason": stop_reason,
                },
                "runtime": runtime_evidence,
                **training_data,
            }
            checkpoint_name = f"{project_name}-{prepared.assessment.task}-{run_id[:8]}"
            arrays_path, metadata_path = save_checkpoint(
                self.manager.checkpoints_dir / project_name,
                checkpoint_name,
                model.parameters(),
                architecture=architecture,
                metrics=final_metrics,
                training_contract=contract,
            )
            run_manifest: Path | None = None
            if project_path is not None:
                from daedalus.services.project_standards import ProjectStandardsService

                run_manifest = ProjectStandardsService(self.manager).write_run_manifest(
                    project_path,
                    run_id,
                    {
                        "status": "completed",
                        "project": project_name,
                        "dataset": dataset_name,
                        "task": prepared.assessment.task,
                        "configuration": run_config,
                        "result": {
                            "epochs_completed": len(history.epochs),
                            "stop_reason": stop_reason,
                            "metrics": final_metrics,
                        },
                        "checkpoint": {
                            "arrays_file": arrays_path.name,
                            "metadata_file": metadata_path.name,
                            "arrays_sha256": _file_digest(arrays_path),
                            "metadata_sha256": _file_digest(metadata_path),
                            "training_contract_sha256": _contract_digest(contract),
                        },
                    },
                )
            registry.transition(
                run_id,
                "completed",
                metrics=final_metrics,
                checkpoint=str(metadata_path),
            )
            return TrainingResult(
                run_id=run_id,
                status="completed",
                task=prepared.assessment.task,
                dataset=dataset_name,
                seed=request.seed,
                epochs_requested=request.epochs,
                epochs_completed=len(history.epochs),
                stop_reason=stop_reason,
                final_metrics=final_metrics,
                parameter_shapes={
                    name: list(array.shape) for name, array in model.state_dict().items()
                },
                checkpoint={
                    "arrays": str(arrays_path),
                    "metadata": str(metadata_path),
                    **(
                        {"run_manifest": str(run_manifest)}
                        if run_manifest is not None
                        else {}
                    ),
                },
                training_data=training_data,
            )
        except Exception as exc:
            try:
                registry.transition(
                    run_id,
                    "failed",
                    error=f"{type(exc).__name__}: {str(exc).strip()[:300]}",
                )
            except Exception:
                pass
            raise


__all__ = [
    "EpochProgress",
    "RequestedTask",
    "TrainingRequest",
    "TrainingResult",
    "TrainingService",
]
