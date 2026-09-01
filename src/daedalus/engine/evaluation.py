"""Safe, reproducible evaluation of completed Daedalus training checkpoints.

The evaluator deliberately supports only the architecture and data contracts
written by :class:`daedalus.services.training.TrainingService`.  Checkpoints
remain data: model layers are reconstructed from a small allowlist and arrays
are loaded by the checksum-verifying NPZ checkpoint service with pickle
disabled.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import uuid
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np

from daedalus.core import Tensor, no_grad
from daedalus.engine.datasets import make_regression, make_xor
from daedalus.engine.training_data import (
    PreparedTrainingData,
    evaluate_predictions,
    prepare_training_data,
)
from daedalus.layers import Linear, ReLU, Sequential, Tanh
from daedalus.workspace.checkpoints import load_checkpoint
from daedalus.workspace.datasets import DatasetService
from daedalus.workspace.manager import WorkspaceManager, safe_project_name
from daedalus.workspace.run_registry import RunRecord, RunRegistry

HeldOutSplit = Literal["validation", "test"]
MetricDirection = Literal["higher", "lower"]

_SHA256 = re.compile(r"[0-9a-f]{64}")
_GENERATION_ARRAY = re.compile(r"^(?P<stem>.+)\.(?P<generation>[0-9a-f]{32})\.npz$")
_METRIC_DIRECTIONS: dict[str, MetricDirection] = {
    "accuracy": "higher",
    "balanced_accuracy": "higher",
    "macro_f1": "higher",
    "weighted_f1": "higher",
    "mse": "lower",
    "rmse": "lower",
    "mae": "lower",
    "r2": "higher",
}


class EvaluationError(ValueError):
    """Raised when an artifact cannot be evaluated without breaking its contract."""


@dataclass(frozen=True, slots=True)
class EvaluationLimits:
    """Resource ceilings for one local evaluation."""

    max_metadata_bytes: int = 2 * 1024 * 1024
    max_checkpoint_bytes: int = 512 * 1024 * 1024
    max_dataset_bytes: int = 256 * 1024 * 1024
    max_registry_bytes: int = 1024 * 1024 * 1024
    max_rows: int = 250_000
    max_features: int = 4_096
    max_data_values: int = 16_000_000
    max_prediction_values: int = 16_000_000
    max_classes: int = 128
    max_layers: int = 128
    max_parameter_tensors: int = 256
    max_parameter_values: int = 20_000_000
    inference_batch_size: int = 4_096

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    """JSON-safe evidence produced by one held-out evaluation."""

    id: str
    created_utc: str
    report_file: str
    project: str
    run_id: str
    dataset: str
    task: str
    split: HeldOutSplit
    sample_count: int
    checkpoint: dict[str, Any]
    contract_sha256: str
    metrics: dict[str, float]
    details: dict[str, Any]
    comparisons: list[dict[str, Any]]
    accepted: bool

    def to_dict(self) -> dict[str, Any]:
        return {"available": True, "schema": 1, **asdict(self)}


def _json_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise EvaluationError(f"JSON object contains duplicate key {key!r}.")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise EvaluationError(f"JSON contains non-finite number {value!r}.")


def _validate_json_tree(value: Any) -> None:
    stack: list[tuple[Any, int]] = [(value, 0)]
    nodes = 0
    while stack:
        item, depth = stack.pop()
        nodes += 1
        if nodes > 50_000 or depth > 16:
            raise EvaluationError("Artifact JSON is too large or deeply nested.")
        if item is None or isinstance(item, (bool, str, int)):
            continue
        if isinstance(item, float):
            if not math.isfinite(item):
                raise EvaluationError("Artifact JSON numbers must be finite.")
            continue
        if isinstance(item, dict):
            if any(not isinstance(key, str) or len(key) > 200 for key in item):
                raise EvaluationError("Artifact JSON contains an invalid object key.")
            stack.extend((child, depth + 1) for child in item.values())
            continue
        if isinstance(item, list):
            stack.extend((child, depth + 1) for child in item)
            continue
        raise EvaluationError("Artifact JSON contains an unsupported value.")


def _read_json_object(
    path: Path, *, maximum_bytes: int, label: str
) -> tuple[dict[str, Any], bytes]:
    size = path.stat().st_size
    if size <= 0 or size > maximum_bytes:
        raise EvaluationError(f"{label} exceeds its configured size limit.")
    with path.open("rb") as handle:
        encoded = handle.read(maximum_bytes + 1)
    if len(encoded) > maximum_bytes:
        raise EvaluationError(f"{label} exceeds its configured size limit.")
    try:
        raw = json.loads(
            encoded.decode("utf-8"),
            object_pairs_hook=_json_pairs,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise EvaluationError(f"{label} is not valid bounded UTF-8 JSON.") from exc
    if not isinstance(raw, dict):
        raise EvaluationError(f"{label} must be a JSON object.")
    _validate_json_tree(raw)
    return raw, encoded


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise EvaluationError(f"{label} must be a JSON object.")
    return value


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise EvaluationError("Evaluation contract is not finite JSON data.") from exc


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_digest(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in arrays:
        array = np.ascontiguousarray(value)
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise EvaluationError(f"{label} must be a positive integer.")
    return value


def _same_path(left: str | os.PathLike[str], right: str | os.PathLike[str]) -> bool:
    try:
        return Path(left).resolve(strict=False) == Path(right).resolve(strict=False)
    except (OSError, RuntimeError):
        return False


class ModelEvaluator:
    """Evaluate only completed, contract-bearing Daedalus checkpoints."""

    def __init__(
        self,
        manager: WorkspaceManager,
        *,
        limits: EvaluationLimits | None = None,
    ) -> None:
        self.manager = manager
        self.limits = limits or EvaluationLimits()
        self.manager.bootstrap()

    def _checkpoint_paths(
        self, checkpoint: str | os.PathLike[str]
    ) -> tuple[Path, Path | None, str]:
        unresolved = Path(checkpoint)
        if unresolved.is_symlink() or unresolved.parent.is_symlink():
            raise EvaluationError("Checkpoint paths cannot be symbolic links.")
        resolved = self.manager.resolve_user_path(unresolved, must_exist=True)
        if not resolved.is_file():
            raise EvaluationError("Checkpoint must be a regular file.")
        checkpoint_root = self.manager.checkpoints_dir
        if checkpoint_root.is_symlink():
            raise EvaluationError("The private checkpoint directory cannot be a symbolic link.")
        root = checkpoint_root.resolve(strict=True)
        if resolved.parent.parent != root or resolved.parent.is_symlink():
            raise EvaluationError("Checkpoint must be inside one direct private project folder.")
        project = resolved.parent.name
        if safe_project_name(project) != project:
            raise EvaluationError("Checkpoint project folder has an invalid name.")

        suffix = resolved.suffix.casefold()
        if suffix == ".json":
            metadata_path = resolved
            selected_array: Path | None = None
        elif suffix == ".npz":
            match = _GENERATION_ARRAY.fullmatch(resolved.name)
            metadata_name = f"{match.group('stem')}.json" if match else f"{resolved.stem}.json"
            metadata_path = resolved.parent / metadata_name
            selected_array = resolved
        else:
            raise EvaluationError(
                "Only Daedalus JSON metadata or NPZ data checkpoints are accepted."
            )
        if metadata_path.is_symlink() or not metadata_path.is_file():
            raise EvaluationError("Checkpoint metadata is missing or is a symbolic link.")
        metadata_path = metadata_path.resolve(strict=True)
        if metadata_path.parent != resolved.parent:
            raise EvaluationError("Checkpoint metadata escapes its project checkpoint folder.")
        return metadata_path, selected_array, project

    def _project_directory(self, project: str) -> Path:
        projects_root = self.manager.projects_dir
        project_path = projects_root / project
        if projects_root.is_symlink() or project_path.is_symlink():
            raise EvaluationError("Private project paths cannot be symbolic links.")
        if not project_path.is_dir():
            raise EvaluationError(
                "A matching private project must exist before an evaluation report can be written."
            )
        resolved = project_path.resolve(strict=True)
        if resolved.parent != projects_root.resolve(strict=True):
            raise EvaluationError("Project path escapes the private project directory.")
        return resolved

    def _metadata(
        self, metadata_path: Path, selected_array: Path | None
    ) -> tuple[dict[str, Any], bytes, Path]:
        metadata, encoded = _read_json_object(
            metadata_path,
            maximum_bytes=self.limits.max_metadata_bytes,
            label="Checkpoint metadata",
        )
        if metadata.get("format") != "daedalus-npz" or metadata.get("schema") != 2:
            raise EvaluationError("Evaluation requires a schema-2 Daedalus NPZ checkpoint.")
        array_file = metadata.get("array_file")
        if not isinstance(array_file, str) or Path(array_file).name != array_file:
            raise EvaluationError("Checkpoint metadata contains an unsafe array-file path.")
        array_path = metadata_path.parent / array_file
        if array_path.is_symlink() or not array_path.is_file():
            raise EvaluationError("Checkpoint array file is missing or is a symbolic link.")
        array_path = array_path.resolve(strict=True)
        if array_path.parent != metadata_path.parent:
            raise EvaluationError("Checkpoint array file escapes its metadata folder.")
        if selected_array is not None and selected_array.resolve(strict=True) != array_path:
            raise EvaluationError(
                "Selected NPZ file does not match its stable checkpoint metadata."
            )
        if array_path.stat().st_size > self.limits.max_checkpoint_bytes:
            raise EvaluationError("Checkpoint arrays exceed the configured evaluation size limit.")
        checksum = metadata.get("sha256")
        if not isinstance(checksum, str) or _SHA256.fullmatch(checksum) is None:
            raise EvaluationError("Checkpoint metadata contains an invalid SHA-256 digest.")
        parameter_count = _positive_int(metadata.get("parameter_count"), "parameter_count")
        if parameter_count > self.limits.max_parameter_tensors:
            raise EvaluationError("Checkpoint contains too many parameter tensors.")
        return metadata, encoded, array_path

    def _completed_run(self, metadata_path: Path, project: str) -> RunRecord:
        database = self.manager.runs_dir / "runs.sqlite3"
        if database.is_symlink() or not database.is_file():
            raise EvaluationError("Checkpoint has no private training-run registry.")
        if database.stat().st_size > self.limits.max_registry_bytes:
            raise EvaluationError("Training-run registry exceeds the evaluation size limit.")
        records = RunRegistry(database).list_runs(project=project, limit=1000)
        matches = [
            record
            for record in records
            if record.checkpoint is not None and _same_path(record.checkpoint, metadata_path)
        ]
        if len(matches) != 1:
            raise EvaluationError(
                "Checkpoint must resolve to exactly one of the project's 1,000 most recent runs."
            )
        record = matches[0]
        if record.status != "completed":
            raise EvaluationError("Only a completed training run can be evaluated.")
        if not isinstance(record.config, dict) or not isinstance(record.metrics, dict):
            raise EvaluationError("Completed run registry data is malformed.")
        return record

    def _source_data(
        self,
        record: RunRecord,
        source: dict[str, Any],
        task: str,
        seed: int,
    ) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], str]:
        source_name = source.get("name")
        source_sha256 = source.get("sha256")
        feature_columns = source.get("feature_columns")
        target_column = source.get("target_column")
        if source_name != record.dataset or not isinstance(source_name, str):
            raise EvaluationError("Dataset identity differs between the run and checkpoint.")
        if not isinstance(source_sha256, str) or _SHA256.fullmatch(source_sha256) is None:
            raise EvaluationError("Training source has an invalid SHA-256 digest.")
        if (
            not isinstance(feature_columns, list)
            or not feature_columns
            or len(feature_columns) > self.limits.max_features
            or any(not isinstance(name, str) or not name for name in feature_columns)
        ):
            raise EvaluationError("Training source has invalid feature-column metadata.")
        names = tuple(feature_columns)
        if len(set(names)) != len(names) or not isinstance(target_column, str):
            raise EvaluationError("Training source columns are incomplete or duplicated.")

        if source_name == "built-in XOR":
            if task != "classification" or names != ("x1", "x2") or target_column != "target":
                raise EvaluationError("Built-in XOR contract is inconsistent.")
            features, targets = make_xor(200, noise=0.08, seed=seed)
            digest = _array_digest(features, targets)
        elif source_name == "built-in regression":
            if task != "regression" or names != ("feature_0",) or target_column != "target":
                raise EvaluationError("Built-in regression contract is inconsistent.")
            features, targets = make_regression(200, 1, 1, noise=0.05, seed=seed)
            digest = _array_digest(features, targets)
        else:
            dataset_name = source_name
            if Path(dataset_name).name != dataset_name:
                raise EvaluationError("Registered dataset name contains unsafe path characters.")
            metadata_path = self.manager.datasets_dir / f"{dataset_name}.dataset.json"
            if metadata_path.parent != self.manager.datasets_dir or metadata_path.is_symlink():
                raise EvaluationError("Registered dataset metadata path is unsafe.")
            raw, _encoded = _read_json_object(
                metadata_path,
                maximum_bytes=self.limits.max_metadata_bytes,
                label="Dataset metadata",
            )
            data_file = raw.get("file")
            if not isinstance(data_file, str) or Path(data_file).name != data_file:
                raise EvaluationError("Dataset metadata contains an unsafe data-file path.")
            data_path = self.manager.datasets_dir / data_file
            if data_path.is_symlink() or not data_path.is_file():
                raise EvaluationError("Registered dataset data is missing or is a symbolic link.")
            if data_path.stat().st_size > self.limits.max_dataset_bytes:
                raise EvaluationError("Registered dataset exceeds the evaluation size limit.")
            features, targets, dataset_metadata = DatasetService(
                self.manager, maximum_bytes=self.limits.max_dataset_bytes
            ).load(dataset_name)
            if (
                dataset_metadata.sha256 != source_sha256
                or dataset_metadata.feature_columns != names
                or dataset_metadata.target_column != target_column
            ):
                raise EvaluationError("Registered dataset no longer matches its training contract.")
            if _sha256_file(data_path) != source_sha256:
                raise EvaluationError("Registered dataset changed while it was being loaded.")
            digest = dataset_metadata.sha256
        if digest != source_sha256:
            raise EvaluationError("Training source checksum does not match the recorded contract.")
        x = np.asarray(features, dtype=np.float64)
        y = np.asarray(targets, dtype=np.float64)
        if (
            x.ndim != 2
            or len(x) > self.limits.max_rows
            or x.shape[1] > self.limits.max_features
            or x.size > self.limits.max_data_values
        ):
            raise EvaluationError("Training source exceeds the bounded evaluation dimensions.")
        return x, y, names, source_name

    def _replay_preparation(
        self, record: RunRecord, contract: dict[str, Any]
    ) -> PreparedTrainingData:
        if contract.get("schema") != 1:
            raise EvaluationError("Evaluation requires a schema-1 training contract.")
        task = contract.get("task")
        if task not in {"classification", "regression"}:
            raise EvaluationError("Training contract has an unsupported task.")
        training = _mapping(contract.get("training"), "training contract training section")
        seed = training.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2_147_483_647:
            raise EvaluationError("Training contract has an invalid seed.")
        source = _mapping(contract.get("source"), "training source")
        expected_preparation = _mapping(contract.get("preparation"), "preparation contract")
        manifest = _mapping(expected_preparation.get("split_manifest"), "split manifest")
        standardization = _mapping(
            expected_preparation.get("standardization"), "standardization contract"
        )
        validation_fraction = manifest.get("requested_validation_fraction")
        test_fraction = manifest.get("requested_test_fraction")
        standardize = standardization.get("applied")
        if not isinstance(standardize, bool):
            raise EvaluationError("Standardization contract must record a boolean applied flag.")
        if not isinstance(validation_fraction, (int, float)) or not isinstance(
            test_fraction, (int, float)
        ):
            raise EvaluationError("Split contract is missing its requested fractions.")

        features, targets, feature_names, _dataset_name = self._source_data(
            record, source, task, seed
        )
        prepared = prepare_training_data(
            features,
            targets,
            task=task,
            feature_names=feature_names,
            validation_fraction=float(validation_fraction),
            test_fraction=float(test_fraction),
            seed=seed,
            standardize=standardize,
        )
        replayed = prepared.to_dict()
        if _canonical(replayed) != _canonical(expected_preparation):
            raise EvaluationError(
                "Held-out split or preprocessing replay differs from the checkpoint contract."
            )
        run_training_data = record.config.get("training_data")
        if _canonical(run_training_data) != _canonical(
            {"source": source, "preparation": expected_preparation}
        ):
            raise EvaluationError("Run registry lineage differs from the checkpoint contract.")
        if (
            record.config.get("resolved_task") != task
            or record.config.get("dataset") != record.dataset
        ):
            raise EvaluationError("Run registry task or dataset differs from the checkpoint.")
        return prepared

    def _build_model(
        self,
        architecture: Any,
        *,
        input_width: int,
        output_width: int,
    ) -> Sequential:
        if (
            not isinstance(architecture, list)
            or not 1 <= len(architecture) <= self.limits.max_layers
        ):
            raise EvaluationError("Checkpoint architecture has an invalid layer count.")
        layers: list[Any] = []
        width: int | None = None
        parameter_values = 0
        for index, raw in enumerate(architecture):
            layer = _mapping(raw, f"architecture layer {index}")
            layer_type = layer.get("type")
            layer_index = layer.get("index")
            if (
                isinstance(layer_index, bool)
                or not isinstance(layer_index, int)
                or layer_index != index
            ):
                raise EvaluationError("Checkpoint architecture layer indices are not contiguous.")
            if layer_type == "Linear":
                if set(layer) != {"index", "type", "in_features", "out_features", "bias"}:
                    raise EvaluationError(
                        "Linear architecture metadata contains unexpected fields."
                    )
                in_features = _positive_int(layer.get("in_features"), "Linear.in_features")
                out_features = _positive_int(layer.get("out_features"), "Linear.out_features")
                bias = layer.get("bias")
                if not isinstance(bias, bool):
                    raise EvaluationError("Linear.bias must be a boolean.")
                if width is None and in_features != input_width:
                    raise EvaluationError("Model input width differs from the prepared dataset.")
                if width is not None and in_features != width:
                    raise EvaluationError("Adjacent model layer widths do not connect.")
                parameter_values += in_features * out_features + (out_features if bias else 0)
                if parameter_values > self.limits.max_parameter_values:
                    raise EvaluationError(
                        "Checkpoint model exceeds the evaluation parameter limit."
                    )
                layers.append(Linear(in_features, out_features, bias=bias, seed=0))
                width = out_features
            elif layer_type in {"Tanh", "ReLU"}:
                if set(layer) != {"index", "type"} or width is None:
                    raise EvaluationError("Activation architecture metadata is invalid.")
                layers.append(Tanh() if layer_type == "Tanh" else ReLU())
            else:
                raise EvaluationError(
                    "Checkpoint architecture includes a layer outside the evaluation allowlist."
                )
        if not isinstance(layers[-1], Linear) or width != output_width:
            raise EvaluationError("Model output width differs from the recorded task contract.")
        model = Sequential(*layers)
        if len(model.parameters()) > self.limits.max_parameter_tensors:
            raise EvaluationError("Checkpoint model contains too many parameter tensors.")
        return model

    def _predict(self, model: Sequential, features: np.ndarray, output_width: int) -> np.ndarray:
        if len(features) * output_width > self.limits.max_prediction_values:
            raise EvaluationError("Held-out predictions exceed the evaluation memory limit.")
        output = np.empty((len(features), output_width), dtype=np.float64)
        model.eval()
        with no_grad():
            for start in range(0, len(features), self.limits.inference_batch_size):
                stop = min(len(features), start + self.limits.inference_batch_size)
                batch = model(Tensor(features[start:stop])).data
                if batch.shape != (stop - start, output_width):
                    raise EvaluationError("Model produced an unexpected held-out output shape.")
                output[start:stop] = batch
        if not np.all(np.isfinite(output)):
            raise EvaluationError("Model produced non-finite held-out predictions.")
        return output

    @staticmethod
    def _classification_metrics(
        truth: np.ndarray,
        logits: np.ndarray,
        prepared: PreparedTrainingData,
    ) -> tuple[dict[str, float], dict[str, Any]]:
        labels = list(prepared.label_mapping)
        class_count = len(labels)
        target = np.asarray(truth).reshape(-1)
        if not np.all(np.equal(target, np.floor(target))):
            raise EvaluationError("Classification holdout contains non-integer encoded labels.")
        target = target.astype(np.int64)
        predicted = np.argmax(logits, axis=1).astype(np.int64)
        if np.any(target < 0) or np.any(target >= class_count):
            raise EvaluationError("Classification holdout contains an out-of-range encoded label.")
        confusion = np.zeros((class_count, class_count), dtype=np.int64)
        np.add.at(confusion, (target, predicted), 1)
        true_positive = np.diag(confusion).astype(np.float64)
        support = np.sum(confusion, axis=1).astype(np.float64)
        predicted_count = np.sum(confusion, axis=0).astype(np.float64)
        precision = np.divide(
            true_positive,
            predicted_count,
            out=np.zeros(class_count, dtype=np.float64),
            where=predicted_count != 0,
        )
        recall = np.divide(
            true_positive,
            support,
            out=np.zeros(class_count, dtype=np.float64),
            where=support != 0,
        )
        denominator = precision + recall
        f1 = np.divide(
            2 * precision * recall,
            denominator,
            out=np.zeros(class_count, dtype=np.float64),
            where=denominator != 0,
        )
        accuracy = float(np.sum(true_positive) / len(target))
        balanced_accuracy = float(np.mean(recall))
        macro_f1 = float(np.mean(f1))
        weighted_f1 = float(np.sum(f1 * support) / np.sum(support))
        metrics = {
            "accuracy": accuracy,
            "balanced_accuracy": balanced_accuracy,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
        }
        per_class = [
            {
                "original_label": entry.original_label,
                "encoded_label": entry.encoded_label,
                "support": int(support[index]),
                "predicted_count": int(predicted_count[index]),
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
            }
            for index, entry in enumerate(labels)
        ]
        return metrics, {
            "labels": [entry.original_label for entry in labels],
            "confusion_matrix": confusion.tolist(),
            "per_class": per_class,
        }

    @staticmethod
    def _regression_metrics(
        truth: np.ndarray, predictions: np.ndarray
    ) -> tuple[dict[str, float], dict[str, Any]]:
        metrics = evaluate_predictions(truth, predictions, task="regression").to_dict()
        residuals = predictions.reshape(-1) - np.asarray(truth, dtype=np.float64).reshape(-1)
        quantiles = np.quantile(residuals, [0.05, 0.25, 0.5, 0.75, 0.95])
        summary = {
            "count": len(residuals),
            "minimum": float(np.min(residuals)),
            "maximum": float(np.max(residuals)),
            "mean": float(np.mean(residuals)),
            "standard_deviation": float(np.std(residuals)),
            "q05": float(quantiles[0]),
            "q25": float(quantiles[1]),
            "median": float(quantiles[2]),
            "q75": float(quantiles[3]),
            "q95": float(quantiles[4]),
        }
        return metrics, {"residual_summary": summary}

    @staticmethod
    def _metric_mapping(value: Mapping[str, float] | None, label: str) -> dict[str, float]:
        if value is None:
            return {}
        if not isinstance(value, Mapping) or len(value) > 64:
            raise EvaluationError(f"{label} must be a bounded metric mapping.")
        raw: Mapping[str, Any] = value
        nested = raw.get("metrics")
        if nested is not None:
            if len(raw) != 1 or not isinstance(nested, Mapping):
                raise EvaluationError(f"{label} baseline report must contain only a metrics map.")
            raw = nested
        result: dict[str, float] = {}
        for name, number in raw.items():
            if name not in _METRIC_DIRECTIONS:
                raise EvaluationError(f"Unsupported comparison metric: {name!r}.")
            if isinstance(number, bool) or not isinstance(number, (int, float)):
                raise EvaluationError(f"{label} value for {name!r} must be numeric.")
            resolved = float(number)
            if not math.isfinite(resolved):
                raise EvaluationError(f"{label} value for {name!r} must be finite.")
            result[name] = resolved
        return result

    def _compare(
        self,
        metrics: dict[str, float],
        thresholds: Mapping[str, float] | None,
        baseline: Mapping[str, float] | None,
    ) -> tuple[list[dict[str, Any]], bool]:
        requested = self._metric_mapping(thresholds, "Acceptance thresholds")
        reference = self._metric_mapping(baseline, "Baseline")
        comparisons: list[dict[str, Any]] = []
        gates: list[bool] = []
        for name in sorted(set(requested) | set(reference)):
            if name not in metrics:
                raise EvaluationError(f"Metric {name!r} is unavailable for this task.")
            value = metrics[name]
            direction = _METRIC_DIRECTIONS[name]
            item: dict[str, Any] = {"metric": name, "value": value, "direction": direction}
            if name in requested:
                threshold = requested[name]
                passed = value >= threshold if direction == "higher" else value <= threshold
                item["threshold"] = threshold
                item["threshold_passed"] = passed
                gates.append(passed)
            if name in reference:
                base = reference[name]
                improvement = value - base if direction == "higher" else base - value
                passed = improvement >= 0
                item.update(
                    {
                        "baseline": base,
                        "improvement": float(improvement),
                        "baseline_passed": passed,
                    }
                )
                gates.append(passed)
            comparisons.append(item)
        return comparisons, all(gates)

    @staticmethod
    def _write_report(project: Path, report: EvaluationReport) -> Path:
        reports = project / "reports"
        if reports.is_symlink():
            raise EvaluationError("Project report folders cannot be symbolic links.")
        reports.mkdir(exist_ok=True)
        if (
            reports.is_symlink()
            or not reports.is_dir()
            or reports.resolve(strict=True).parent != project
        ):
            raise EvaluationError("Project report path escapes the private project.")
        destination = reports / report.report_file
        temporary = reports / f".{report.id}.json.tmp"
        encoded = (
            json.dumps(
                report.to_dict(), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
            ).encode("utf-8")
            + b"\n"
        )
        try:
            with temporary.open("xb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            # A hard link publishes the fully synced inode atomically and fails
            # rather than replacing a pre-existing immutable report name.
            os.link(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        return destination.resolve(strict=True)

    def evaluate_checkpoint(
        self,
        path: str | os.PathLike[str] | None = None,
        *,
        checkpoint: str | os.PathLike[str] | None = None,
        split: HeldOutSplit = "test",
        acceptance_thresholds: Mapping[str, float] | None = None,
        baseline: Mapping[str, float] | None = None,
    ) -> dict[str, Any]:
        """Evaluate a completed checkpoint and atomically publish an immutable report."""

        selected = _selected_checkpoint(path, checkpoint)
        if split not in {"validation", "test"}:
            raise ValueError("split must be validation or test")
        metadata_path, selected_array, project = self._checkpoint_paths(selected)
        project_path = self._project_directory(project)
        metadata, metadata_bytes, array_path = self._metadata(metadata_path, selected_array)
        record = self._completed_run(metadata_path, project)
        contract = _mapping(metadata.get("training_contract"), "training contract")
        architecture = metadata.get("architecture")
        if _canonical(architecture) != _canonical(contract.get("architecture")):
            raise EvaluationError("Checkpoint architecture differs from its training contract.")
        if _canonical(metadata.get("metrics")) != _canonical(record.metrics):
            raise EvaluationError("Checkpoint metrics differ from the completed run registry.")
        prepared = self._replay_preparation(record, contract)
        if split == "test":
            features, targets = prepared.test_features, prepared.test_targets
        else:
            features, targets = prepared.validation_features, prepared.validation_targets
        if not len(features):
            raise EvaluationError(f"Recorded {split} split is empty.")
        task = prepared.assessment.task
        output_width = len(prepared.label_mapping) if task == "classification" else 1
        if output_width > self.limits.max_classes:
            raise EvaluationError("Classification output exceeds the evaluation class limit.")
        model = self._build_model(
            architecture,
            input_width=features.shape[1],
            output_width=output_width,
        )
        loaded = load_checkpoint(metadata_path, model.parameters())
        current_metadata, current_bytes = _read_json_object(
            metadata_path,
            maximum_bytes=self.limits.max_metadata_bytes,
            label="Checkpoint metadata",
        )
        if current_bytes != metadata_bytes or _canonical(loaded) != _canonical(current_metadata):
            raise EvaluationError("Checkpoint metadata changed during evaluation.")
        if array_path.stat().st_size > self.limits.max_checkpoint_bytes:
            raise EvaluationError("Checkpoint arrays changed size during evaluation.")
        predictions = self._predict(model, features, output_width)
        if task == "classification":
            metrics, details = self._classification_metrics(targets, predictions, prepared)
        else:
            metrics, details = self._regression_metrics(targets, predictions)
        if not all(math.isfinite(value) for value in metrics.values()):
            raise EvaluationError("Evaluation produced a non-finite metric.")
        comparisons, accepted = self._compare(metrics, acceptance_thresholds, baseline)
        report_id = uuid.uuid4().hex
        array_sha256 = _sha256_file(array_path)
        if array_sha256 != metadata.get("sha256"):
            raise EvaluationError("Checkpoint arrays changed during evaluation.")
        report = EvaluationReport(
            id=report_id,
            created_utc=datetime.now(UTC).isoformat(),
            report_file=f"evaluation-{record.id}-{report_id}.json",
            project=project,
            run_id=record.id,
            dataset=record.dataset,
            task=task,
            split=split,
            sample_count=len(features),
            checkpoint={
                "metadata": str(metadata_path.relative_to(self.manager.workspace_root)),
                "metadata_sha256": hashlib.sha256(metadata_bytes).hexdigest(),
                "arrays": str(array_path.relative_to(self.manager.workspace_root)),
                "arrays_sha256": array_sha256,
            },
            contract_sha256=hashlib.sha256(_canonical(contract)).hexdigest(),
            metrics=metrics,
            details=details,
            comparisons=comparisons,
            accepted=accepted,
        )
        report_path = self._write_report(project_path, report)
        return {**report.to_dict(), "report_path": str(report_path)}

    def evaluate(
        self,
        path: str | os.PathLike[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Alias for :meth:`evaluate_checkpoint`."""

        return self.evaluate_checkpoint(path, **kwargs)


def _selected_checkpoint(
    path: str | os.PathLike[str] | None,
    checkpoint: str | os.PathLike[str] | None,
) -> str | os.PathLike[str]:
    if path is None and checkpoint is None:
        raise ValueError("A checkpoint path is required.")
    if path is not None and checkpoint is not None and not _same_path(path, checkpoint):
        raise ValueError("path and checkpoint refer to different files.")
    return path if path is not None else checkpoint  # type: ignore[return-value]


def evaluate_checkpoint(
    path: str | os.PathLike[str] | None = None,
    *,
    checkpoint: str | os.PathLike[str] | None = None,
    manager: WorkspaceManager | None = None,
    split: HeldOutSplit = "test",
    acceptance_thresholds: Mapping[str, float] | None = None,
    baseline: Mapping[str, float] | None = None,
    limits: EvaluationLimits | None = None,
) -> dict[str, Any]:
    """Convenience entry point used by the GUI and integrations."""

    selected_manager = manager or WorkspaceManager.from_environment()
    return ModelEvaluator(selected_manager, limits=limits).evaluate_checkpoint(
        path,
        checkpoint=checkpoint,
        split=split,
        acceptance_thresholds=acceptance_thresholds,
        baseline=baseline,
    )


__all__ = [
    "EvaluationError",
    "EvaluationLimits",
    "EvaluationReport",
    "HeldOutSplit",
    "ModelEvaluator",
    "evaluate_checkpoint",
]
