"""Bounded data assessment, leak-free preparation, and held-out metrics."""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike

TaskType = Literal["classification", "regression"]
RequestedTask = Literal["auto", "classification", "regression"]
IssueSeverity = Literal["info", "warning", "error"]


@dataclass(frozen=True, slots=True)
class DataIssue:
    severity: IssueSeverity
    code: str
    message: str


@dataclass(frozen=True, slots=True)
class FeatureStatistic:
    index: int
    name: str
    minimum: float
    maximum: float
    mean: float
    standard_deviation: float


@dataclass(frozen=True, slots=True)
class ClassDistributionEntry:
    label: int | float
    count: int
    fraction: float


@dataclass(frozen=True, slots=True)
class DatasetAssessment:
    task: TaskType
    sample_count: int
    feature_count: int
    inspected_row_count: int
    inspected_feature_count: int
    unique_target_count: int
    duplicate_feature_rows: int
    conflicting_duplicate_groups: int
    constant_features: tuple[str, ...]
    feature_statistics: tuple[FeatureStatistic, ...]
    class_distribution: tuple[ClassDistributionEntry, ...]
    target_minimum: float
    target_maximum: float
    target_mean: float
    target_standard_deviation: float
    issues: tuple[DataIssue, ...]
    recommendations: tuple[str, ...]

    @property
    def has_errors(self) -> bool:
        return any(issue.severity == "error" for issue in self.issues)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class LabelMappingEntry:
    original_label: int | float
    encoded_label: int


@dataclass(frozen=True, slots=True)
class StandardizationMetadata:
    applied: bool
    feature_mean: tuple[float, ...]
    feature_scale: tuple[float, ...]
    constant_feature_indices: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SplitManifest:
    schema: int
    algorithm: str
    seed: int
    requested_validation_fraction: float
    requested_test_fraction: float
    total_rows: int
    train_rows: int
    validation_rows: int
    test_rows: int
    train_indices_sha256: str
    validation_indices_sha256: str
    test_indices_sha256: str
    combined_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PreparedTrainingData:
    task: TaskType
    train_features: np.ndarray
    train_targets: np.ndarray
    validation_features: np.ndarray
    validation_targets: np.ndarray
    test_features: np.ndarray
    test_targets: np.ndarray
    train_indices: np.ndarray
    validation_indices: np.ndarray
    test_indices: np.ndarray
    assessment: DatasetAssessment
    label_mapping: tuple[LabelMappingEntry, ...]
    standardization: StandardizationMetadata
    split_manifest: SplitManifest
    preparation_warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return metadata only; raw examples and row indices are intentionally omitted."""

        return {
            "task": self.task,
            "assessment": self.assessment.to_dict(),
            "label_mapping": [asdict(entry) for entry in self.label_mapping],
            "standardization": self.standardization.to_dict(),
            "split_manifest": self.split_manifest.to_dict(),
            "preparation_warnings": list(self.preparation_warnings),
        }


@dataclass(frozen=True, slots=True)
class EvaluationMetrics:
    task: TaskType
    accuracy: float | None = None
    balanced_accuracy: float | None = None
    mse: float | None = None
    rmse: float | None = None
    mae: float | None = None
    r2: float | None = None

    def to_dict(self) -> dict[str, float]:
        return {
            name: float(value)
            for name, value in (
                ("accuracy", self.accuracy),
                ("balanced_accuracy", self.balanced_accuracy),
                ("mse", self.mse),
                ("rmse", self.rmse),
                ("mae", self.mae),
                ("r2", self.r2),
            )
            if value is not None
        }


def _coerce_data(
    features: ArrayLike,
    targets: ArrayLike,
    feature_names: tuple[str, ...] | list[str] | None,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    try:
        x = np.asarray(features, dtype=np.float64)
        y = np.asarray(targets, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError("Training features and targets must be numeric.") from exc
    if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] == 0:
        raise ValueError("features must be a non-empty 2D sample matrix")
    if y.ndim == 2 and y.shape[1] == 1:
        y = y.reshape(-1)
    if y.ndim != 1:
        raise ValueError("targets must contain one scalar target per sample")
    if len(x) != len(y):
        raise ValueError("features and targets must contain the same number of samples")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("training data must contain only finite values")
    if feature_names is None:
        names = tuple(f"feature_{index}" for index in range(x.shape[1]))
    else:
        names = tuple(str(name).strip() for name in feature_names)
        if len(names) != x.shape[1] or any(not name for name in names):
            raise ValueError("feature_names must provide one non-empty name per feature")
        if len(set(names)) != len(names):
            raise ValueError("feature_names must be unique")
    return x, y, names


def _infer_task(targets: np.ndarray) -> TaskType:
    unique = np.unique(targets)
    integer_like = np.all(np.isclose(targets, np.round(targets), rtol=0.0, atol=1e-9))
    class_limit = min(128, max(20, int(round(math.sqrt(len(targets)) * 2))))
    if 2 <= len(unique) <= class_limit and integer_like:
        return "classification"
    return "regression"


def _label_value(value: float) -> int | float:
    return int(value) if float(value).is_integer() else float(value)


def assess_training_data(
    features: ArrayLike,
    targets: ArrayLike,
    *,
    task: RequestedTask = "auto",
    feature_names: tuple[str, ...] | list[str] | None = None,
    max_rows_inspected: int = 50_000,
    max_features_inspected: int = 256,
) -> DatasetAssessment:
    """Build a bounded, non-mutating quality report for numeric supervised data."""

    if task not in {"auto", "classification", "regression"}:
        raise ValueError("task must be auto, classification, or regression")
    if max_rows_inspected <= 0 or max_features_inspected <= 0:
        raise ValueError("inspection limits must be positive")
    x, y, names = _coerce_data(features, targets, feature_names)
    resolved_task = _infer_task(y) if task == "auto" else task
    row_count = min(len(x), int(max_rows_inspected))
    feature_count = min(x.shape[1], int(max_features_inspected))
    inspected_x = x[:row_count, :feature_count]
    inspected_y = y[:row_count]

    unique_rows, inverse, counts = np.unique(
        inspected_x, axis=0, return_inverse=True, return_counts=True
    )
    del unique_rows
    duplicate_rows = int(np.sum(counts - 1))
    conflicts = 0
    for group in np.flatnonzero(counts > 1):
        if len(np.unique(inspected_y[inverse == group])) > 1:
            conflicts += 1

    means = np.mean(x, axis=0)
    standard_deviations = np.std(x, axis=0)
    minima = np.min(x, axis=0)
    maxima = np.max(x, axis=0)
    inspected_indices = range(feature_count)
    feature_statistics = tuple(
        FeatureStatistic(
            index=index,
            name=names[index],
            minimum=float(minima[index]),
            maximum=float(maxima[index]),
            mean=float(means[index]),
            standard_deviation=float(standard_deviations[index]),
        )
        for index in inspected_indices
    )
    constant_indices = np.flatnonzero(np.isclose(standard_deviations, 0.0))
    constant_names = tuple(names[index] for index in constant_indices[:max_features_inspected])

    unique_targets, target_counts = np.unique(y, return_counts=True)
    class_distribution: tuple[ClassDistributionEntry, ...] = ()
    if resolved_task == "classification":
        class_distribution = tuple(
            ClassDistributionEntry(
                label=_label_value(float(label)),
                count=int(count),
                fraction=float(count / len(y)),
            )
            for label, count in zip(unique_targets, target_counts, strict=True)
        )

    issues: list[DataIssue] = []
    if len(x) < 30:
        issues.append(
            DataIssue(
                "warning",
                "small_dataset",
                "Fewer than 30 examples makes held-out metrics highly variable.",
            )
        )
    if x.shape[1] >= len(x):
        issues.append(
            DataIssue(
                "warning",
                "features_outnumber_samples",
                "Feature count is at least the sample count; overfitting risk is high.",
            )
        )
    if constant_names:
        issues.append(
            DataIssue(
                "warning",
                "constant_features",
                f"{len(constant_indices)} feature column(s) are constant and add no signal.",
            )
        )
    if duplicate_rows:
        qualifier = "within the bounded inspection sample" if row_count < len(x) else ""
        issues.append(
            DataIssue(
                "warning",
                "duplicate_rows",
                f"Found {duplicate_rows} repeated feature row(s) {qualifier}.".strip(),
            )
        )
    if conflicts:
        issues.append(
            DataIssue(
                "warning",
                "conflicting_duplicates",
                f"{conflicts} repeated feature group(s) have different targets.",
            )
        )
    nonzero_scales = standard_deviations[standard_deviations > 1e-12]
    if len(nonzero_scales) and float(np.max(nonzero_scales) / np.min(nonzero_scales)) > 1_000:
        issues.append(
            DataIssue(
                "warning",
                "feature_scale_mismatch",
                "Feature scales differ by more than 1,000×; train-only standardization is recommended.",
            )
        )
    if resolved_task == "classification":
        if not 2 <= len(unique_targets) <= 128:
            issues.append(
                DataIssue(
                    "error",
                    "invalid_class_count",
                    "Classification requires between 2 and 128 distinct target labels.",
                )
            )
        elif int(np.min(target_counts)) < 2:
            issues.append(
                DataIssue(
                    "error",
                    "singleton_class",
                    "Every class needs at least two examples to preserve a training example and a holdout.",
                )
            )
        if len(target_counts) >= 2 and int(np.max(target_counts)) / int(np.min(target_counts)) >= 10:
            issues.append(
                DataIssue(
                    "warning",
                    "class_imbalance",
                    "The largest class has at least 10× as many examples as the smallest class.",
                )
            )
    elif np.isclose(np.std(y), 0.0):
        issues.append(
            DataIssue(
                "error",
                "constant_target",
                "A constant regression target gives the model nothing meaningful to learn.",
            )
        )
    if row_count < len(x) or feature_count < x.shape[1]:
        issues.append(
            DataIssue(
                "info",
                "bounded_inspection",
                "Duplicate and per-feature detail was bounded to keep analysis responsive.",
            )
        )

    recommendations = [
        "Confirm collection rights, consent, retention, and permitted training use.",
        "Keep validation and final-test rows unavailable to the optimizer.",
        "Record the dataset checksum, split seed, preprocessing, and label map with the checkpoint.",
    ]
    if constant_names:
        recommendations.append("Remove constant feature columns before a production experiment.")
    if any(issue.code == "class_imbalance" for issue in issues):
        recommendations.append("Collect more minority-class examples or use a declared balance strategy.")
    if conflicts:
        recommendations.append("Review conflicting duplicate examples and document genuine ambiguity.")

    return DatasetAssessment(
        task=resolved_task,
        sample_count=len(x),
        feature_count=x.shape[1],
        inspected_row_count=row_count,
        inspected_feature_count=feature_count,
        unique_target_count=len(unique_targets),
        duplicate_feature_rows=duplicate_rows,
        conflicting_duplicate_groups=conflicts,
        constant_features=constant_names,
        feature_statistics=feature_statistics,
        class_distribution=class_distribution,
        target_minimum=float(np.min(y)),
        target_maximum=float(np.max(y)),
        target_mean=float(np.mean(y)),
        target_standard_deviation=float(np.std(y)),
        issues=tuple(issues),
        recommendations=tuple(recommendations),
    )


def _split_hash(indices: np.ndarray) -> str:
    values = np.ascontiguousarray(indices, dtype="<i8")
    return hashlib.sha256(values.tobytes()).hexdigest()


def _partition_counts(
    sample_count: int,
    validation_fraction: float,
    test_fraction: float,
    maximum_holdout: int,
) -> tuple[int, int, bool]:
    validation_count = max(1, int(round(sample_count * validation_fraction)))
    test_count = 0 if test_fraction == 0 else max(1, int(round(sample_count * test_fraction)))
    requested_total = validation_count + test_count
    if maximum_holdout < (1 + (test_fraction > 0)):
        raise ValueError("Not enough repeated examples to create the requested held-out splits")
    reduced = requested_total > maximum_holdout
    if reduced:
        ratio = validation_fraction / (validation_fraction + test_fraction)
        validation_count = max(1, int(round(maximum_holdout * ratio)))
        test_count = maximum_holdout - validation_count
        if test_fraction > 0 and test_count == 0:
            test_count = 1
            validation_count = maximum_holdout - 1
    return validation_count, test_count, reduced


def _stratified_indices(
    encoded_targets: np.ndarray,
    validation_count: int,
    test_count: int,
    validation_fraction: float,
    test_fraction: float,
    generator: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    classes, counts = np.unique(encoded_targets, return_counts=True)
    capacities = counts.astype(int) - 1
    allocations = np.zeros((2, len(classes)), dtype=int)
    fractions = (validation_fraction, test_fraction)
    desired = (validation_count, test_count)
    remaining = capacities.copy()
    for split_index, split_count in enumerate(desired):
        for _ in range(split_count):
            candidates = np.flatnonzero(remaining > 0)
            if not len(candidates):
                raise ValueError("Class coverage cannot support the requested held-out splits")
            target_quota = counts[candidates] * fractions[split_index]
            deficit = target_quota - allocations[split_index, candidates]
            jitter = generator.random(len(candidates)) * 1e-9
            chosen = int(candidates[int(np.argmax(deficit + jitter))])
            allocations[split_index, chosen] += 1
            remaining[chosen] -= 1

    train_parts: list[np.ndarray] = []
    validation_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []
    for class_position, class_value in enumerate(classes):
        class_indices = np.flatnonzero(encoded_targets == class_value)
        generator.shuffle(class_indices)
        validation_end = allocations[0, class_position]
        test_end = validation_end + allocations[1, class_position]
        validation_parts.append(class_indices[:validation_end])
        test_parts.append(class_indices[validation_end:test_end])
        train_parts.append(class_indices[test_end:])
    train = np.concatenate(train_parts)
    validation = np.concatenate(validation_parts)
    test = np.concatenate(test_parts) if test_count else np.empty(0, dtype=np.int64)
    generator.shuffle(train)
    generator.shuffle(validation)
    generator.shuffle(test)
    return train, validation, test


def prepare_training_data(
    features: ArrayLike,
    targets: ArrayLike,
    *,
    task: RequestedTask = "auto",
    feature_names: tuple[str, ...] | list[str] | None = None,
    validation_fraction: float = 0.2,
    test_fraction: float = 0.2,
    seed: int = 47,
    standardize: bool = True,
    max_rows_inspected: int = 50_000,
    max_features_inspected: int = 256,
) -> PreparedTrainingData:
    """Create deterministic held-out splits and fit preprocessing on training rows only."""

    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")
    if not 0 <= test_fraction < 1:
        raise ValueError("test_fraction must be in [0, 1)")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("held-out fractions must leave training rows")
    if not 0 <= int(seed) <= 2_147_483_647:
        raise ValueError("seed must be between 0 and 2,147,483,647")
    x, y, names = _coerce_data(features, targets, feature_names)
    assessment = assess_training_data(
        x,
        y,
        task=task,
        feature_names=names,
        max_rows_inspected=max_rows_inspected,
        max_features_inspected=max_features_inspected,
    )
    fatal = [issue.message for issue in assessment.issues if issue.severity == "error"]
    if fatal:
        raise ValueError("Training data is not ready: " + " ".join(fatal))
    generator = np.random.default_rng(int(seed))
    label_mapping: tuple[LabelMappingEntry, ...] = ()
    if assessment.task == "classification":
        labels, encoded = np.unique(y, return_inverse=True)
        label_mapping = tuple(
            LabelMappingEntry(_label_value(float(label)), index)
            for index, label in enumerate(labels)
        )
        encoded_targets = encoded.astype(np.int64)
        maximum_holdout = len(x) - len(labels)
    else:
        encoded_targets = y.reshape(-1, 1)
        maximum_holdout = len(x) - 1
    validation_count, test_count, reduced = _partition_counts(
        len(x), validation_fraction, test_fraction, maximum_holdout
    )
    warnings: list[str] = []
    if reduced:
        warnings.append(
            "Held-out row counts were reduced to preserve at least one training example per class."
        )
    if assessment.task == "classification":
        train_indices, validation_indices, test_indices = _stratified_indices(
            encoded_targets,
            validation_count,
            test_count,
            validation_fraction,
            test_fraction,
            generator,
        )
        split_algorithm = "stratified-v1"
    else:
        order = generator.permutation(len(x))
        validation_indices = order[:validation_count]
        test_indices = order[validation_count : validation_count + test_count]
        train_indices = order[validation_count + test_count :]
        split_algorithm = "random-v1"
    if not len(train_indices) or not len(validation_indices) or (test_fraction and not len(test_indices)):
        raise ValueError("Data is too small for non-empty train, validation, and test splits")

    raw_train = x[train_indices]
    if standardize:
        feature_mean = np.mean(raw_train, axis=0)
        raw_scale = np.std(raw_train, axis=0)
        constant = np.flatnonzero(np.isclose(raw_scale, 0.0))
        feature_scale = np.where(np.isclose(raw_scale, 0.0), 1.0, raw_scale)
        prepared_x = (x - feature_mean) / feature_scale
    else:
        feature_mean = np.zeros(x.shape[1], dtype=np.float64)
        feature_scale = np.ones(x.shape[1], dtype=np.float64)
        constant = np.flatnonzero(np.isclose(np.std(raw_train, axis=0), 0.0))
        prepared_x = x.copy()
    standardization_metadata = StandardizationMetadata(
        applied=bool(standardize),
        feature_mean=tuple(float(value) for value in feature_mean),
        feature_scale=tuple(float(value) for value in feature_scale),
        constant_feature_indices=tuple(int(value) for value in constant),
    )
    train_hash = _split_hash(train_indices)
    validation_hash = _split_hash(validation_indices)
    test_hash = _split_hash(test_indices)
    combined = hashlib.sha256(
        bytes.fromhex(train_hash) + bytes.fromhex(validation_hash) + bytes.fromhex(test_hash)
    ).hexdigest()
    manifest = SplitManifest(
        schema=1,
        algorithm=split_algorithm,
        seed=int(seed),
        requested_validation_fraction=float(validation_fraction),
        requested_test_fraction=float(test_fraction),
        total_rows=len(x),
        train_rows=len(train_indices),
        validation_rows=len(validation_indices),
        test_rows=len(test_indices),
        train_indices_sha256=train_hash,
        validation_indices_sha256=validation_hash,
        test_indices_sha256=test_hash,
        combined_sha256=combined,
    )
    return PreparedTrainingData(
        task=assessment.task,
        train_features=np.asarray(prepared_x[train_indices], dtype=np.float64),
        train_targets=np.asarray(encoded_targets[train_indices]),
        validation_features=np.asarray(prepared_x[validation_indices], dtype=np.float64),
        validation_targets=np.asarray(encoded_targets[validation_indices]),
        test_features=np.asarray(prepared_x[test_indices], dtype=np.float64),
        test_targets=np.asarray(encoded_targets[test_indices]),
        train_indices=np.asarray(train_indices, dtype=np.int64),
        validation_indices=np.asarray(validation_indices, dtype=np.int64),
        test_indices=np.asarray(test_indices, dtype=np.int64),
        assessment=assessment,
        label_mapping=label_mapping,
        standardization=standardization_metadata,
        split_manifest=manifest,
        preparation_warnings=tuple(warnings),
    )


def evaluate_predictions(
    targets: ArrayLike,
    predictions: ArrayLike,
    *,
    task: TaskType,
) -> EvaluationMetrics:
    """Calculate bounded classification or regression metrics without dependencies."""

    if task not in {"classification", "regression"}:
        raise ValueError("task must be classification or regression")
    truth = np.asarray(targets)
    predicted = np.asarray(predictions)
    if not len(truth):
        raise ValueError("evaluation data cannot be empty")
    if task == "classification":
        truth = truth.reshape(-1)
        if predicted.ndim == 2:
            if predicted.shape[0] != len(truth) or predicted.shape[1] < 2:
                raise ValueError("classification logits must be shaped (samples, classes)")
            predicted_labels = np.argmax(predicted, axis=1)
        else:
            predicted_labels = predicted.reshape(-1)
        if len(predicted_labels) != len(truth):
            raise ValueError("prediction and target counts must match")
        accuracy = float(np.mean(predicted_labels == truth))
        recalls = [
            float(np.mean(predicted_labels[truth == label] == label))
            for label in np.unique(truth)
        ]
        return EvaluationMetrics(
            task="classification",
            accuracy=accuracy,
            balanced_accuracy=float(np.mean(recalls)),
        )
    truth_values = np.asarray(truth, dtype=np.float64).reshape(-1)
    predicted_values = np.asarray(predicted, dtype=np.float64).reshape(-1)
    if len(predicted_values) != len(truth_values):
        raise ValueError("prediction and target counts must match")
    if not np.all(np.isfinite(predicted_values)) or not np.all(np.isfinite(truth_values)):
        raise ValueError("evaluation values must be finite")
    residuals = predicted_values - truth_values
    mse = float(np.mean(np.square(residuals)))
    mae = float(np.mean(np.abs(residuals)))
    denominator = float(np.sum(np.square(truth_values - np.mean(truth_values))))
    r2 = 1.0 if denominator == 0 and mse == 0 else (
        0.0 if denominator == 0 else 1.0 - float(np.sum(np.square(residuals))) / denominator
    )
    return EvaluationMetrics(
        task="regression",
        mse=mse,
        rmse=math.sqrt(mse),
        mae=mae,
        r2=float(r2),
    )


__all__ = [
    "ClassDistributionEntry",
    "DataIssue",
    "DatasetAssessment",
    "EvaluationMetrics",
    "FeatureStatistic",
    "LabelMappingEntry",
    "PreparedTrainingData",
    "SplitManifest",
    "StandardizationMetadata",
    "TaskType",
    "assess_training_data",
    "evaluate_predictions",
    "prepare_training_data",
]
