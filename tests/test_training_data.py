from __future__ import annotations

import json

import numpy as np
import pytest

from daedalus.engine.training_data import (
    assess_training_data,
    evaluate_predictions,
    prepare_training_data,
)


def test_assessment_detects_task_quality_and_bounded_metadata() -> None:
    features = np.array(
        [
            [0.0, 7.0],
            [0.0, 7.0],
            [1.0, 7.0],
            [1.0, 7.0],
            [2.0, 7.0],
            [2.0, 7.0],
        ]
    )
    targets = np.array([10, 20, 10, 20, 10, 20])
    report = assess_training_data(features, targets, feature_names=("signal", "constant"))

    assert report.task == "classification"
    assert report.duplicate_feature_rows == 3
    assert report.conflicting_duplicate_groups == 3
    assert report.constant_features == ("constant",)
    assert {issue.code for issue in report.issues} >= {
        "small_dataset",
        "constant_features",
        "duplicate_rows",
        "conflicting_duplicates",
    }
    encoded = json.dumps(report.to_dict())
    assert "signal" in encoded
    assert "training" in encoded.casefold()


def test_classification_split_is_deterministic_disjoint_and_stratified() -> None:
    generator = np.random.default_rng(8)
    features = generator.normal(size=(60, 3))
    targets = np.repeat(np.array([-3.0, 9.0, 42.0]), 20)
    first = prepare_training_data(
        features,
        targets,
        task="classification",
        validation_fraction=0.2,
        test_fraction=0.2,
        seed=91,
    )
    second = prepare_training_data(
        features,
        targets,
        task="classification",
        validation_fraction=0.2,
        test_fraction=0.2,
        seed=91,
    )

    np.testing.assert_array_equal(first.train_indices, second.train_indices)
    np.testing.assert_array_equal(first.validation_indices, second.validation_indices)
    np.testing.assert_array_equal(first.test_indices, second.test_indices)
    assert first.split_manifest.combined_sha256 == second.split_manifest.combined_sha256
    all_indices = np.concatenate(
        [first.train_indices, first.validation_indices, first.test_indices]
    )
    assert sorted(all_indices.tolist()) == list(range(len(features)))
    assert len(set(first.train_indices) & set(first.validation_indices)) == 0
    assert len(set(first.train_indices) & set(first.test_indices)) == 0
    assert set(np.unique(first.train_targets)) == {0, 1, 2}
    assert [entry.original_label for entry in first.label_mapping] == [-3, 9, 42]
    metadata = first.to_dict()
    assert "train_indices" not in metadata
    assert metadata["split_manifest"]["train_indices_sha256"]


def test_standardization_is_fit_on_training_rows_only() -> None:
    features = np.arange(80, dtype=np.float64).reshape(40, 2)
    targets = np.linspace(-1.0, 1.0, 40)
    prepared = prepare_training_data(
        features,
        targets,
        task="regression",
        validation_fraction=0.2,
        test_fraction=0.1,
        seed=12,
        standardize=True,
    )

    expected_mean = features[prepared.train_indices].mean(axis=0)
    np.testing.assert_allclose(prepared.standardization.feature_mean, expected_mean)
    np.testing.assert_allclose(prepared.train_features.mean(axis=0), 0.0, atol=1e-12)
    assert not np.allclose(prepared.validation_features.mean(axis=0), 0.0)
    assert prepared.train_targets.shape[1] == 1


def test_singleton_class_cannot_be_leaked_out_of_training() -> None:
    features = np.arange(18, dtype=np.float64).reshape(9, 2)
    targets = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2])
    with pytest.raises(ValueError, match="at least two examples"):
        prepare_training_data(features, targets, task="classification")


def test_prediction_metrics_cover_classification_and_regression() -> None:
    classification = evaluate_predictions(
        np.array([0, 0, 1, 1]),
        np.array([[4, 1], [3, 2], [2, 3], [5, 1]]),
        task="classification",
    ).to_dict()
    assert classification == {"accuracy": 0.75, "balanced_accuracy": 0.75}

    regression = evaluate_predictions(
        np.array([[1.0], [2.0], [3.0]]),
        np.array([[1.0], [2.0], [4.0]]),
        task="regression",
    ).to_dict()
    assert regression["mse"] == pytest.approx(1 / 3)
    assert regression["rmse"] == pytest.approx(np.sqrt(1 / 3))
    assert regression["mae"] == pytest.approx(1 / 3)
    assert regression["r2"] == pytest.approx(0.5)
