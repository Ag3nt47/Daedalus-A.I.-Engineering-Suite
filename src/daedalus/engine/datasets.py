"""Deterministic toy datasets used by examples and validation tests."""

from __future__ import annotations

from typing import Literal, overload

import numpy as np
from numpy.typing import DTypeLike, NDArray


def make_xor(
    n_samples: int = 200,
    *,
    noise: float = 0.08,
    seed: int = 0,
    dtype: DTypeLike = np.float64,
) -> tuple[NDArray[np.floating], NDArray[np.int64]]:
    """Create a balanced, shuffled two-class XOR dataset.

    Features are centered around ``(-1, -1)``, ``(-1, 1)``, ``(1, -1)``, and
    ``(1, 1)``.  Building from a tiled base keeps the dataset balanced before
    deterministic shuffling.
    """

    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    if noise < 0:
        raise ValueError("noise cannot be negative")
    generator = np.random.default_rng(seed)
    base = np.array([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
    repeats = (n_samples + len(base) - 1) // len(base)
    features = np.tile(base, (repeats, 1))[:n_samples]
    labels = (features[:, 0] != features[:, 1]).astype(np.int64)
    if noise:
        features = features + generator.normal(0.0, noise, size=features.shape)
    order = generator.permutation(n_samples)
    return features[order].astype(dtype), labels[order]


@overload
def make_regression(
    n_samples: int = 200,
    n_features: int = 1,
    n_targets: int = 1,
    *,
    noise: float = 0.05,
    seed: int = 0,
    weights: np.ndarray | None = None,
    bias: np.ndarray | float | None = None,
    dtype: DTypeLike = np.float64,
    return_parameters: Literal[False] = False,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]: ...


@overload
def make_regression(
    n_samples: int = 200,
    n_features: int = 1,
    n_targets: int = 1,
    *,
    noise: float = 0.05,
    seed: int = 0,
    weights: np.ndarray | None = None,
    bias: np.ndarray | float | None = None,
    dtype: DTypeLike = np.float64,
    return_parameters: Literal[True],
) -> tuple[
    NDArray[np.floating],
    NDArray[np.floating],
    NDArray[np.floating],
    NDArray[np.floating],
]: ...


def make_regression(
    n_samples: int = 200,
    n_features: int = 1,
    n_targets: int = 1,
    *,
    noise: float = 0.05,
    seed: int = 0,
    weights: np.ndarray | None = None,
    bias: np.ndarray | float | None = None,
    dtype: DTypeLike = np.float64,
    return_parameters: bool = False,
) -> tuple[NDArray[np.floating], ...]:
    """Create a reproducible linear-regression problem."""

    if n_samples <= 0 or n_features <= 0 or n_targets <= 0:
        raise ValueError("sample, feature, and target counts must be positive")
    if noise < 0:
        raise ValueError("noise cannot be negative")
    generator = np.random.default_rng(seed)
    feature_values = generator.normal(size=(n_samples, n_features))
    if weights is None:
        true_weights = generator.normal(size=(n_features, n_targets))
    else:
        true_weights = np.asarray(weights, dtype=np.float64)
        if true_weights.shape == (n_features,) and n_targets == 1:
            true_weights = true_weights.reshape(n_features, 1)
        if true_weights.shape != (n_features, n_targets):
            raise ValueError(
                f"weights must have shape {(n_features, n_targets)}, got {true_weights.shape}"
            )
    if bias is None:
        true_bias = generator.normal(size=(n_targets,))
    else:
        true_bias = np.broadcast_to(np.asarray(bias, dtype=np.float64), (n_targets,)).copy()
    targets = feature_values @ true_weights + true_bias
    if noise:
        targets = targets + generator.normal(0.0, noise, size=targets.shape)
    result = (feature_values.astype(dtype), targets.astype(dtype))
    if return_parameters:
        return result + (true_weights.astype(dtype), true_bias.astype(dtype))
    return result


def train_test_split(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    test_fraction: float = 0.2,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Deterministically shuffle and split aligned arrays."""

    if len(features) != len(targets):
        raise ValueError("features and targets must contain the same number of samples")
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be between zero and one")
    generator = np.random.default_rng(seed)
    order = generator.permutation(len(features))
    test_count = max(1, int(round(len(features) * test_fraction)))
    test_indices = order[:test_count]
    train_indices = order[test_count:]
    if len(train_indices) == 0:
        raise ValueError("split leaves no training samples")
    return (
        features[train_indices],
        features[test_indices],
        targets[train_indices],
        targets[test_indices],
    )


def xor_dataset(
    samples: int = 200,
    *,
    noise: float = 0.08,
    seed: int = 0,
    dtype: DTypeLike = np.float64,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Compatibility dataset for single-output MSE tutorial projects."""

    features, labels = make_xor(samples, noise=noise, seed=seed, dtype=dtype)
    return features, labels.astype(dtype).reshape(-1, 1)


def regression_dataset(
    samples: int = 200,
    *,
    noise: float = 0.05,
    seed: int = 0,
    dtype: DTypeLike = np.float64,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Compatibility alias for a one-feature, one-target regression problem."""

    return make_regression(samples, 1, 1, noise=noise, seed=seed, dtype=dtype)
