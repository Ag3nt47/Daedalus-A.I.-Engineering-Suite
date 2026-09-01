from __future__ import annotations

import numpy as np
import pytest

from daedalus.engine import (
    analyze_gradients,
    calculate_output_shapes,
    check_model_gradients,
    count_parameters,
    estimate_array_memory,
    estimate_model_memory,
    gradient_check,
    parameter_summary,
    profile_callable,
)
from daedalus.layers import Linear, Sequential, Tanh
from daedalus.losses import MSELoss


def test_parameter_and_shape_calculators() -> None:
    model = Sequential(Linear(3, 5, seed=1), Tanh(), Linear(5, 2, seed=2))
    summary = parameter_summary(model)
    assert summary.total == (3 * 5 + 5) + (5 * 2 + 2) == 32
    assert count_parameters(model, trainable_only=True) == 32

    shapes = calculate_output_shapes(model, (16, 3))
    assert [item.output_shape for item in shapes] == [(16, 5), (16, 5), (16, 2)]
    assert [item.parameters for item in shapes] == [20, 0, 12]


def test_shape_calculator_rejects_incompatible_linear_layer() -> None:
    with pytest.raises(ValueError, match="expects final dimension 4"):
        calculate_output_shapes(Linear(4, 2, seed=1), (8, 3))


def test_memory_calculator_accounts_for_parameters_states_and_activations() -> None:
    model = Sequential(Linear(3, 2, seed=1))
    estimate = estimate_model_memory(
        model,
        batch_size=4,
        input_shape=(3,),
        dtype=np.float32,
        optimizer="adam",
    )
    parameter_bytes = (3 * 2 + 2) * 4
    activation_bytes = (4 * 3 + 4 * 2) * 4
    assert estimate.parameter_bytes == parameter_bytes
    assert estimate.gradient_bytes == parameter_bytes
    assert estimate.optimizer_bytes == parameter_bytes * 2
    assert estimate.activation_bytes == activation_bytes
    assert estimate.total_bytes == parameter_bytes * 4 + activation_bytes
    assert estimate_array_memory((2, 3), dtype=np.float64, copies=2) == 96


def test_gradient_checker_handles_composed_tensor_function() -> None:
    check = gradient_check(
        lambda left, right: ((left @ right).tanh() ** 2).mean(),
        [np.array([[0.2, -0.3], [0.7, 0.5]]), np.array([[0.4], [-0.8]])],
    )
    assert check.passed, check.failures
    assert check.checked_values == 6
    assert check.max_absolute_error < 1e-6


def test_model_gradient_checker_verifies_linear_loss() -> None:
    model = Sequential(Linear(2, 1, seed=3))
    features = np.array([[0.2, -0.5], [1.0, 0.3], [-0.7, 0.2]])
    targets = np.array([[0.1], [0.8], [-0.2]])
    check = check_model_gradients(model, MSELoss(), features, targets)
    assert check.passed, check.failures
    assert check.checked_values == 3


def test_gradient_profiler_classifies_health_states() -> None:
    model = Sequential(Linear(2, 2, seed=4))
    weight, bias = model.parameters()
    weight.grad = np.full_like(weight.data, 1e4)
    bias.grad = np.array([np.nan, 1.0])
    diagnostics = analyze_gradients(model)
    assert [item.status for item in diagnostics] == ["exploding", "non_finite"]
    assert not diagnostics[1].finite


def test_callable_profiler_returns_positive_timing_and_throughput() -> None:
    stats = profile_callable(lambda values: values @ values.T, np.eye(8), warmup=0, repeats=3)
    assert stats.repeats == 3
    assert 0 <= stats.min_seconds <= stats.max_seconds
    assert stats.mean_seconds >= 0
    assert stats.throughput_per_second > 0

