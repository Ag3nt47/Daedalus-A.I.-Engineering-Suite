"""Finite-difference verification for Tensor functions and model parameters."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from daedalus.core import Tensor, no_grad
from daedalus.layers import Layer

from .trainer import LossFunction


@dataclass(frozen=True)
class GradientCheckResult:
    passed: bool
    checked_values: int
    max_absolute_error: float
    max_relative_error: float
    failures: tuple[str, ...]


def _relative_error(analytic: float, numerical: float) -> float:
    return abs(analytic - numerical) / max(1.0, abs(analytic), abs(numerical))


def gradient_check(
    function: Callable[..., Tensor],
    inputs: Sequence[ArrayLike],
    *,
    epsilon: float = 1e-6,
    atol: float = 1e-5,
    rtol: float = 1e-4,
) -> GradientCheckResult:
    """Compare autograd and central differences for a scalar function."""

    if epsilon <= 0 or atol < 0 or rtol < 0:
        raise ValueError("epsilon must be positive and tolerances non-negative")
    values = [np.asarray(value, dtype=np.float64).copy() for value in inputs]
    analytic_inputs = [Tensor(value.copy(), requires_grad=True) for value in values]
    output = function(*analytic_inputs)
    if not isinstance(output, Tensor) or output.size != 1:
        raise ValueError("gradient_check function must return a scalar Tensor")
    output.backward()
    analytic = [tensor.grad.copy() if tensor.grad is not None else None for tensor in analytic_inputs]

    max_absolute = 0.0
    max_relative = 0.0
    checked = 0
    failures: list[str] = []
    for input_index, (base, gradient) in enumerate(zip(values, analytic, strict=True)):
        if gradient is None:
            failures.append(f"input {input_index}: analytical gradient is missing")
            continue
        for index in np.ndindex(base.shape):
            plus = [value.copy() for value in values]
            minus = [value.copy() for value in values]
            plus[input_index][index] += epsilon
            minus[input_index][index] -= epsilon
            with no_grad():
                plus_value = float(np.real(function(*(Tensor(value) for value in plus)).item()))
                minus_value = float(np.real(function(*(Tensor(value) for value in minus)).item()))
            numerical = (plus_value - minus_value) / (2.0 * epsilon)
            analytical = float(np.real(gradient[index]))
            absolute = abs(analytical - numerical)
            relative = _relative_error(analytical, numerical)
            max_absolute = max(max_absolute, absolute)
            max_relative = max(max_relative, relative)
            checked += 1
            if absolute > atol + rtol * abs(numerical):
                failures.append(
                    f"input {input_index}{index}: analytic={analytical:.8g}, numerical={numerical:.8g}"
                )

    return GradientCheckResult(
        passed=not failures,
        checked_values=checked,
        max_absolute_error=max_absolute,
        max_relative_error=max_relative,
        failures=tuple(failures[:20]),
    )


def check_model_gradients(
    model: Layer,
    loss_function: LossFunction,
    features: ArrayLike,
    targets: ArrayLike,
    *,
    epsilon: float = 1e-6,
    atol: float = 1e-5,
    rtol: float = 1e-4,
    max_values_per_parameter: int | None = None,
) -> GradientCheckResult:
    """Numerically verify trainable model parameters on a fixed batch."""

    if max_values_per_parameter is not None and max_values_per_parameter <= 0:
        raise ValueError("max_values_per_parameter must be positive")
    x = np.asarray(features)
    y = np.asarray(targets)
    model.zero_grad()
    loss = loss_function(model(Tensor(x)), y)
    if loss.size != 1:
        raise ValueError("model gradient checking requires a scalar loss")
    loss.backward()

    max_absolute = 0.0
    max_relative = 0.0
    checked = 0
    failures: list[str] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None:
            failures.append(f"{name}: analytical gradient is missing")
            continue
        indices = list(np.ndindex(parameter.shape))
        if max_values_per_parameter is not None:
            indices = indices[:max_values_per_parameter]
        for index in indices:
            original = parameter.data[index].copy()
            try:
                parameter.data[index] = original + epsilon
                with no_grad():
                    plus = float(np.real(loss_function(model(Tensor(x)), y).item()))
                parameter.data[index] = original - epsilon
                with no_grad():
                    minus = float(np.real(loss_function(model(Tensor(x)), y).item()))
            finally:
                parameter.data[index] = original
            numerical = (plus - minus) / (2.0 * epsilon)
            analytical = float(np.real(parameter.grad[index]))
            absolute = abs(analytical - numerical)
            relative = _relative_error(analytical, numerical)
            max_absolute = max(max_absolute, absolute)
            max_relative = max(max_relative, relative)
            checked += 1
            if absolute > atol + rtol * abs(numerical):
                failures.append(
                    f"{name}{index}: analytic={analytical:.8g}, numerical={numerical:.8g}"
                )

    return GradientCheckResult(
        passed=not failures,
        checked_values=checked,
        max_absolute_error=max_absolute,
        max_relative_error=max_relative,
        failures=tuple(failures[:20]),
    )

