"""Timing and gradient-health diagnostics."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np

from daedalus.layers import Layer, Parameter


@dataclass(frozen=True)
class GradientDiagnostic:
    name: str
    status: str
    finite: bool
    minimum: float | None
    maximum: float | None
    mean: float | None
    l2_norm: float | None
    max_abs: float | None
    zero_fraction: float | None


@dataclass(frozen=True)
class TimingStats:
    repeats: int
    mean_seconds: float
    median_seconds: float
    p95_seconds: float
    min_seconds: float
    max_seconds: float
    throughput_per_second: float


def analyze_gradients(
    source: Layer | Iterable[Parameter],
    *,
    vanishing_threshold: float = 1e-8,
    exploding_threshold: float = 1e3,
) -> list[GradientDiagnostic]:
    """Classify each parameter gradient as healthy, missing, or hazardous."""

    if vanishing_threshold < 0 or exploding_threshold <= vanishing_threshold:
        raise ValueError("gradient thresholds must satisfy 0 <= vanishing < exploding")
    if isinstance(source, Layer):
        named = list(source.named_parameters())
    else:
        named = [(parameter.name or f"parameter_{index}", parameter) for index, parameter in enumerate(source)]
    diagnostics: list[GradientDiagnostic] = []
    for name, parameter in named:
        gradient = parameter.grad
        if gradient is None:
            diagnostics.append(
                GradientDiagnostic(name, "missing", True, None, None, None, None, None, None)
            )
            continue
        finite = bool(np.all(np.isfinite(gradient)))
        minimum = float(np.nanmin(gradient)) if gradient.size else 0.0
        maximum = float(np.nanmax(gradient)) if gradient.size else 0.0
        mean = float(np.nanmean(gradient)) if gradient.size else 0.0
        l2_norm = float(np.linalg.norm(np.nan_to_num(gradient)))
        max_abs = float(np.nanmax(np.abs(gradient))) if gradient.size else 0.0
        zero_fraction = float(np.mean(gradient == 0)) if gradient.size else 1.0
        if not finite:
            status = "non_finite"
        elif max_abs >= exploding_threshold:
            status = "exploding"
        elif max_abs == 0.0:
            status = "zero"
        elif max_abs <= vanishing_threshold:
            status = "vanishing"
        else:
            status = "healthy"
        diagnostics.append(
            GradientDiagnostic(
                name=name,
                status=status,
                finite=finite,
                minimum=minimum,
                maximum=maximum,
                mean=mean,
                l2_norm=l2_norm,
                max_abs=max_abs,
                zero_fraction=zero_fraction,
            )
        )
    return diagnostics


def profile_callable(
    function: Callable[..., Any],
    *args: Any,
    warmup: int = 1,
    repeats: int = 10,
    items_per_call: int = 1,
    **kwargs: Any,
) -> TimingStats:
    """Measure wall-clock latency and derived throughput."""

    if warmup < 0 or repeats <= 0 or items_per_call <= 0:
        raise ValueError("warmup must be non-negative; repeats and items_per_call must be positive")
    for _ in range(warmup):
        function(*args, **kwargs)
    samples = np.empty(repeats, dtype=np.float64)
    for index in range(repeats):
        start = perf_counter()
        function(*args, **kwargs)
        samples[index] = perf_counter() - start
    mean = float(samples.mean())
    return TimingStats(
        repeats=repeats,
        mean_seconds=mean,
        median_seconds=float(np.median(samples)),
        p95_seconds=float(np.percentile(samples, 95)),
        min_seconds=float(samples.min()),
        max_seconds=float(samples.max()),
        throughput_per_second=float(items_per_call / mean) if mean else float("inf"),
    )

