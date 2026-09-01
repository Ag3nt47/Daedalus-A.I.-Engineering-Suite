"""Educational calculators for architecture design and training planning."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import DTypeLike


class CalculatorError(ValueError):
    """Base class for actionable calculator input errors."""


class ShapeCalculationError(CalculatorError):
    """Raised when dimensions cannot participate in a requested operation."""


class EstimateInputError(CalculatorError):
    """Raised when an estimate receives a non-physical configuration."""


@dataclass(frozen=True)
class AttentionActivationEstimate:
    qkv_bytes: int
    score_bytes: int
    probability_bytes: int
    context_bytes: int
    output_bytes: int
    total_bytes: int


@dataclass(frozen=True)
class TransformerActivationEstimate:
    attention_bytes: int
    normalization_bytes: int
    feed_forward_bytes: int
    residual_bytes: int
    total_bytes: int


@dataclass(frozen=True)
class QuantizedMemoryEstimate:
    weight_bytes: int
    scale_bytes: int
    zero_point_bytes: int
    total_bytes: int
    compression_ratio_vs_float32: float


@dataclass(frozen=True)
class TrainingSchedule:
    batches_per_epoch: int
    optimizer_steps_per_epoch: int
    total_batches: int
    total_optimizer_steps: int


@dataclass(frozen=True)
class TrainingTimeEstimate:
    seconds_per_epoch: float
    total_seconds: float

    @property
    def total_minutes(self) -> float:
        return self.total_seconds / 60.0

    @property
    def total_hours(self) -> float:
        return self.total_seconds / 3600.0


def _shape(shape: Sequence[int], name: str) -> tuple[int, ...]:
    resolved = tuple(int(dimension) for dimension in shape)
    if any(dimension < 0 for dimension in resolved):
        raise ShapeCalculationError(f"{name} dimensions cannot be negative: {resolved}")
    return resolved


def _positive(name: str, value: int) -> int:
    resolved = int(value)
    if resolved <= 0:
        raise EstimateInputError(f"{name} must be positive, got {value}")
    return resolved


def _dimensions(
    value: int | Sequence[int],
    count: int,
    name: str,
    *,
    allow_zero: bool = False,
) -> tuple[int, ...]:
    values = (int(value),) * count if isinstance(value, int) else tuple(int(item) for item in value)
    if len(values) != count:
        raise ShapeCalculationError(f"{name} must contain {count} dimensions, got {len(values)}")
    minimum = 0 if allow_zero else 1
    if any(item < minimum for item in values):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ShapeCalculationError(f"{name} dimensions must be {qualifier}, got {values}")
    return values


def broadcast_output_shape(*shapes: Sequence[int]) -> tuple[int, ...]:
    """Return NumPy broadcasting's result shape without allocating arrays."""

    if not shapes:
        return ()
    resolved = [_shape(shape, f"shape {index}") for index, shape in enumerate(shapes)]
    width = max(len(shape) for shape in resolved)
    output: list[int] = []
    for offset in range(1, width + 1):
        dimensions = [shape[-offset] if offset <= len(shape) else 1 for shape in resolved]
        non_singletons = {dimension for dimension in dimensions if dimension != 1}
        if len(non_singletons) > 1:
            raise ShapeCalculationError(
                f"cannot broadcast {tuple(resolved)}: trailing dimensions {dimensions} conflict"
            )
        output.append(non_singletons.pop() if non_singletons else 1)
    return tuple(reversed(output))


def matmul_output_shape(
    left_shape: Sequence[int],
    right_shape: Sequence[int],
) -> tuple[int, ...]:
    """Infer ``numpy.matmul`` output shape, including vectors and batch axes."""

    left = _shape(left_shape, "left_shape")
    right = _shape(right_shape, "right_shape")
    if not left or not right:
        raise ShapeCalculationError("matmul operands must each have at least one dimension")
    left_contract = left[-1]
    right_contract = right[-2] if len(right) > 1 else right[-1]
    if left_contract != right_contract:
        raise ShapeCalculationError(
            f"matmul contraction mismatch: left has {left_contract}, right has {right_contract}"
        )
    left_batch = left[:-2] if len(left) > 1 else ()
    right_batch = right[:-2] if len(right) > 1 else ()
    batch = broadcast_output_shape(left_batch, right_batch)
    if len(left) == 1 and len(right) == 1:
        return batch
    if len(left) == 1:
        return (*batch, right[-1])
    if len(right) == 1:
        return (*batch, left[-2])
    return (*batch, left[-2], right[-1])


def convolution_output_shape(
    spatial_shape: Sequence[int],
    kernel_size: int | Sequence[int],
    *,
    stride: int | Sequence[int] = 1,
    padding: int | Sequence[int] = 0,
    dilation: int | Sequence[int] = 1,
) -> tuple[int, ...]:
    """Calculate N-dimensional convolution spatial output dimensions."""

    spatial = _shape(spatial_shape, "spatial_shape")
    if not spatial or any(dimension <= 0 for dimension in spatial):
        raise ShapeCalculationError("spatial_shape must contain positive dimensions")
    count = len(spatial)
    kernels = _dimensions(kernel_size, count, "kernel_size")
    strides = _dimensions(stride, count, "stride")
    paddings = _dimensions(padding, count, "padding", allow_zero=True)
    dilations = _dimensions(dilation, count, "dilation")
    output: list[int] = []
    for dimension, kernel, step, pad, spacing in zip(
        spatial,
        kernels,
        strides,
        paddings,
        dilations,
        strict=True,
    ):
        effective_kernel = spacing * (kernel - 1) + 1
        available = dimension + 2 * pad - effective_kernel
        if available < 0:
            raise ShapeCalculationError(
                f"effective kernel {effective_kernel} does not fit input dimension {dimension} "
                f"with padding {pad}"
            )
        output.append(available // step + 1)
    return tuple(output)


def convolution_parameter_count(
    in_channels: int,
    out_channels: int,
    kernel_size: int | Sequence[int],
    *,
    groups: int = 1,
    bias: bool = True,
) -> int:
    """Count grouped-convolution weights and optional output-channel biases."""

    inputs = _positive("in_channels", in_channels)
    outputs = _positive("out_channels", out_channels)
    group_count = _positive("groups", groups)
    if inputs % group_count or outputs % group_count:
        raise EstimateInputError("groups must divide both input and output channel counts")
    kernels = (kernel_size,) if isinstance(kernel_size, int) else tuple(kernel_size)
    kernel_shape = _dimensions(kernels, len(kernels), "kernel_size")
    weights = outputs * (inputs // group_count) * int(np.prod(kernel_shape, dtype=np.int64))
    return weights + (outputs if bias else 0)


def attention_parameter_count(embedding_dim: int, num_heads: int, *, bias: bool = True) -> int:
    """Count separate query, key, value, and output projections."""

    width = _positive("embedding_dim", embedding_dim)
    heads = _positive("num_heads", num_heads)
    if width % heads:
        raise EstimateInputError("embedding_dim must be divisible by num_heads")
    return 4 * width * width + (4 * width if bias else 0)


def transformer_block_parameter_count(
    embedding_dim: int,
    num_heads: int,
    feed_forward_dim: int | None = None,
    *,
    bias: bool = True,
) -> int:
    """Count the parameters in the suite's affine pre-norm TransformerBlock."""

    width = _positive("embedding_dim", embedding_dim)
    hidden = 4 * width if feed_forward_dim is None else _positive(
        "feed_forward_dim", feed_forward_dim
    )
    attention = attention_parameter_count(width, num_heads, bias=bias)
    feed_forward = 2 * width * hidden + (hidden + width if bias else 0)
    layer_norms = 4 * width
    return attention + feed_forward + layer_norms


def estimate_attention_activation_memory(
    batch_size: int,
    sequence_length: int,
    embedding_dim: int,
    num_heads: int,
    *,
    dtype: DTypeLike = np.float32,
) -> AttentionActivationEstimate:
    """Estimate dense forward buffers retained by scaled dot-product attention."""

    batch = _positive("batch_size", batch_size)
    sequence = _positive("sequence_length", sequence_length)
    width = _positive("embedding_dim", embedding_dim)
    heads = _positive("num_heads", num_heads)
    if width % heads:
        raise EstimateInputError("embedding_dim must be divisible by num_heads")
    item_size = np.dtype(dtype).itemsize
    token_elements = batch * sequence * width
    matrix_elements = batch * heads * sequence * sequence
    qkv_bytes = 3 * token_elements * item_size
    score_bytes = matrix_elements * item_size
    probability_bytes = matrix_elements * item_size
    context_bytes = token_elements * item_size
    output_bytes = token_elements * item_size
    total = qkv_bytes + score_bytes + probability_bytes + context_bytes + output_bytes
    return AttentionActivationEstimate(
        qkv_bytes=qkv_bytes,
        score_bytes=score_bytes,
        probability_bytes=probability_bytes,
        context_bytes=context_bytes,
        output_bytes=output_bytes,
        total_bytes=total,
    )


def estimate_transformer_activation_memory(
    batch_size: int,
    sequence_length: int,
    embedding_dim: int,
    num_heads: int,
    feed_forward_dim: int | None = None,
    *,
    dtype: DTypeLike = np.float32,
) -> TransformerActivationEstimate:
    """Estimate attention, normalization, feed-forward, and residual buffers."""

    width = _positive("embedding_dim", embedding_dim)
    hidden = 4 * width if feed_forward_dim is None else _positive(
        "feed_forward_dim", feed_forward_dim
    )
    batch = _positive("batch_size", batch_size)
    sequence = _positive("sequence_length", sequence_length)
    item_size = np.dtype(dtype).itemsize
    attention = estimate_attention_activation_memory(
        batch,
        sequence,
        width,
        num_heads,
        dtype=dtype,
    )
    token_elements = batch * sequence * width
    normalization_bytes = 2 * token_elements * item_size
    feed_forward_bytes = 2 * batch * sequence * hidden * item_size
    residual_bytes = 2 * token_elements * item_size
    total = (
        attention.total_bytes
        + normalization_bytes
        + feed_forward_bytes
        + residual_bytes
    )
    return TransformerActivationEstimate(
        attention_bytes=attention.total_bytes,
        normalization_bytes=normalization_bytes,
        feed_forward_bytes=feed_forward_bytes,
        residual_bytes=residual_bytes,
        total_bytes=total,
    )


def estimate_quantized_model_bytes(
    parameter_count: int,
    *,
    bits_per_weight: int = 8,
    group_size: int | None = None,
    scale_dtype: DTypeLike = np.float32,
    zero_point_bits: int = 0,
) -> QuantizedMemoryEstimate:
    """Estimate packed weights plus optional per-group scale/zero-point metadata."""

    count = _positive("parameter_count", parameter_count)
    bits = _positive("bits_per_weight", bits_per_weight)
    if bits > 32:
        raise EstimateInputError("bits_per_weight above 32 is not a quantized representation")
    if zero_point_bits < 0:
        raise EstimateInputError("zero_point_bits cannot be negative")
    weight_bytes = math.ceil(count * bits / 8)
    groups = 0 if group_size is None else math.ceil(count / _positive("group_size", group_size))
    scale_bytes = groups * np.dtype(scale_dtype).itemsize
    zero_point_bytes = groups * math.ceil(zero_point_bits / 8)
    total = weight_bytes + scale_bytes + zero_point_bytes
    float32_bytes = count * np.dtype(np.float32).itemsize
    return QuantizedMemoryEstimate(
        weight_bytes=weight_bytes,
        scale_bytes=scale_bytes,
        zero_point_bytes=zero_point_bytes,
        total_bytes=total,
        compression_ratio_vs_float32=float32_bytes / total,
    )


def batches_per_epoch(sample_count: int, batch_size: int, *, drop_last: bool = False) -> int:
    samples = _positive("sample_count", sample_count)
    batch = _positive("batch_size", batch_size)
    return samples // batch if drop_last else math.ceil(samples / batch)


def training_schedule(
    sample_count: int,
    batch_size: int,
    *,
    epochs: int = 1,
    gradient_accumulation: int = 1,
    drop_last: bool = False,
) -> TrainingSchedule:
    """Calculate data batches and optimizer updates for an entire run."""

    epoch_count = _positive("epochs", epochs)
    accumulation = _positive("gradient_accumulation", gradient_accumulation)
    batches = batches_per_epoch(sample_count, batch_size, drop_last=drop_last)
    if batches == 0:
        raise EstimateInputError("drop_last removes every batch; reduce batch_size or keep the remainder")
    optimizer_steps = math.ceil(batches / accumulation)
    return TrainingSchedule(
        batches_per_epoch=batches,
        optimizer_steps_per_epoch=optimizer_steps,
        total_batches=batches * epoch_count,
        total_optimizer_steps=optimizer_steps * epoch_count,
    )


def project_training_time(
    seconds_per_step: float,
    steps_per_epoch: int,
    epochs: int,
    *,
    overhead_fraction: float = 0.0,
) -> TrainingTimeEstimate:
    """Project wall time from measured optimizer-step latency."""

    if seconds_per_step <= 0:
        raise EstimateInputError("seconds_per_step must be positive")
    steps = _positive("steps_per_epoch", steps_per_epoch)
    epoch_count = _positive("epochs", epochs)
    if overhead_fraction < 0:
        raise EstimateInputError("overhead_fraction cannot be negative")
    seconds_per_epoch = float(seconds_per_step) * steps * (1.0 + overhead_fraction)
    return TrainingTimeEstimate(
        seconds_per_epoch=seconds_per_epoch,
        total_seconds=seconds_per_epoch * epoch_count,
    )


def initialization_scale(
    fan_in: int,
    fan_out: int | None = None,
    *,
    method: str = "xavier_uniform",
) -> float:
    """Return an initializer's uniform bound or normal standard deviation.

    ``*_uniform`` methods return the symmetric bound ``[-scale, scale]``;
    ``*_normal`` methods return the target standard deviation.
    """

    inputs = _positive("fan_in", fan_in)
    outputs = None if fan_out is None else _positive("fan_out", fan_out)
    normalized = method.lower().replace("-", "_")
    if normalized == "xavier_uniform":
        if outputs is None:
            raise EstimateInputError("xavier initialization requires fan_out")
        return math.sqrt(6.0 / (inputs + outputs))
    if normalized == "xavier_normal":
        if outputs is None:
            raise EstimateInputError("xavier initialization requires fan_out")
        return math.sqrt(2.0 / (inputs + outputs))
    if normalized == "he_uniform":
        return math.sqrt(6.0 / inputs)
    if normalized == "he_normal":
        return math.sqrt(2.0 / inputs)
    if normalized == "lecun_normal":
        return math.sqrt(1.0 / inputs)
    raise EstimateInputError(
        "method must be xavier_uniform, xavier_normal, he_uniform, he_normal, or lecun_normal"
    )

