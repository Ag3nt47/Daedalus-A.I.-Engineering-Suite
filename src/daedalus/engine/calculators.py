"""Shape, parameter, and projected-memory calculators."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import DTypeLike

from daedalus.layers import (
    Embedding,
    Flatten,
    Layer,
    LayerNorm,
    Linear,
    MultiHeadSelfAttention,
    Sequential,
    TransformerBlock,
)


@dataclass(frozen=True)
class ParameterSummary:
    total: int
    trainable: int
    frozen: int


@dataclass(frozen=True)
class LayerShape:
    name: str
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    parameters: int


@dataclass(frozen=True)
class MemoryEstimate:
    parameter_bytes: int
    gradient_bytes: int
    optimizer_bytes: int
    activation_bytes: int
    total_bytes: int

    @property
    def megabytes(self) -> float:
        return self.total_bytes / (1024**2)


def parameter_summary(model: Layer) -> ParameterSummary:
    parameters = model.parameters()
    total = sum(parameter.size for parameter in parameters)
    trainable = sum(parameter.size for parameter in parameters if parameter.requires_grad)
    return ParameterSummary(total=total, trainable=trainable, frozen=total - trainable)


def count_parameters(model: Layer, *, trainable_only: bool = False) -> int:
    summary = parameter_summary(model)
    return summary.trainable if trainable_only else summary.total


def _flatten_layers(model: Layer) -> list[tuple[str, Layer]]:
    if isinstance(model, Sequential):
        return [(f"layers.{index}", layer) for index, layer in enumerate(model.layers)]
    return [(model.__class__.__name__, model)]


def calculate_output_shapes(
    model: Layer,
    input_shape: tuple[int, ...],
) -> list[LayerShape]:
    """Infer each layer's shape without executing or allocating a model graph."""

    if not input_shape or any(dimension <= 0 for dimension in input_shape):
        raise ValueError("input_shape must contain positive dimensions")
    current = tuple(int(dimension) for dimension in input_shape)
    results: list[LayerShape] = []
    for name, layer in _flatten_layers(model):
        incoming = current
        if isinstance(layer, Linear):
            if incoming[-1] != layer.in_features:
                raise ValueError(
                    f"{name} expects final dimension {layer.in_features}, got {incoming[-1]}"
                )
            current = (*incoming[:-1], layer.out_features)
        elif isinstance(layer, Flatten):
            current = layer.output_shape(incoming)
        elif isinstance(layer, Embedding):
            current = (*incoming, layer.embedding_dim)
        elif isinstance(layer, LayerNorm):
            dimensions = len(layer.normalized_shape)
            if len(incoming) < dimensions or incoming[-dimensions:] != layer.normalized_shape:
                raise ValueError(
                    f"{name} expects trailing dimensions {layer.normalized_shape}, got {incoming}"
                )
        elif isinstance(layer, (MultiHeadSelfAttention, TransformerBlock)):
            if len(incoming) != 3 or incoming[-1] != layer.embedding_dim:
                raise ValueError(
                    f"{name} expects shape (batch, sequence, {layer.embedding_dim}), got {incoming}"
                )
        elif isinstance(layer, Sequential):
            nested = calculate_output_shapes(layer, incoming)
            results.extend(
                LayerShape(
                    name=f"{name}.{item.name}",
                    input_shape=item.input_shape,
                    output_shape=item.output_shape,
                    parameters=item.parameters,
                )
                for item in nested
            )
            if nested:
                current = nested[-1].output_shape
            continue
        else:
            current = incoming
        layer_parameters = sum(parameter.size for parameter in layer.parameters())
        results.append(
            LayerShape(
                name=name,
                input_shape=incoming,
                output_shape=current,
                parameters=layer_parameters,
            )
        )
    return results


def infer_output_shape(model: Layer, input_shape: tuple[int, ...]) -> tuple[int, ...]:
    shapes = calculate_output_shapes(model, input_shape)
    return shapes[-1].output_shape if shapes else input_shape


def estimate_array_memory(
    shape: tuple[int, ...],
    *,
    dtype: DTypeLike = np.float32,
    copies: int = 1,
) -> int:
    if copies < 0 or any(dimension < 0 for dimension in shape):
        raise ValueError("shape dimensions and copies cannot be negative")
    return int(np.prod(shape, dtype=np.int64)) * np.dtype(dtype).itemsize * copies


def estimate_model_memory(
    model: Layer,
    *,
    batch_size: int = 1,
    input_shape: tuple[int, ...] | None = None,
    dtype: DTypeLike = np.float32,
    include_gradients: bool = True,
    optimizer: str | None = None,
) -> MemoryEstimate:
    """Project parameter, gradient, optimizer-state, and activation memory."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    item_size = np.dtype(dtype).itemsize
    summary = parameter_summary(model)
    parameter_bytes = summary.total * item_size
    gradient_bytes = summary.trainable * item_size if include_gradients else 0
    optimizer_name = (optimizer or "none").lower()
    if optimizer_name in {"none", "sgd"}:
        optimizer_multiplier = 0
    elif optimizer_name in {"momentum", "sgd_momentum"}:
        optimizer_multiplier = 1
    elif optimizer_name == "adam":
        optimizer_multiplier = 2
    else:
        raise ValueError("optimizer must be one of: none, sgd, momentum, adam")
    optimizer_bytes = summary.trainable * item_size * optimizer_multiplier

    activation_bytes = 0
    if input_shape is not None:
        batched = (batch_size, *input_shape)
        activation_bytes += estimate_array_memory(batched, dtype=dtype)
        for layer_shape in calculate_output_shapes(model, batched):
            activation_bytes += estimate_array_memory(layer_shape.output_shape, dtype=dtype)
    total = parameter_bytes + gradient_bytes + optimizer_bytes + activation_bytes
    return MemoryEstimate(
        parameter_bytes=parameter_bytes,
        gradient_bytes=gradient_bytes,
        optimizer_bytes=optimizer_bytes,
        activation_bytes=activation_bytes,
        total_bytes=total,
    )


def format_bytes(byte_count: int) -> str:
    if byte_count < 0:
        raise ValueError("byte_count cannot be negative")
    value = float(byte_count)
    for suffix in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or suffix == "TiB":
            return f"{value:.2f} {suffix}"
        value /= 1024.0
    raise AssertionError("unreachable")
