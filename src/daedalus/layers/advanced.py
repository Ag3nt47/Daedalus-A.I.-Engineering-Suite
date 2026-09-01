"""Advanced educational layers assembled from Daedalus and NumPy primitives.

The transformer implementation intentionally uses a pre-normalization layout:
LayerNorm -> attention -> residual, followed by LayerNorm -> feed-forward ->
residual.  Pre-norm is compact, stable for small teaching models, and keeps the
data flow easy to inspect.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, DTypeLike

from daedalus.core import Tensor

from .activations import ReLU, Softmax
from .base import Layer, Parameter
from .linear import Linear


class LayerConfigurationError(ValueError):
    """Raised when a layer cannot be constructed with the supplied settings."""


class LayerInputError(ValueError):
    """Raised when input data is incompatible with a configured layer."""


class Flatten(Layer):
    """Flatten a contiguous range of dimensions while preserving gradients."""

    def __init__(self, start_dim: int = 1, end_dim: int = -1) -> None:
        super().__init__()
        self.start_dim = int(start_dim)
        self.end_dim = int(end_dim)

    def output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        ndim = len(input_shape)
        start = self.start_dim + ndim if self.start_dim < 0 else self.start_dim
        end = self.end_dim + ndim if self.end_dim < 0 else self.end_dim
        if not 0 <= start < ndim or not 0 <= end < ndim or start > end:
            raise LayerInputError(
                f"Flatten dimensions [{self.start_dim}, {self.end_dim}] are invalid "
                f"for an input with {ndim} dimensions"
            )
        flattened = int(np.prod(input_shape[start : end + 1], dtype=np.int64))
        return (*input_shape[:start], flattened, *input_shape[end + 1 :])

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs.reshape(self.output_shape(inputs.shape))


class Dropout(Layer):
    """Inverted dropout with a reproducible per-layer random stream."""

    def __init__(self, probability: float = 0.5, *, seed: int = 0) -> None:
        super().__init__()
        if not 0.0 <= probability < 1.0:
            raise LayerConfigurationError("dropout probability must be in [0, 1)")
        self.probability = float(probability)
        self.seed = int(seed)
        self._generator = np.random.default_rng(self.seed)

    def reset_seed(self, seed: int | None = None) -> None:
        """Restart the deterministic mask stream, optionally with a new seed."""

        if seed is not None:
            self.seed = int(seed)
        self._generator = np.random.default_rng(self.seed)

    def forward(self, inputs: Tensor) -> Tensor:
        if not self.training or self.probability == 0.0:
            return inputs
        keep_probability = 1.0 - self.probability
        mask = (self._generator.random(inputs.shape) < keep_probability).astype(inputs.dtype)
        return inputs * Tensor(mask / keep_probability)


class LayerNorm(Layer):
    """Normalize the final dimensions and learn an elementwise scale and shift."""

    def __init__(
        self,
        normalized_shape: int | Sequence[int],
        *,
        epsilon: float = 1e-5,
        elementwise_affine: bool = True,
        dtype: DTypeLike = np.float64,
    ) -> None:
        super().__init__()
        shape = (normalized_shape,) if isinstance(normalized_shape, int) else tuple(normalized_shape)
        if not shape or any(dimension <= 0 for dimension in shape):
            raise LayerConfigurationError("normalized_shape must contain positive dimensions")
        if epsilon <= 0:
            raise LayerConfigurationError("LayerNorm epsilon must be positive")
        self.normalized_shape = tuple(int(dimension) for dimension in shape)
        self.epsilon = float(epsilon)
        self.scale = (
            Parameter(np.ones(self.normalized_shape, dtype=dtype), name="scale")
            if elementwise_affine
            else None
        )
        self.shift = (
            Parameter(np.zeros(self.normalized_shape, dtype=dtype), name="shift")
            if elementwise_affine
            else None
        )

    def forward(self, inputs: Tensor) -> Tensor:
        dimensions = len(self.normalized_shape)
        if inputs.ndim < dimensions or inputs.shape[-dimensions:] != self.normalized_shape:
            raise LayerInputError(
                f"LayerNorm expected trailing dimensions {self.normalized_shape}, "
                f"got input shape {inputs.shape}"
            )
        axes = tuple(range(inputs.ndim - dimensions, inputs.ndim))
        mean = inputs.mean(axis=axes, keepdims=True)
        centered = inputs - mean
        variance = (centered**2).mean(axis=axes, keepdims=True)
        normalized = centered / ((variance + self.epsilon) ** 0.5)
        if self.scale is not None and self.shift is not None:
            return normalized * self.scale + self.shift
        return normalized


class Embedding(Layer):
    """Map integer token IDs to trainable vectors using differentiable lookup."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        padding_index: int | None = None,
        seed: int = 0,
        dtype: DTypeLike = np.float64,
    ) -> None:
        super().__init__()
        if num_embeddings <= 0 or embedding_dim <= 0:
            raise LayerConfigurationError("embedding table dimensions must be positive")
        if padding_index is not None and not 0 <= padding_index < num_embeddings:
            raise LayerConfigurationError("padding_index must identify a row in the embedding table")
        self.num_embeddings = int(num_embeddings)
        self.embedding_dim = int(embedding_dim)
        self.padding_index = padding_index
        generator = np.random.default_rng(seed)
        scale = 1.0 / math.sqrt(self.embedding_dim)
        values = generator.normal(
            0.0,
            scale,
            size=(self.num_embeddings, self.embedding_dim),
        ).astype(dtype)
        if padding_index is not None:
            values[padding_index] = 0.0
        self.weight = Parameter(values, name="weight")

    def forward(self, inputs: Tensor) -> Tensor:
        raw = inputs.data
        if not np.all(np.isfinite(raw)) or not np.all(raw == np.floor(raw)):
            raise LayerInputError("Embedding inputs must contain finite integer token IDs")
        indices = raw.astype(np.int64)
        if np.any(indices < 0) or np.any(indices >= self.num_embeddings):
            raise LayerInputError(
                f"Embedding token IDs must be in [0, {self.num_embeddings - 1}]"
            )
        output = self.weight[indices]
        if self.padding_index is not None:
            mask = (indices != self.padding_index).astype(output.dtype)
            output = output * Tensor(np.expand_dims(mask, axis=-1))
        return output


class MultiHeadSelfAttention(Layer):
    """Scaled dot-product multi-head self-attention.

    Set ``causal=True`` at construction or pass a temporary override to
    ``forward``/``__call__``.  ``last_attention_weights`` stores a detached
    copy for educational visualization and diagnostics.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        *,
        causal: bool = False,
        bias: bool = True,
        seed: int = 0,
        dtype: DTypeLike = np.float64,
    ) -> None:
        super().__init__()
        if embedding_dim <= 0 or num_heads <= 0:
            raise LayerConfigurationError("embedding_dim and num_heads must be positive")
        if embedding_dim % num_heads:
            raise LayerConfigurationError("embedding_dim must be divisible by num_heads")
        self.embedding_dim = int(embedding_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.embedding_dim // self.num_heads
        self.causal = bool(causal)
        generator = np.random.default_rng(seed)
        self.query = Linear(
            self.embedding_dim,
            self.embedding_dim,
            bias=bias,
            rng=generator,
            dtype=dtype,
        )
        self.key = Linear(
            self.embedding_dim,
            self.embedding_dim,
            bias=bias,
            rng=generator,
            dtype=dtype,
        )
        self.value = Linear(
            self.embedding_dim,
            self.embedding_dim,
            bias=bias,
            rng=generator,
            dtype=dtype,
        )
        self.output = Linear(
            self.embedding_dim,
            self.embedding_dim,
            bias=bias,
            rng=generator,
            dtype=dtype,
        )
        self.last_attention_weights: np.ndarray | None = None

    def __call__(
        self,
        inputs: Tensor | ArrayLike,
        *,
        causal: bool | None = None,
    ) -> Tensor:
        value = inputs if isinstance(inputs, Tensor) else Tensor(inputs)
        return self.forward(value, causal=causal)

    def forward(self, inputs: Tensor, *, causal: bool | None = None) -> Tensor:
        if inputs.ndim != 3:
            raise LayerInputError(
                "MultiHeadSelfAttention expects input shaped (batch, sequence, embedding)"
            )
        batch_size, sequence_length, embedding_dim = inputs.shape
        if embedding_dim != self.embedding_dim:
            raise LayerInputError(
                f"attention expected embedding dimension {self.embedding_dim}, got {embedding_dim}"
            )

        def split_heads(projected: Tensor) -> Tensor:
            return projected.reshape(
                batch_size,
                sequence_length,
                self.num_heads,
                self.head_dim,
            ).transpose(0, 2, 1, 3)

        query = split_heads(self.query(inputs))
        key = split_heads(self.key(inputs))
        value = split_heads(self.value(inputs))
        scores = (query @ key.transpose(0, 1, 3, 2)) / math.sqrt(self.head_dim)

        use_causal_mask = self.causal if causal is None else bool(causal)
        if use_causal_mask:
            blocked = np.triu(np.ones((sequence_length, sequence_length), dtype=bool), k=1)
            additive_mask = np.where(blocked, -1e9, 0.0).reshape(
                1,
                1,
                sequence_length,
                sequence_length,
            )
            scores = scores + Tensor(additive_mask)

        weights = Softmax(axis=-1)(scores)
        self.last_attention_weights = weights.data.copy()
        context = weights @ value
        merged = context.transpose(0, 2, 1, 3).reshape(
            batch_size,
            sequence_length,
            self.embedding_dim,
        )
        return self.output(merged)


class TransformerBlock(Layer):
    """A small pre-normalization transformer encoder block."""

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        *,
        feed_forward_dim: int | None = None,
        dropout: float = 0.0,
        causal: bool = False,
        bias: bool = True,
        seed: int = 0,
        dtype: DTypeLike = np.float64,
    ) -> None:
        super().__init__()
        hidden_dim = 4 * embedding_dim if feed_forward_dim is None else feed_forward_dim
        if hidden_dim <= 0:
            raise LayerConfigurationError("feed_forward_dim must be positive")
        generator = np.random.default_rng(seed)

        def next_seed() -> int:
            return int(generator.integers(0, np.iinfo(np.int32).max))

        self.embedding_dim = int(embedding_dim)
        self.feed_forward_dim = int(hidden_dim)
        self.normalization1 = LayerNorm(embedding_dim, dtype=dtype)
        self.attention = MultiHeadSelfAttention(
            embedding_dim,
            num_heads,
            causal=causal,
            bias=bias,
            seed=next_seed(),
            dtype=dtype,
        )
        self.attention_dropout = Dropout(dropout, seed=next_seed())
        self.normalization2 = LayerNorm(embedding_dim, dtype=dtype)
        self.feed_forward_input = Linear(
            embedding_dim,
            hidden_dim,
            bias=bias,
            seed=next_seed(),
            dtype=dtype,
        )
        self.activation = ReLU()
        self.feed_forward_output = Linear(
            hidden_dim,
            embedding_dim,
            bias=bias,
            seed=next_seed(),
            dtype=dtype,
        )
        self.feed_forward_dropout = Dropout(dropout, seed=next_seed())

    @property
    def last_attention_weights(self) -> np.ndarray | None:
        return self.attention.last_attention_weights

    def forward(self, inputs: Tensor) -> Tensor:
        attended = self.attention(self.normalization1(inputs))
        residual = inputs + self.attention_dropout(attended)
        hidden = self.feed_forward_input(self.normalization2(residual))
        transformed = self.feed_forward_output(self.activation(hidden))
        return residual + self.feed_forward_dropout(transformed)

