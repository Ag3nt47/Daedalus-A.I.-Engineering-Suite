"""Fully-connected layer."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import DTypeLike

from daedalus.core import Tensor

from .base import Layer, Parameter


class Linear(Layer):
    """Apply ``inputs @ weight + bias`` along the final input dimension."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = True,
        seed: int | None = None,
        rng: np.random.Generator | None = None,
        dtype: DTypeLike = np.float64,
    ) -> None:
        super().__init__()
        if in_features <= 0 or out_features <= 0:
            raise ValueError("in_features and out_features must be positive")
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        generator = rng if rng is not None else np.random.default_rng(seed)
        limit = math.sqrt(6.0 / (self.in_features + self.out_features))
        values = generator.uniform(
            -limit,
            limit,
            size=(self.in_features, self.out_features),
        ).astype(dtype)
        self.weight = Parameter(values, name="weight")
        self.bias = (
            Parameter(np.zeros(self.out_features, dtype=dtype), name="bias") if bias else None
        )

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim == 0 or inputs.shape[-1] != self.in_features:
            actual = inputs.shape[-1] if inputs.ndim else None
            raise ValueError(f"Linear expected {self.in_features} input features, got {actual}")
        output = inputs @ self.weight
        return output + self.bias if self.bias is not None else output

    def __repr__(self) -> str:
        return (
            f"Linear(in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None})"
        )

