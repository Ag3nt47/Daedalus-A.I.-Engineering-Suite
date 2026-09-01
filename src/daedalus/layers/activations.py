"""Activation layers built directly from Tensor operations."""

from __future__ import annotations

import numpy as np

from daedalus.core import Tensor

from .base import Layer


class ReLU(Layer):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs.relu()


class Sigmoid(Layer):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs.sigmoid()


class Tanh(Layer):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs.tanh()


class Softmax(Layer):
    """Numerically stable softmax over a configurable axis."""

    def __init__(self, axis: int = -1) -> None:
        super().__init__()
        self.axis = axis

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim == 0:
            raise ValueError("Softmax requires at least one dimension")
        axis = self.axis % inputs.ndim
        maximum = np.max(inputs.data, axis=axis, keepdims=True)
        shifted = inputs - Tensor(maximum)
        exponentials = shifted.exp()
        return exponentials / exponentials.sum(axis=axis, keepdims=True)

