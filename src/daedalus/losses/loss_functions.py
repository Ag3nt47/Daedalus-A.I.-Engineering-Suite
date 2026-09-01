"""Loss functions expressed through the Daedalus Tensor API."""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from daedalus.core import Tensor

Reduction = Literal["none", "mean", "sum"]


def _validate_reduction(reduction: str) -> Reduction:
    if reduction not in {"none", "mean", "sum"}:
        raise ValueError("reduction must be 'none', 'mean', or 'sum'")
    return reduction  # type: ignore[return-value]


class MSELoss:
    """Mean-squared error with configurable reduction."""

    def __init__(self, reduction: Reduction = "mean") -> None:
        self.reduction = _validate_reduction(reduction)

    def __call__(self, predictions: Tensor, targets: Tensor | ArrayLike) -> Tensor:
        target = targets if isinstance(targets, Tensor) else Tensor(targets)
        squared = (predictions - target) ** 2
        if self.reduction == "sum":
            return squared.sum()
        if self.reduction == "mean":
            return squared.mean()
        return squared


class CrossEntropyLoss:
    """Numerically stable softmax cross-entropy for logits.

    Targets may be integer class indices or one-hot/probability rows matching
    the logits shape.  The class dimension is the final dimension.
    """

    def __init__(self, reduction: Reduction = "mean") -> None:
        self.reduction = _validate_reduction(reduction)

    def __call__(self, logits: Tensor, targets: Tensor | ArrayLike) -> Tensor:
        if logits.ndim not in {1, 2}:
            raise ValueError("CrossEntropyLoss expects logits shaped (classes,) or (batch, classes)")
        single = logits.ndim == 1
        scores = logits.reshape(1, -1) if single else logits
        if scores.shape[-1] < 2:
            raise ValueError("CrossEntropyLoss requires at least two classes")

        target_array = targets.data if isinstance(targets, Tensor) else np.asarray(targets)
        maximum = np.max(scores.data, axis=-1, keepdims=True)
        shifted = scores - Tensor(maximum)
        log_probabilities = shifted - shifted.exp().sum(axis=-1, keepdims=True).log()

        if target_array.shape == scores.shape:
            weights = Tensor(np.asarray(target_array, dtype=scores.dtype))
            losses = -(log_probabilities * weights).sum(axis=-1)
        else:
            indices = np.asarray(target_array)
            if single and indices.ndim == 0:
                indices = indices.reshape(1)
            indices = indices.reshape(-1)
            if indices.shape[0] != scores.shape[0]:
                raise ValueError(
                    f"target count {indices.shape[0]} does not match batch size {scores.shape[0]}"
                )
            if not np.all(np.equal(indices, indices.astype(np.int64))):
                raise TypeError("class-index targets must contain integers")
            class_indices = indices.astype(np.int64)
            if np.any(class_indices < 0) or np.any(class_indices >= scores.shape[-1]):
                raise ValueError("target class index is out of range")
            losses = -log_probabilities[(np.arange(scores.shape[0]), class_indices)]

        if self.reduction == "sum":
            return losses.sum()
        if self.reduction == "mean":
            return losses.mean()
        return losses[0] if single else losses

