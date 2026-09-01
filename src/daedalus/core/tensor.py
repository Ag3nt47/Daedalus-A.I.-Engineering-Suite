"""A transparent NumPy tensor with a compact reverse-mode autograd engine."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Callable, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .autograd import is_grad_enabled, topological_sort, unbroadcast

Axis: TypeAlias = int | tuple[int, ...] | None
BackwardFn: TypeAlias = Callable[[np.ndarray], None]


def _floating_array(data: ArrayLike | NDArray[Any], *, copy: bool = False) -> np.ndarray:
    array = np.array(data, copy=copy) if copy else np.asarray(data)
    if array.dtype.kind not in {"f", "c"}:
        array = array.astype(np.float64)
    return array


class Tensor:
    """A NumPy-backed value that records operations for reverse differentiation.

    The public ``data`` and ``grad`` arrays are intentional: this project is an
    educational engineering suite, so state remains inspectable rather than
    hidden behind an opaque framework runtime.
    """

    __array_priority__ = 1000
    __hash__ = object.__hash__

    def __init__(
        self,
        data: ArrayLike | NDArray[Any],
        *,
        requires_grad: bool = False,
        dtype: np.dtype[Any] | type[Any] | None = None,
        name: str | None = None,
    ) -> None:
        array = np.asarray(data, dtype=dtype)
        if array.dtype.kind not in {"f", "c"}:
            array = array.astype(np.float64)
        self.data: np.ndarray = array
        self.requires_grad = bool(requires_grad)
        self.grad: np.ndarray | None = None
        self.name = name
        self._parents: tuple[Tensor, ...] = ()
        self._backward: BackwardFn = lambda _gradient: None
        self._op = "leaf"

    @classmethod
    def _from_operation(
        cls,
        data: np.ndarray,
        parents: tuple[Tensor, ...],
        backward: BackwardFn,
        op: str,
    ) -> Tensor:
        requires_grad = is_grad_enabled() and any(parent.requires_grad for parent in parents)
        output = cls(data, requires_grad=requires_grad)
        if requires_grad:
            output._parents = tuple(parent for parent in parents if parent.requires_grad)
            output._backward = backward
            output._op = op
        return output

    @staticmethod
    def ensure(value: Tensor | ArrayLike) -> Tensor:
        return value if isinstance(value, Tensor) else Tensor(value)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def size(self) -> int:
        return self.data.size

    @property
    def dtype(self) -> np.dtype[Any]:
        return self.data.dtype

    def numpy(self, *, copy: bool = True) -> np.ndarray:
        """Return the backing value, copied by default to prevent accidents."""

        return self.data.copy() if copy else self.data

    def item(self) -> float | complex:
        return self.data.item()

    def detach(self) -> Tensor:
        return Tensor(self.data.copy(), name=self.name)

    def requires_grad_(self, enabled: bool = True) -> Tensor:
        self.requires_grad = bool(enabled)
        if not enabled:
            self.grad = None
        return self

    def zero_grad(self, *, set_to_none: bool = True) -> None:
        if set_to_none:
            self.grad = None
        else:
            self.grad = np.zeros_like(self.data)

    def _accumulate_grad(self, gradient: np.ndarray) -> None:
        if not self.requires_grad:
            return
        value = np.asarray(gradient, dtype=self.data.dtype)
        if value.shape != self.shape:
            value = value.reshape(self.shape)
        if self.grad is None:
            self.grad = value.copy()
        else:
            self.grad += value

    def backward(self, gradient: ArrayLike | None = None) -> None:
        """Differentiate this tensor with respect to every reachable leaf.

        A gradient may be omitted only for scalar outputs.  Leaf gradients
        accumulate across calls, while transient graph-node gradients are
        recomputed so repeated backward passes do not double-count internals.
        """

        if not self.requires_grad:
            raise RuntimeError("cannot call backward() on a tensor that does not require gradients")
        if gradient is None:
            if self.size != 1:
                raise ValueError("a gradient is required for non-scalar outputs")
            seed = np.ones_like(self.data)
        else:
            seed = np.asarray(gradient, dtype=self.data.dtype)
            if seed.shape != self.shape:
                raise ValueError(f"gradient shape {seed.shape} does not match output shape {self.shape}")

        ordered = topological_sort(self)
        for node in ordered:
            if node._parents:
                node.grad = None

        if self._parents:
            self.grad = seed.copy()
        else:
            self._accumulate_grad(seed)

        for node in reversed(ordered):
            if node.grad is not None:
                node._backward(node.grad)

    def __add__(self, other: Tensor | ArrayLike) -> Tensor:
        rhs = Tensor.ensure(other)
        data = self.data + rhs.data

        def backward(gradient: np.ndarray) -> None:
            self._accumulate_grad(unbroadcast(gradient, self.shape))
            rhs._accumulate_grad(unbroadcast(gradient, rhs.shape))

        return Tensor._from_operation(data, (self, rhs), backward, "add")

    def __radd__(self, other: Tensor | ArrayLike) -> Tensor:
        return self + other

    def __neg__(self) -> Tensor:
        data = -self.data

        def backward(gradient: np.ndarray) -> None:
            self._accumulate_grad(-gradient)

        return Tensor._from_operation(data, (self,), backward, "neg")

    def __sub__(self, other: Tensor | ArrayLike) -> Tensor:
        return self + (-Tensor.ensure(other))

    def __rsub__(self, other: Tensor | ArrayLike) -> Tensor:
        return Tensor.ensure(other) - self

    def __mul__(self, other: Tensor | ArrayLike) -> Tensor:
        rhs = Tensor.ensure(other)
        data = self.data * rhs.data

        def backward(gradient: np.ndarray) -> None:
            self._accumulate_grad(unbroadcast(gradient * rhs.data, self.shape))
            rhs._accumulate_grad(unbroadcast(gradient * self.data, rhs.shape))

        return Tensor._from_operation(data, (self, rhs), backward, "mul")

    def __rmul__(self, other: Tensor | ArrayLike) -> Tensor:
        return self * other

    def __truediv__(self, other: Tensor | ArrayLike) -> Tensor:
        rhs = Tensor.ensure(other)
        data = self.data / rhs.data

        def backward(gradient: np.ndarray) -> None:
            self._accumulate_grad(unbroadcast(gradient / rhs.data, self.shape))
            rhs_grad = -gradient * self.data / np.square(rhs.data)
            rhs._accumulate_grad(unbroadcast(rhs_grad, rhs.shape))

        return Tensor._from_operation(data, (self, rhs), backward, "div")

    def __rtruediv__(self, other: Tensor | ArrayLike) -> Tensor:
        return Tensor.ensure(other) / self

    def __pow__(self, exponent: Tensor | ArrayLike) -> Tensor:
        rhs = Tensor.ensure(exponent)
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            data = np.power(self.data, rhs.data)

        def backward(gradient: np.ndarray) -> None:
            if self.requires_grad:
                base_grad = gradient * rhs.data * np.power(self.data, rhs.data - 1)
                self._accumulate_grad(unbroadcast(base_grad, self.shape))
            if rhs.requires_grad:
                with np.errstate(invalid="ignore", divide="ignore"):
                    exponent_grad = gradient * data * np.log(self.data)
                rhs._accumulate_grad(unbroadcast(exponent_grad, rhs.shape))

        return Tensor._from_operation(data, (self, rhs), backward, "pow")

    def __rpow__(self, base: Tensor | ArrayLike) -> Tensor:
        return Tensor.ensure(base) ** self

    def __matmul__(self, other: Tensor | ArrayLike) -> Tensor:
        rhs = Tensor.ensure(other)
        data = np.matmul(self.data, rhs.data)

        def backward(gradient: np.ndarray) -> None:
            left = self.data
            right = rhs.data
            left_was_vector = left.ndim == 1
            right_was_vector = right.ndim == 1
            left_matrix = np.expand_dims(left, -2) if left_was_vector else left
            right_matrix = np.expand_dims(right, -1) if right_was_vector else right

            if left_was_vector and right_was_vector:
                grad_matrix = np.asarray(gradient).reshape(1, 1)
            elif left_was_vector:
                grad_matrix = np.expand_dims(gradient, -2)
            elif right_was_vector:
                grad_matrix = np.expand_dims(gradient, -1)
            else:
                grad_matrix = gradient

            left_grad = np.matmul(grad_matrix, np.swapaxes(right_matrix, -1, -2))
            right_grad = np.matmul(np.swapaxes(left_matrix, -1, -2), grad_matrix)
            if left_was_vector:
                left_grad = np.squeeze(left_grad, axis=-2)
            if right_was_vector:
                right_grad = np.squeeze(right_grad, axis=-1)

            self._accumulate_grad(unbroadcast(left_grad, self.shape))
            rhs._accumulate_grad(unbroadcast(right_grad, rhs.shape))

        return Tensor._from_operation(data, (self, rhs), backward, "matmul")

    def __rmatmul__(self, other: Tensor | ArrayLike) -> Tensor:
        return Tensor.ensure(other) @ self

    def sum(self, axis: Axis = None, keepdims: bool = False) -> Tensor:
        data = self.data.sum(axis=axis, keepdims=keepdims)

        def backward(gradient: np.ndarray) -> None:
            expanded = gradient
            if axis is not None and not keepdims:
                axes = (axis,) if isinstance(axis, int) else axis
                normalized = sorted(item % self.ndim for item in axes)
                for item in normalized:
                    expanded = np.expand_dims(expanded, item)
            self._accumulate_grad(np.broadcast_to(expanded, self.shape))

        return Tensor._from_operation(np.asarray(data), (self,), backward, "sum")

    def mean(self, axis: Axis = None, keepdims: bool = False) -> Tensor:
        if axis is None:
            count = self.size
        else:
            axes = (axis,) if isinstance(axis, int) else axis
            count = int(np.prod([self.shape[item % self.ndim] for item in axes]))
        return self.sum(axis=axis, keepdims=keepdims) / count

    def reshape(self, *shape: int | Sequence[int]) -> Tensor:
        if len(shape) == 1 and isinstance(shape[0], Sequence):
            target = tuple(int(item) for item in shape[0])
        else:
            target = tuple(int(item) for item in shape)
        data = self.data.reshape(target)

        def backward(gradient: np.ndarray) -> None:
            self._accumulate_grad(gradient.reshape(self.shape))

        return Tensor._from_operation(data, (self,), backward, "reshape")

    def transpose(self, *axes: int | Sequence[int]) -> Tensor:
        if len(axes) == 1 and isinstance(axes[0], Sequence):
            permutation = tuple(int(item) for item in axes[0])
        elif axes:
            permutation = tuple(int(item) for item in axes)
        else:
            permutation = tuple(reversed(range(self.ndim)))
        data = self.data.transpose(permutation)
        inverse = tuple(int(item) for item in np.argsort(permutation))

        def backward(gradient: np.ndarray) -> None:
            self._accumulate_grad(gradient.transpose(inverse))

        return Tensor._from_operation(data, (self,), backward, "transpose")

    @property
    def T(self) -> Tensor:
        return self.transpose()

    def exp(self) -> Tensor:
        data = np.exp(self.data)

        def backward(gradient: np.ndarray) -> None:
            self._accumulate_grad(gradient * data)

        return Tensor._from_operation(data, (self,), backward, "exp")

    def log(self) -> Tensor:
        data = np.log(self.data)

        def backward(gradient: np.ndarray) -> None:
            self._accumulate_grad(gradient / self.data)

        return Tensor._from_operation(data, (self,), backward, "log")

    def tanh(self) -> Tensor:
        data = np.tanh(self.data)

        def backward(gradient: np.ndarray) -> None:
            self._accumulate_grad(gradient * (1.0 - np.square(data)))

        return Tensor._from_operation(data, (self,), backward, "tanh")

    def relu(self) -> Tensor:
        data = np.maximum(self.data, 0.0)

        def backward(gradient: np.ndarray) -> None:
            self._accumulate_grad(gradient * (self.data > 0.0))

        return Tensor._from_operation(data, (self,), backward, "relu")

    def sigmoid(self) -> Tensor:
        data = np.empty_like(self.data)
        positive = self.data >= 0
        data[positive] = 1.0 / (1.0 + np.exp(-self.data[positive]))
        negative_exp = np.exp(self.data[~positive])
        data[~positive] = negative_exp / (1.0 + negative_exp)

        def backward(gradient: np.ndarray) -> None:
            self._accumulate_grad(gradient * data * (1.0 - data))

        return Tensor._from_operation(data, (self,), backward, "sigmoid")

    def __getitem__(self, item: Any) -> Tensor:
        data = self.data[item]

        def backward(gradient: np.ndarray) -> None:
            parent_grad = np.zeros_like(self.data)
            np.add.at(parent_grad, item, gradient)
            self._accumulate_grad(parent_grad)

        return Tensor._from_operation(np.asarray(data), (self,), backward, "slice")

    def __len__(self) -> int:
        return len(self.data)

    def __array__(self, dtype: np.dtype[Any] | None = None) -> np.ndarray:
        return np.asarray(self.data, dtype=dtype)

    def __repr__(self) -> str:
        details = f", requires_grad={self.requires_grad}"
        if self.name:
            details += f", name={self.name!r}"
        return f"Tensor({self.data!r}{details})"
