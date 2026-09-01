"""Layer and trainable-parameter abstractions."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from daedalus.core import Tensor


class Parameter(Tensor):
    """A leaf tensor intended to be updated by an optimizer."""

    def __init__(
        self,
        data: ArrayLike,
        *,
        requires_grad: bool = True,
        name: str | None = None,
    ) -> None:
        super().__init__(data, requires_grad=requires_grad, name=name)

    @property
    def trainable(self) -> bool:
        return self.requires_grad

    @trainable.setter
    def trainable(self, enabled: bool) -> None:
        self.requires_grad_(enabled)


class Layer:
    """Base class for composable differentiable modules."""

    def __init__(self) -> None:
        self.training = True

    def forward(self, inputs: Tensor) -> Tensor:
        raise NotImplementedError

    def __call__(self, inputs: Tensor | ArrayLike) -> Tensor:
        value = inputs if isinstance(inputs, Tensor) else Tensor(inputs)
        return self.forward(value)

    def _named_members(
        self,
        value: Any,
        prefix: str,
        seen: set[int],
    ) -> Iterator[tuple[str, Parameter]]:
        if isinstance(value, Parameter):
            if id(value) not in seen:
                seen.add(id(value))
                yield prefix, value
        elif isinstance(value, Layer):
            for name, child in vars(value).items():
                if name.startswith("_") or name == "training":
                    continue
                child_prefix = f"{prefix}.{name}" if prefix else name
                yield from self._named_members(child, child_prefix, seen)
        elif isinstance(value, Mapping):
            for name, child in value.items():
                child_prefix = f"{prefix}.{name}" if prefix else str(name)
                yield from self._named_members(child, child_prefix, seen)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, np.ndarray)):
            for index, child in enumerate(value):
                child_prefix = f"{prefix}.{index}" if prefix else str(index)
                yield from self._named_members(child, child_prefix, seen)

    def named_parameters(self, prefix: str = "") -> Iterator[tuple[str, Parameter]]:
        seen: set[int] = set()
        for name, value in vars(self).items():
            if name.startswith("_") or name == "training":
                continue
            member_prefix = f"{prefix}.{name}" if prefix else name
            yield from self._named_members(value, member_prefix, seen)

    def parameters(self, *, trainable_only: bool = False) -> list[Parameter]:
        parameters = [parameter for _, parameter in self.named_parameters()]
        if trainable_only:
            return [parameter for parameter in parameters if parameter.requires_grad]
        return parameters

    def children(self) -> Iterator[Layer]:
        seen: set[int] = set()

        def walk(value: Any) -> Iterator[Layer]:
            if isinstance(value, Layer):
                if id(value) not in seen:
                    seen.add(id(value))
                    yield value
            elif isinstance(value, Mapping):
                for child in value.values():
                    yield from walk(child)
            elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, np.ndarray)):
                for child in value:
                    yield from walk(child)

        for name, value in vars(self).items():
            if name.startswith("_") or name == "training":
                continue
            yield from walk(value)

    def zero_grad(self, *, set_to_none: bool = True) -> None:
        for parameter in self.parameters():
            parameter.zero_grad(set_to_none=set_to_none)

    def freeze(self) -> Layer:
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        return self

    def unfreeze(self) -> Layer:
        for parameter in self.parameters():
            parameter.requires_grad_(True)
        return self

    def train(self, mode: bool = True) -> Layer:
        self.training = bool(mode)
        for child in self.children():
            child.train(self.training)
        return self

    def eval(self) -> Layer:
        return self.train(False)

    def state_dict(self) -> dict[str, np.ndarray]:
        return {name: parameter.data.copy() for name, parameter in self.named_parameters()}

    def load_state_dict(self, state: Mapping[str, ArrayLike], *, strict: bool = True) -> None:
        current = dict(self.named_parameters())
        if strict:
            missing = sorted(set(current) - set(state))
            unexpected = sorted(set(state) - set(current))
            if missing or unexpected:
                raise KeyError(f"state mismatch: missing={missing}, unexpected={unexpected}")
        for name, value in state.items():
            if name not in current:
                continue
            array = np.asarray(value, dtype=current[name].dtype)
            if array.shape != current[name].shape:
                raise ValueError(
                    f"shape mismatch for {name!r}: expected {current[name].shape}, got {array.shape}"
                )
            current[name].data[...] = array
