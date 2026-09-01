"""Layer containers."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence

from daedalus.core import Tensor

from .base import Layer


class Sequential(Layer):
    """Run a sequence of layers in order."""

    def __init__(self, *layers: Layer | Sequence[Layer]) -> None:
        super().__init__()
        if len(layers) == 1 and isinstance(layers[0], Sequence):
            resolved = list(layers[0])
        else:
            resolved = list(layers)  # type: ignore[list-item]
        if not all(isinstance(layer, Layer) for layer in resolved):
            raise TypeError("Sequential accepts Layer instances only")
        self.layers: list[Layer] = resolved

    def forward(self, inputs: Tensor) -> Tensor:
        output = inputs
        for layer in self.layers:
            output = layer(output)
        return output

    def append(self, layer: Layer) -> None:
        if not isinstance(layer, Layer):
            raise TypeError("only Layer instances can be appended")
        self.layers.append(layer)

    def freeze_layers(self, indices: Iterable[int]) -> None:
        for index in indices:
            self.layers[index].freeze()

    def unfreeze_layers(self, indices: Iterable[int] | None = None) -> None:
        targets = range(len(self.layers)) if indices is None else indices
        for index in targets:
            self.layers[index].unfreeze()

    def __getitem__(self, index: int | slice) -> Layer | Sequential:
        if isinstance(index, slice):
            return Sequential(self.layers[index])
        return self.layers[index]

    def __iter__(self) -> Iterator[Layer]:
        return iter(self.layers)

    def __len__(self) -> int:
        return len(self.layers)

    def __repr__(self) -> str:
        body = ", ".join(repr(layer) for layer in self.layers)
        return f"Sequential({body})"

