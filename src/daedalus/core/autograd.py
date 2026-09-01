"""Small utilities shared by Daedalus' reverse-mode autograd engine.

The implementation deliberately depends only on NumPy.  Keeping graph
traversal and broadcasting reduction here makes the operation definitions in
``tensor.py`` compact and, more importantly, independently testable.
"""

from __future__ import annotations

from contextlib import AbstractContextManager, contextmanager
from contextvars import ContextVar
from typing import Iterator, Protocol, TypeVar

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.floating]


class GraphNode(Protocol):
    """The minimal graph surface needed for a reverse topological walk."""

    _parents: tuple["GraphNode", ...]


NodeT = TypeVar("NodeT", bound=GraphNode)
_GRAD_ENABLED: ContextVar[bool] = ContextVar("daedalus_grad_enabled", default=True)


def is_grad_enabled() -> bool:
    """Return whether newly-created tensors should retain graph history."""

    return _GRAD_ENABLED.get()


@contextmanager
def set_grad_enabled(enabled: bool) -> Iterator[None]:
    """Temporarily enable or disable graph construction."""

    token = _GRAD_ENABLED.set(bool(enabled))
    try:
        yield
    finally:
        _GRAD_ENABLED.reset(token)


def no_grad() -> AbstractContextManager[None]:
    """Context manager that disables graph creation for inference/evaluation."""

    return set_grad_enabled(False)


def unbroadcast(gradient: np.ndarray, target_shape: tuple[int, ...]) -> np.ndarray:
    """Reduce a broadcasted gradient back to an operand's original shape."""

    reduced = np.asarray(gradient)
    if target_shape == ():
        return np.asarray(reduced.sum()).reshape(())

    while reduced.ndim > len(target_shape):
        reduced = reduced.sum(axis=0)

    for axis, size in enumerate(target_shape):
        if size == 1 and reduced.shape[axis] != 1:
            reduced = reduced.sum(axis=axis, keepdims=True)

    return reduced.reshape(target_shape)


def topological_sort(root: NodeT) -> list[NodeT]:
    """Return graph nodes in parent-before-child topological order.

    An iterative walk avoids recursion limits for long educational examples.
    Object identity is used instead of equality because tensors contain NumPy
    arrays, whose elementwise equality is not a useful graph key.
    """

    ordered: list[NodeT] = []
    visited: set[int] = set()
    stack: list[tuple[NodeT, bool]] = [(root, False)]

    while stack:
        node, expanded = stack.pop()
        node_id = id(node)
        if expanded:
            if node_id not in visited:
                visited.add(node_id)
                ordered.append(node)
            continue
        if node_id in visited:
            continue
        stack.append((node, True))
        for parent in reversed(node._parents):
            if id(parent) not in visited:
                stack.append((parent, False))

    return ordered
