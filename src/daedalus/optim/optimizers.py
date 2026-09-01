"""NumPy implementations of common parameter optimizers."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from daedalus.layers import Parameter


class Optimizer:
    """Base optimizer with deterministic parameter ordering and deduplication."""

    def __init__(self, parameters: Iterable[Parameter]) -> None:
        self.parameters: list[Parameter] = []
        seen: set[int] = set()
        for parameter in parameters:
            if not isinstance(parameter, Parameter):
                raise TypeError("optimizers accept Parameter instances")
            if id(parameter) not in seen:
                seen.add(id(parameter))
                self.parameters.append(parameter)
        if not self.parameters:
            raise ValueError("optimizer received no parameters")

    def zero_grad(self, *, set_to_none: bool = True) -> None:
        for parameter in self.parameters:
            parameter.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        raise NotImplementedError


class SGD(Optimizer):
    """Stochastic gradient descent with optional momentum and Nesterov update."""

    def __init__(
        self,
        parameters: Iterable[Parameter],
        *,
        lr: float | None = None,
        learning_rate: float | None = None,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
    ) -> None:
        super().__init__(parameters)
        if lr is not None and learning_rate is not None:
            raise ValueError("provide either lr or learning_rate, not both")
        resolved_lr = 1e-2 if lr is None and learning_rate is None else (
            lr if lr is not None else learning_rate
        )
        assert resolved_lr is not None
        if resolved_lr <= 0:
            raise ValueError("lr must be positive")
        if not 0.0 <= momentum < 1.0:
            raise ValueError("momentum must be in [0, 1)")
        if weight_decay < 0:
            raise ValueError("weight_decay cannot be negative")
        if nesterov and momentum == 0:
            raise ValueError("Nesterov momentum requires momentum > 0")
        self.lr = float(resolved_lr)
        self.momentum = float(momentum)
        self.weight_decay = float(weight_decay)
        self.nesterov = bool(nesterov)
        self._velocity: dict[int, np.ndarray] = {}

    def step(self) -> None:
        for parameter in self.parameters:
            if not parameter.requires_grad or parameter.grad is None:
                continue
            gradient = np.asarray(parameter.grad, dtype=parameter.dtype)
            if not np.all(np.isfinite(gradient)):
                raise FloatingPointError("SGD received a non-finite gradient")
            if self.weight_decay:
                gradient = gradient + self.weight_decay * parameter.data
            if self.momentum:
                velocity = self._velocity.setdefault(id(parameter), np.zeros_like(parameter.data))
                velocity *= self.momentum
                velocity += gradient
                update = gradient + self.momentum * velocity if self.nesterov else velocity
            else:
                update = gradient
            parameter.data -= self.lr * update


class Adam(Optimizer):
    """Adam with bias correction and optional L2 weight decay."""

    def __init__(
        self,
        parameters: Iterable[Parameter],
        *,
        lr: float | None = None,
        learning_rate: float | None = None,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(parameters)
        beta1, beta2 = betas
        if lr is not None and learning_rate is not None:
            raise ValueError("provide either lr or learning_rate, not both")
        resolved_lr = 1e-3 if lr is None and learning_rate is None else (
            lr if lr is not None else learning_rate
        )
        assert resolved_lr is not None
        if resolved_lr <= 0:
            raise ValueError("lr must be positive")
        if not 0.0 <= beta1 < 1.0 or not 0.0 <= beta2 < 1.0:
            raise ValueError("Adam betas must be in [0, 1)")
        if eps <= 0:
            raise ValueError("eps must be positive")
        if weight_decay < 0:
            raise ValueError("weight_decay cannot be negative")
        self.lr = float(resolved_lr)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)
        self._step = 0
        self._parameter_steps: dict[int, int] = {}
        self._first_moment: dict[int, np.ndarray] = {}
        self._second_moment: dict[int, np.ndarray] = {}

    def step(self) -> None:
        self._step += 1
        for parameter in self.parameters:
            if not parameter.requires_grad or parameter.grad is None:
                continue
            gradient = np.asarray(parameter.grad, dtype=parameter.dtype)
            if not np.all(np.isfinite(gradient)):
                raise FloatingPointError("Adam received a non-finite gradient")
            if self.weight_decay:
                gradient = gradient + self.weight_decay * parameter.data

            parameter_id = id(parameter)
            parameter_step = self._parameter_steps.get(parameter_id, 0) + 1
            self._parameter_steps[parameter_id] = parameter_step
            correction1 = 1.0 - self.beta1**parameter_step
            correction2 = 1.0 - self.beta2**parameter_step

            first = self._first_moment.setdefault(parameter_id, np.zeros_like(parameter.data))
            second = self._second_moment.setdefault(parameter_id, np.zeros_like(parameter.data))
            first *= self.beta1
            first += (1.0 - self.beta1) * gradient
            second *= self.beta2
            second += (1.0 - self.beta2) * np.square(gradient)
            first_hat = first / correction1
            second_hat = second / correction2
            parameter.data -= self.lr * first_hat / (np.sqrt(second_hat) + self.eps)
