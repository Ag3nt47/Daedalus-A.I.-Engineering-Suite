"""Deterministic mini-batch training with callbacks and layer freezing."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, Protocol

import numpy as np
from numpy.typing import ArrayLike

from daedalus.core import Tensor, no_grad
from daedalus.layers import Layer, Sequential
from daedalus.optim import Optimizer


class LossFunction(Protocol):
    def __call__(self, predictions: Tensor, targets: Tensor | ArrayLike) -> Tensor: ...


Metric = Callable[[np.ndarray, np.ndarray], float]


class History(dict[str, list[float]]):
    """Dictionary-compatible metric history with explicit epoch numbers."""

    def __init__(self) -> None:
        super().__init__()
        self.epochs: list[int] = []

    def record(self, epoch: int, values: Mapping[str, float]) -> None:
        self.epochs.append(int(epoch))
        for name, value in values.items():
            self.setdefault(name, []).append(float(value))

    @property
    def loss(self) -> list[float]:
        return self.get("loss", [])


class Callback:
    """Subclass and override only the lifecycle events of interest."""

    def on_train_begin(self, trainer: Trainer, logs: Mapping[str, Any]) -> None:
        pass

    def on_epoch_begin(self, trainer: Trainer, logs: Mapping[str, Any]) -> None:
        pass

    def on_batch_end(self, trainer: Trainer, logs: Mapping[str, Any]) -> None:
        pass

    def on_epoch_end(self, trainer: Trainer, logs: Mapping[str, Any]) -> None:
        pass

    def on_train_end(self, trainer: Trainer, logs: Mapping[str, Any]) -> None:
        pass


class EarlyStopping(Callback):
    """Stop when a scalar metric stalls and optionally restore the best model state."""

    def __init__(
        self,
        *,
        monitor: str = "val_loss",
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = "min",
        restore_best: bool = True,
    ) -> None:
        if not str(monitor).strip():
            raise ValueError("monitor cannot be empty")
        if int(patience) < 0:
            raise ValueError("patience cannot be negative")
        if not np.isfinite(min_delta) or float(min_delta) < 0:
            raise ValueError("min_delta must be finite and non-negative")
        if mode not in {"min", "max"}:
            raise ValueError("mode must be 'min' or 'max'")
        self.monitor = str(monitor).strip()
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.mode = mode
        self.restore_best = bool(restore_best)
        self.best_epoch: int | None = None
        self.best_value: float | None = None
        self.stopped_epoch: int | None = None
        self._wait = 0
        self._best_state: dict[str, np.ndarray] | None = None

    def on_train_begin(self, trainer: Trainer, logs: Mapping[str, Any]) -> None:
        self.best_epoch = None
        self.best_value = None
        self.stopped_epoch = None
        self._wait = 0
        self._best_state = None

    def _is_improvement(self, value: float) -> bool:
        if self.best_value is None:
            return True
        if self.mode == "min":
            return value < self.best_value - self.min_delta
        return value > self.best_value + self.min_delta

    def on_epoch_end(self, trainer: Trainer, logs: Mapping[str, Any]) -> None:
        if self.monitor not in logs:
            raise ValueError(f"EarlyStopping monitor {self.monitor!r} is not in epoch logs")
        value = float(logs[self.monitor])
        if not np.isfinite(value):
            raise FloatingPointError(
                f"EarlyStopping monitor {self.monitor!r} became non-finite"
            )
        epoch = int(logs.get("epoch", len(trainer.history.epochs) - 1)) + 1
        if self._is_improvement(value):
            self.best_value = value
            self.best_epoch = epoch
            self._wait = 0
            if self.restore_best:
                self._best_state = trainer.model.state_dict()
            return
        self._wait += 1
        threshold = 1 if self.patience == 0 else self.patience
        if self._wait >= threshold:
            self.stopped_epoch = epoch
            trainer.stop_training = True

    def on_train_end(self, trainer: Trainer, logs: Mapping[str, Any]) -> None:
        if self.restore_best and self._best_state is not None:
            trainer.model.load_state_dict(self._best_state)


class Trainer:
    """Coordinate a model, loss, and optimizer without a framework runtime."""

    def __init__(
        self,
        model: Layer,
        loss: LossFunction,
        optimizer: Optimizer,
        *,
        seed: int = 0,
        callbacks: Sequence[Callback] | None = None,
        metrics: Mapping[str, Metric] | None = None,
    ) -> None:
        self.model = model
        self.loss_function = loss
        self.optimizer = optimizer
        self.seed = int(seed)
        self.callbacks = list(callbacks or ())
        self.metrics = dict(metrics or {})
        self.stop_training = False
        self.history = History()

    def _notify(
        self,
        event: str,
        callbacks: Sequence[Callback],
        logs: Mapping[str, Any],
    ) -> None:
        for callback in callbacks:
            method = getattr(callback, event, None)
            if method is not None:
                method(self, logs)

    def _resolve_layers(self, layers: Iterable[int | Layer]) -> list[Layer]:
        resolved: list[Layer] = []
        for layer in layers:
            if isinstance(layer, int):
                if not isinstance(self.model, Sequential):
                    raise TypeError("integer layer selection requires a Sequential model")
                resolved.append(self.model.layers[layer])
            elif isinstance(layer, Layer):
                resolved.append(layer)
            else:
                raise TypeError("frozen layers must be layer indices or Layer objects")
        return resolved

    def freeze_layers(self, layers: Iterable[int | Layer]) -> None:
        for layer in self._resolve_layers(layers):
            layer.freeze()

    def unfreeze_layers(self, layers: Iterable[int | Layer]) -> None:
        for layer in self._resolve_layers(layers):
            layer.unfreeze()

    def fit(
        self,
        features: ArrayLike,
        targets: ArrayLike,
        *,
        epochs: int = 100,
        batch_size: int | None = None,
        shuffle: bool = True,
        validation_data: tuple[ArrayLike, ArrayLike] | None = None,
        callbacks: Sequence[Callback] | None = None,
        frozen_layers: Iterable[int | Layer] | None = None,
        seed: int | None = None,
        verbose: bool = False,
    ) -> History:
        x = np.asarray(features)
        y = np.asarray(targets)
        if x.ndim == 0 or y.ndim == 0 or len(x) != len(y):
            raise ValueError("features and targets must be aligned sample arrays")
        if len(x) == 0:
            raise ValueError("training data cannot be empty")
        if epochs <= 0:
            raise ValueError("epochs must be positive")
        resolved_batch = len(x) if batch_size is None else int(batch_size)
        if resolved_batch <= 0:
            raise ValueError("batch_size must be positive")
        if frozen_layers is not None:
            self.freeze_layers(frozen_layers)

        active_callbacks = [*self.callbacks, *(callbacks or ())]
        self.history = History()
        self.stop_training = False
        self.model.train()
        generator = np.random.default_rng(self.seed if seed is None else int(seed))
        self._notify("on_train_begin", active_callbacks, {"epochs": epochs})

        for epoch in range(epochs):
            self._notify("on_epoch_begin", active_callbacks, {"epoch": epoch})
            order = generator.permutation(len(x)) if shuffle else np.arange(len(x))
            total_loss = 0.0
            observed = 0
            for batch_index, start in enumerate(range(0, len(x), resolved_batch)):
                indices = order[start : start + resolved_batch]
                batch_x = Tensor(x[indices])
                batch_y = y[indices]
                self.optimizer.zero_grad()
                predictions = self.model(batch_x)
                loss = self.loss_function(predictions, batch_y)
                if loss.size != 1:
                    raise ValueError("training loss must reduce to a scalar")
                loss.backward()
                self.optimizer.step()
                batch_loss = float(np.real(loss.item()))
                total_loss += batch_loss * len(indices)
                observed += len(indices)
                self._notify(
                    "on_batch_end",
                    active_callbacks,
                    {
                        "epoch": epoch,
                        "batch": batch_index,
                        "batch_size": len(indices),
                        "loss": batch_loss,
                    },
                )

            logs: dict[str, float] = {"loss": total_loss / observed}
            predictions = self.predict(x)
            for name, metric in self.metrics.items():
                logs[name] = float(metric(predictions, y))
            if validation_data is not None:
                logs["val_loss"] = self.evaluate(*validation_data)
            self.history.record(epoch, logs)
            epoch_logs: dict[str, Any] = {"epoch": epoch, **logs}
            self._notify("on_epoch_end", active_callbacks, epoch_logs)
            if verbose:
                metrics = ", ".join(f"{name}={value:.6f}" for name, value in logs.items())
                print(f"epoch {epoch + 1}/{epochs}: {metrics}")
            if self.stop_training:
                break

        self._notify(
            "on_train_end",
            active_callbacks,
            {"epochs_completed": len(self.history.epochs), **{k: v[-1] for k, v in self.history.items()}},
        )
        return self.history

    def predict(self, features: ArrayLike) -> np.ndarray:
        was_training = self.model.training
        self.model.eval()
        with no_grad():
            output = self.model(Tensor(np.asarray(features))).data.copy()
        self.model.train(was_training)
        return output

    def evaluate(self, features: ArrayLike, targets: ArrayLike) -> float:
        was_training = self.model.training
        self.model.eval()
        with no_grad():
            predictions = self.model(Tensor(np.asarray(features)))
            loss = self.loss_function(predictions, np.asarray(targets))
        self.model.train(was_training)
        if loss.size != 1:
            raise ValueError("evaluation loss must reduce to a scalar")
        return float(np.real(loss.item()))
