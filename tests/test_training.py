from __future__ import annotations

import numpy as np

from daedalus.engine import Callback, EarlyStopping, Trainer, make_regression, make_xor
from daedalus.layers import Linear, Sequential, Tanh
from daedalus.losses import CrossEntropyLoss, MSELoss
from daedalus.optim import SGD, Adam


def test_toy_datasets_are_seed_deterministic() -> None:
    xor_a = make_xor(41, seed=9)
    xor_b = make_xor(41, seed=9)
    regression_a = make_regression(30, 3, seed=11)
    regression_b = make_regression(30, 3, seed=11)
    for left, right in (*zip(xor_a, xor_b, strict=True), *zip(regression_a, regression_b, strict=True)):
        np.testing.assert_array_equal(left, right)


def test_adam_trainer_converges_on_regression() -> None:
    features, targets = make_regression(120, 2, noise=0.01, seed=2)
    model = Sequential(Linear(2, 1, seed=3))
    trainer = Trainer(model, MSELoss(), Adam(model.parameters(), lr=0.05), seed=4)
    history = trainer.fit(features, targets, epochs=80, batch_size=20)

    assert len(history.epochs) == 80
    assert history.loss[-1] < history.loss[0] * 0.01
    assert trainer.evaluate(features, targets) < 5e-4


def test_xor_training_reaches_high_accuracy() -> None:
    features, targets = make_xor(160, noise=0.06, seed=5)
    model = Sequential(Linear(2, 8, seed=6), Tanh(), Linear(8, 2, seed=7))
    trainer = Trainer(
        model,
        CrossEntropyLoss(),
        Adam(model.parameters(), lr=0.03),
        seed=8,
    )
    history = trainer.fit(features, targets, epochs=60, batch_size=16)
    accuracy = np.mean(trainer.predict(features).argmax(axis=1) == targets)
    assert history.loss[-1] < 0.05
    assert accuracy > 0.97


class RecordingCallback(Callback):
    def __init__(self) -> None:
        self.epochs: list[int] = []
        self.batch_count = 0

    def on_batch_end(self, trainer: Trainer, logs: dict[str, float]) -> None:
        self.batch_count += 1

    def on_epoch_end(self, trainer: Trainer, logs: dict[str, float]) -> None:
        self.epochs.append(int(logs["epoch"]))


def test_callbacks_fire_and_frozen_layer_does_not_change() -> None:
    features, targets = make_regression(48, 2, noise=0.0, seed=10)
    model = Sequential(Linear(2, 3, seed=11), Tanh(), Linear(3, 1, seed=12))
    frozen_before = model.layers[0].state_dict()
    callback = RecordingCallback()
    trainer = Trainer(model, MSELoss(), SGD(model.parameters(), lr=0.02), seed=13)
    trainer.fit(
        features,
        targets,
        epochs=5,
        batch_size=12,
        callbacks=[callback],
        frozen_layers=[0],
    )

    assert callback.epochs == [0, 1, 2, 3, 4]
    assert callback.batch_count == 20
    for name, parameter in model.layers[0].named_parameters():
        np.testing.assert_array_equal(parameter.data, frozen_before[name])
    assert any(parameter.grad is not None for parameter in model.layers[2].parameters())


def test_sgd_momentum_update_matches_manual_steps() -> None:
    layer = Linear(1, 1, bias=False, seed=14)
    layer.weight.data[...] = 1.0
    optimizer = SGD(layer.parameters(), lr=0.1, momentum=0.5)
    layer.weight.grad = np.array([[2.0]])
    optimizer.step()
    np.testing.assert_allclose(layer.weight.data, [[0.8]])
    layer.weight.grad = np.array([[2.0]])
    optimizer.step()
    np.testing.assert_allclose(layer.weight.data, [[0.5]])


class ScriptedMetric(Callback):
    def __init__(self, values: list[float]) -> None:
        self.values = values

    def on_epoch_end(self, trainer: Trainer, logs: dict[str, float]) -> None:
        epoch = int(logs["epoch"])
        logs["scripted"] = self.values[epoch]
        trainer.model.parameters()[0].data.fill(epoch + 1)


def test_early_stopping_restores_best_weights() -> None:
    features = np.arange(8, dtype=np.float64).reshape(-1, 1)
    targets = features * 0.5
    model = Sequential(Linear(1, 1, bias=False, seed=20))
    early = EarlyStopping(
        monitor="scripted",
        patience=2,
        mode="min",
        restore_best=True,
    )
    trainer = Trainer(
        model,
        MSELoss(),
        SGD(model.parameters(), lr=0.001),
        callbacks=[ScriptedMetric([3.0, 2.0, 2.1, 2.2, 2.3]), early],
    )
    history = trainer.fit(features, targets, epochs=5, batch_size=4, shuffle=False)

    assert len(history.epochs) == 4
    assert early.best_epoch == 2
    assert early.best_value == 2.0
    assert early.stopped_epoch == 4
    np.testing.assert_array_equal(model.parameters()[0].data, [[2.0]])


def test_early_stopping_requires_monitored_metric() -> None:
    features, targets = make_regression(12, 1, seed=21)
    model = Sequential(Linear(1, 1, seed=22))
    trainer = Trainer(
        model,
        MSELoss(),
        SGD(model.parameters(), lr=0.001),
        callbacks=[EarlyStopping(monitor="missing")],
    )
    with np.testing.assert_raises_regex(ValueError, "not in epoch logs"):
        trainer.fit(features, targets, epochs=1)
