from __future__ import annotations

import numpy as np
import pytest

from daedalus.core import Tensor
from daedalus.layers import Linear, ReLU, Sequential, Sigmoid, Softmax, Tanh
from daedalus.losses import CrossEntropyLoss, MSELoss


def test_linear_forward_shape_and_parameter_gradients() -> None:
    layer = Linear(3, 2, seed=4)
    inputs = Tensor(np.arange(12.0).reshape(4, 3), requires_grad=True)
    output = layer(inputs)
    output.sum().backward()

    assert output.shape == (4, 2)
    assert layer.weight.grad is not None and layer.weight.grad.shape == (3, 2)
    assert layer.bias is not None and layer.bias.grad is not None
    assert inputs.grad is not None and inputs.grad.shape == inputs.shape


def test_linear_rejects_incompatible_feature_count() -> None:
    with pytest.raises(ValueError, match="expected 3"):
        Linear(3, 2, seed=1)(Tensor(np.zeros((4, 4))))


def test_activation_layers_match_numpy() -> None:
    values = Tensor([-2.0, 0.0, 2.0])
    np.testing.assert_allclose(ReLU()(values).data, [0.0, 0.0, 2.0])
    np.testing.assert_allclose(Tanh()(values).data, np.tanh(values.data))
    np.testing.assert_allclose(Sigmoid()(values).data, 1.0 / (1.0 + np.exp(-values.data)))


def test_softmax_is_stable_normalized_and_differentiable() -> None:
    logits = Tensor([[1000.0, 1001.0, 1002.0], [-1000.0, -999.0, -998.0]], requires_grad=True)
    probabilities = Softmax()(logits)
    assert np.all(np.isfinite(probabilities.data))
    np.testing.assert_allclose(probabilities.data.sum(axis=1), np.ones(2))
    (probabilities * Tensor([[1.0, 2.0, 3.0], [3.0, 1.0, 2.0]])).sum().backward()
    assert logits.grad is not None and np.all(np.isfinite(logits.grad))


def test_sequential_parameters_state_round_trip_and_freezing() -> None:
    model = Sequential(Linear(2, 4, seed=1), Tanh(), Linear(4, 1, seed=2))
    names = [name for name, _ in model.named_parameters()]
    assert names == [
        "layers.0.weight",
        "layers.0.bias",
        "layers.2.weight",
        "layers.2.bias",
    ]
    original = model.state_dict()
    model.layers[0].freeze()
    assert not model.layers[0].parameters()[0].requires_grad
    assert len(model.parameters(trainable_only=True)) == 2
    for parameter in model.parameters():
        parameter.data += 1.0
    model.load_state_dict(original)
    for name, parameter in model.named_parameters():
        np.testing.assert_allclose(parameter.data, original[name])


def test_mse_loss_value_and_gradient() -> None:
    predictions = Tensor([[1.0], [3.0]], requires_grad=True)
    loss = MSELoss()(predictions, [[2.0], [1.0]])
    assert loss.item() == pytest.approx(2.5)
    loss.backward()
    np.testing.assert_allclose(predictions.grad, [[-1.0], [2.0]])


def test_cross_entropy_is_stable_and_matches_known_value() -> None:
    logits = Tensor([[1000.0, 1001.0, 1002.0], [2.0, 1.0, -3.0]], requires_grad=True)
    targets = np.array([2, 0])
    loss = CrossEntropyLoss()(logits, targets)
    shifted = logits.data - logits.data.max(axis=1, keepdims=True)
    expected = -np.log(np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True))[
        np.arange(2), targets
    ].mean()
    assert loss.item() == pytest.approx(expected)
    loss.backward()
    assert logits.grad is not None and np.all(np.isfinite(logits.grad))
    np.testing.assert_allclose(logits.grad.sum(axis=1), np.zeros(2), atol=1e-12)


def test_cross_entropy_accepts_one_hot_targets() -> None:
    logits = Tensor([[0.5, -0.2, 0.1]], requires_grad=True)
    indexed = CrossEntropyLoss()(logits, np.array([0]))
    one_hot = CrossEntropyLoss()(logits, np.array([[1.0, 0.0, 0.0]]))
    assert indexed.item() == pytest.approx(one_hot.item())

