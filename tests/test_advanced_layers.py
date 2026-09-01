from __future__ import annotations

import numpy as np
import pytest

from daedalus.core import Tensor
from daedalus.layers import (
    Dropout,
    Embedding,
    Flatten,
    LayerInputError,
    LayerNorm,
    MultiHeadSelfAttention,
    TransformerBlock,
)


def test_flatten_preserves_batch_and_backward_mapping() -> None:
    values = Tensor(np.arange(24.0).reshape(2, 3, 4), requires_grad=True)
    output = Flatten(start_dim=1)(values)
    assert output.shape == (2, 12)
    output.sum().backward()
    np.testing.assert_array_equal(values.grad, np.ones_like(values.data))


def test_flatten_reports_invalid_dimension_range() -> None:
    with pytest.raises(LayerInputError, match="invalid"):
        Flatten()(Tensor([1.0, 2.0]))


def test_dropout_is_seed_deterministic_and_eval_is_identity() -> None:
    left = Dropout(0.4, seed=17)
    right = Dropout(0.4, seed=17)
    values = Tensor(np.ones(100), requires_grad=True)
    left_output = left(values)
    right_output = right(Tensor(np.ones(100)))
    np.testing.assert_array_equal(left_output.data, right_output.data)
    unique = np.unique(left_output.data)
    assert len(unique) == 2
    assert unique[0] == 0.0
    assert unique[1] == pytest.approx(1.0 / 0.6)

    left_output.sum().backward()
    np.testing.assert_array_equal(values.grad, left_output.data)
    left.reset_seed()
    np.testing.assert_array_equal(left(values).data, left_output.data)

    left.eval()
    evaluation = left(values)
    assert evaluation is values
    np.testing.assert_array_equal(evaluation.data, values.data)


def test_layer_norm_normalizes_last_dimension_and_backpropagates() -> None:
    generator = np.random.default_rng(8)
    values = Tensor(generator.normal(size=(2, 3, 4)), requires_grad=True)
    layer = LayerNorm(4)
    output = layer(values)
    np.testing.assert_allclose(output.data.mean(axis=-1), 0.0, atol=1e-12)
    np.testing.assert_allclose(output.data.var(axis=-1), 1.0, atol=8e-5)

    weights = Tensor(generator.normal(size=output.shape))
    (output * weights).sum().backward()
    assert values.grad is not None and np.all(np.isfinite(values.grad))
    assert layer.scale is not None and layer.scale.grad is not None
    assert layer.shift is not None and layer.shift.grad is not None


def test_layer_norm_rejects_wrong_trailing_shape() -> None:
    with pytest.raises(LayerInputError, match="trailing dimensions"):
        LayerNorm((2, 3))(Tensor(np.zeros((4, 3, 2))))


def test_embedding_lookup_padding_and_repeated_token_gradients() -> None:
    first = Embedding(6, 3, padding_index=0, seed=4)
    second = Embedding(6, 3, padding_index=0, seed=4)
    np.testing.assert_array_equal(first.weight.data, second.weight.data)
    output = first(np.array([[0, 1, 1, 2]]))
    assert output.shape == (1, 4, 3)
    np.testing.assert_array_equal(output.data[0, 0], np.zeros(3))
    output.sum().backward()
    np.testing.assert_array_equal(first.weight.grad[0], np.zeros(3))
    np.testing.assert_array_equal(first.weight.grad[1], np.full(3, 2.0))
    np.testing.assert_array_equal(first.weight.grad[2], np.ones(3))


@pytest.mark.parametrize("bad_tokens", [[0.5, 1.0], [-1, 2], [0, 6]])
def test_embedding_rejects_invalid_token_ids(bad_tokens: list[float]) -> None:
    with pytest.raises(LayerInputError):
        Embedding(6, 3)(np.asarray(bad_tokens))


def test_causal_attention_normalizes_rows_masks_future_and_backpropagates() -> None:
    generator = np.random.default_rng(12)
    data = generator.normal(size=(2, 4, 8))
    first = MultiHeadSelfAttention(8, 2, causal=True, seed=13)
    second = MultiHeadSelfAttention(8, 2, causal=True, seed=13)
    inputs = Tensor(data, requires_grad=True)
    output = first(inputs)
    np.testing.assert_allclose(output.data, second(Tensor(data)).data)
    assert output.shape == data.shape

    weights = first.last_attention_weights
    assert weights is not None and weights.shape == (2, 2, 4, 4)
    np.testing.assert_allclose(weights.sum(axis=-1), np.ones((2, 2, 4)))
    future = np.triu(np.ones((4, 4)), k=1).reshape(1, 1, 4, 4)
    assert np.max(weights * future) == 0.0

    (output**2).mean().backward()
    assert inputs.grad is not None and np.all(np.isfinite(inputs.grad))
    assert all(parameter.grad is not None for parameter in first.parameters())
    assert all(np.all(np.isfinite(parameter.grad)) for parameter in first.parameters())


def test_attention_causal_mask_can_be_overridden_per_call() -> None:
    layer = MultiHeadSelfAttention(4, 2, causal=True, seed=2)
    layer(np.ones((1, 3, 4)), causal=False)
    weights = layer.last_attention_weights
    assert weights is not None
    assert np.any(weights[:, :, 0, 1:] > 0)


def test_transformer_block_is_deterministic_shape_safe_and_differentiable() -> None:
    data = np.random.default_rng(20).normal(size=(2, 5, 8))
    first = TransformerBlock(8, 2, feed_forward_dim=16, dropout=0.25, seed=21)
    second = TransformerBlock(8, 2, feed_forward_dim=16, dropout=0.25, seed=21)
    inputs = Tensor(data, requires_grad=True)
    output = first(inputs)
    np.testing.assert_allclose(output.data, second(Tensor(data)).data)
    assert output.shape == data.shape

    output.mean().backward()
    assert inputs.grad is not None
    assert all(parameter.grad is not None for parameter in first.parameters())

    first.eval()
    evaluation_one = first(Tensor(data)).data
    evaluation_two = first(Tensor(data)).data
    np.testing.assert_allclose(evaluation_one, evaluation_two)
