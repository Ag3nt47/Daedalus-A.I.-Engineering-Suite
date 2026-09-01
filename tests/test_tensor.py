from __future__ import annotations

import numpy as np
import pytest

from daedalus.core import Tensor, no_grad
from daedalus.engine import gradient_check


def test_broadcast_arithmetic_reduces_gradients_to_operand_shapes() -> None:
    values = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
    scale = Tensor([2.0, -1.0, 0.5], requires_grad=True)

    loss = ((values * scale) + scale - values / 2.0).sum()
    loss.backward()

    np.testing.assert_allclose(values.grad, np.broadcast_to(scale.data - 0.5, values.shape))
    np.testing.assert_allclose(scale.grad, values.data.sum(axis=0) + 2.0)


def test_matmul_vector_and_matrix_gradients() -> None:
    left = Tensor([[1.0, 2.0], [-1.0, 3.0]], requires_grad=True)
    right = Tensor([4.0, -2.0], requires_grad=True)

    (left @ right).sum().backward()

    np.testing.assert_allclose(left.grad, [[4.0, -2.0], [4.0, -2.0]])
    np.testing.assert_allclose(right.grad, left.data.sum(axis=0))


def test_vector_dot_product_gradients() -> None:
    left = Tensor([1.0, -2.0, 3.0], requires_grad=True)
    right = Tensor([4.0, 5.0, -1.0], requires_grad=True)
    output = left @ right
    assert output.shape == ()
    output.backward()
    np.testing.assert_allclose(left.grad, right.data)
    np.testing.assert_allclose(right.grad, left.data)


def test_batched_matmul_broadcast_gradient() -> None:
    left = Tensor(np.arange(12.0).reshape(2, 2, 3), requires_grad=True)
    right = Tensor(np.arange(6.0).reshape(3, 2), requires_grad=True)
    output = (left @ right).mean()
    output.backward()

    check = gradient_check(lambda a, b: (a @ b).mean(), [left.data, right.data])
    assert check.passed, check.failures
    assert left.grad is not None and right.grad is not None


def test_power_exp_log_and_activation_chain_passes_gradient_check() -> None:
    values = np.array([[0.3, 0.8], [1.2, 1.7]])
    check = gradient_check(
        lambda tensor: (
            ((tensor**2).exp().log() + tensor.tanh() + tensor.sigmoid() + tensor.relu()).mean()
        ),
        [values],
    )
    assert check.passed, check.failures
    assert check.max_absolute_error < 1e-6


def test_division_and_tensor_power_pass_gradient_check_with_broadcasting() -> None:
    base = np.array([[0.4, 0.8, 1.2], [1.6, 2.0, 2.4]])
    exponent = np.array([0.5, 1.5, 2.0])
    check = gradient_check(
        lambda left, right: ((left / right) + (left**right)).mean(),
        [base, exponent],
    )
    assert check.passed, check.failures


def test_sum_mean_reshape_and_transpose_gradients() -> None:
    values = Tensor(np.arange(6.0).reshape(2, 3), requires_grad=True)
    output = values.reshape(3, 2).T.mean(axis=0).sum()
    output.backward()
    np.testing.assert_allclose(values.grad, np.full((2, 3), 1.0 / 2.0))


def test_slice_scatter_adds_repeated_index_gradients() -> None:
    values = Tensor([2.0, 3.0, 4.0], requires_grad=True)
    values[[0, 0, 2]].sum().backward()
    np.testing.assert_allclose(values.grad, [2.0, 0.0, 1.0])


def test_leaf_gradients_accumulate_across_backward_calls() -> None:
    value = Tensor(3.0, requires_grad=True)
    (value * 2.0).backward()
    (value**2).backward()
    np.testing.assert_allclose(value.grad, 8.0)
    value.zero_grad(set_to_none=False)
    np.testing.assert_allclose(value.grad, 0.0)


def test_non_scalar_backward_requires_explicit_seed() -> None:
    value = Tensor([1.0, 2.0], requires_grad=True)
    with pytest.raises(ValueError, match="gradient is required"):
        value.relu().backward()
    value.relu().backward([2.0, 3.0])
    np.testing.assert_allclose(value.grad, [2.0, 3.0])


def test_no_grad_disables_graph_construction() -> None:
    value = Tensor([1.0, 2.0], requires_grad=True)
    with no_grad():
        result = (value * 4.0).sum()
    assert not result.requires_grad
    with pytest.raises(RuntimeError, match="does not require"):
        result.backward()


def test_sigmoid_is_finite_for_extreme_values() -> None:
    values = Tensor([-1000.0, 0.0, 1000.0], requires_grad=True)
    result = values.sigmoid()
    np.testing.assert_allclose(result.data, [0.0, 0.5, 1.0], atol=1e-15)
    result.sum().backward()
    assert np.all(np.isfinite(values.grad))
