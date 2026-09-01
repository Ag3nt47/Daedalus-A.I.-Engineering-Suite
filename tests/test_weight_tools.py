import json
from collections.abc import Callable
from urllib.parse import urlparse

import numpy as np
import pytest

from daedalus.engine.weight_tools import (
    WEIGHT_TOOL_SPECS,
    WeightToolError,
    compile_truth_table,
    fit_extreme_learning_machine,
    fit_physics_constrained_polynomial,
    get_weight_tool_spec,
    recommend_weight_tool,
    select_uncertain_candidates,
    synthesize_low_rank_adapter,
    synthesize_recurrent_kernel,
)

EXPECTED_TOOL_KEYS = {
    "meta_weight",
    "logic_compiler",
    "recurrent_kernel",
    "constraint_optimizer",
    "matrix_inverter",
    "uncertainty_sampler",
}


def _diagnostics(result: object) -> dict[str, object]:
    return dict(result.record.diagnostics)  # type: ignore[attr-defined]


def test_tool_catalog_is_complete_and_links_to_primary_sources() -> None:
    assert {spec.key for spec in WEIGHT_TOOL_SPECS} == EXPECTED_TOOL_KEYS
    assert len(WEIGHT_TOOL_SPECS) == len(EXPECTED_TOOL_KEYS)

    for spec in WEIGHT_TOOL_SPECS:
        parsed = urlparse(spec.primary_source_url)
        assert parsed.scheme == "https"
        assert parsed.netloc
        assert spec.primary_source_title.strip()
        assert spec.youtube_query.strip()
        assert spec.formula.strip()
        assert spec.use_when.strip()
        assert spec.avoid_when.strip()
        assert get_weight_tool_spec(spec.key) is spec

    assert recommend_weight_tool("binary rules").key == "logic_compiler"
    assert recommend_weight_tool("expensive-labels").key == "uncertainty_sampler"
    with pytest.raises(WeightToolError, match="Unknown Weight Lab tool"):
        get_weight_tool_spec("not-a-tool")


def test_low_rank_synthesizer_is_seeded_bounded_and_read_only() -> None:
    context = np.array([1.0, 0.25, -0.5])
    first = synthesize_low_rank_adapter(
        context,
        input_dim=7,
        output_dim=5,
        rank=2,
        scale=3.0,
        max_delta_norm=0.2,
        seed=91,
    )
    second = synthesize_low_rank_adapter(
        context,
        input_dim=7,
        output_dim=5,
        rank=2,
        scale=3.0,
        max_delta_norm=0.2,
        seed=91,
    )
    changed = synthesize_low_rank_adapter(context, 7, 5, rank=2, seed=92)

    np.testing.assert_array_equal(first.weights, second.weights)
    np.testing.assert_allclose(
        first.delta, 3.0 * first.left_factor @ first.right_factor / np.sqrt(2)
    )
    assert not np.array_equal(first.weights, changed.weights)
    assert first.weights.shape == (5, 7)
    assert np.linalg.matrix_rank(first.delta, tol=1e-12) <= 2
    assert np.linalg.norm(first.delta) <= 0.2 + 1e-12
    assert not first.weights.flags.writeable
    assert first.record == second.record
    assert first.record.assurance == "initialized_only"
    assert any(hint.code == "hypernetwork.untrained" for hint in first.record.hints)
    json.dumps(first.record.to_dict(), allow_nan=False)


def test_logic_compiler_reproduces_shuffled_xor_exactly() -> None:
    rows = np.array([[1, 0], [0, 0], [1, 1], [0, 1]], dtype=float)
    targets = np.array([1, 0, 0, 1], dtype=float)
    compiled = compile_truth_table(rows, targets)
    canonical_rows = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)

    np.testing.assert_array_equal(compiled.predict(canonical_rows), [0, 1, 1, 0])
    assert compiled.predict(np.array([1, 0])) == pytest.approx(1.0)
    assert compiled.record.assurance == "exact_for_declared_contract"
    assert _diagnostics(compiled)["max_abs_error"] == 0.0
    assert _diagnostics(compiled)["exact_on_declared_domain"] is True
    assert not compiled.hidden_weights.flags.writeable


def test_recurrent_kernel_is_deterministic_and_strictly_contractive() -> None:
    stream = np.sin(np.linspace(0.0, 4.0, 128))
    first = synthesize_recurrent_kernel(
        stream, state_size=12, kernel_length=24, contraction=0.8, seed=19
    )
    second = synthesize_recurrent_kernel(
        stream, state_size=12, kernel_length=24, contraction=0.8, seed=19
    )

    np.testing.assert_array_equal(first.output, second.output)
    np.testing.assert_array_equal(first.reference_kernel, second.reference_kernel)
    assert first.transition_trace.shape == (stream.size, 12)
    assert first.reference_kernel.shape == (24,)
    assert np.max(np.abs(first.transition_trace)) < 0.8
    assert np.all((first.gate_trace > 0.0) & (first.gate_trace < 1.0))
    assert np.all(np.isfinite(first.state_trace))
    assert _diagnostics(first)["contractive_autonomous_transition"] is True


def test_constraint_fit_reports_each_loss_and_satisfies_polynomial_ode() -> None:
    coordinates = np.linspace(-1.0, 1.0, 9)
    observations = coordinates**2
    result = fit_physics_constrained_polynomial(
        coordinates,
        observations,
        degree=2,
        second_derivative=1.0,
        first_derivative=0.0,
        value_coefficient=0.0,
        source=2.0,
        physics_weight=10.0,
        boundary_coordinates=[-1.0, 1.0],
        boundary_values=[1.0, 1.0],
        boundary_weight=5.0,
        collocation_count=31,
        ridge=0.0,
    )
    diagnostics = _diagnostics(result)

    np.testing.assert_allclose(result.predict(coordinates), observations, atol=1e-12)
    np.testing.assert_allclose(result.residuals, 0.0, atol=1e-12)
    assert diagnostics["data_loss"] < 1e-24
    assert diagnostics["physics_loss"] < 1e-24
    assert diagnostics["boundary_loss"] < 1e-24
    assert diagnostics["total_loss"] == pytest.approx(
        diagnostics["data_loss"]
        + 10.0 * diagnostics["physics_loss"]
        + 5.0 * diagnostics["boundary_loss"]
    )
    assert result.record.algorithm.endswith("lstsq-v1")


def test_elm_uses_stable_least_squares_and_is_seeded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def explicit_inverse_is_forbidden(*_args: object, **_kwargs: object) -> np.ndarray:
        raise AssertionError("ELM must not form an explicit matrix inverse")

    monkeypatch.setattr(np.linalg, "inv", explicit_inverse_is_forbidden)
    monkeypatch.setattr(np.linalg, "pinv", explicit_inverse_is_forbidden)
    features = np.linspace(-1.0, 1.0, 24)[:, None]
    targets = 0.75 * features[:, 0] - 0.2
    first = fit_extreme_learning_machine(
        features, targets, hidden_units=24, activation="tanh", ridge=1e-9, seed=7
    )
    second = fit_extreme_learning_machine(
        features, targets, hidden_units=24, activation="tanh", ridge=1e-9, seed=7
    )

    np.testing.assert_array_equal(first.input_weights, second.input_weights)
    np.testing.assert_array_equal(first.output_weights, second.output_weights)
    np.testing.assert_allclose(first.predict(features), first.predictions[:, 0], atol=1e-14)
    assert _diagnostics(first)["training_rmse"] < 1e-4
    assert first.record.algorithm == "extreme-learning-machine-lstsq-v1"
    assert not first.output_weights.flags.writeable


def test_gp_excludes_observed_rows_and_breaks_uncertainty_ties_by_index() -> None:
    result = select_uncertain_candidates(
        [[0.0]],
        [4.0],
        [[-1.0], [1.0], [0.0]],
        query_count=2,
        length_scale=0.5,
        noise_std=0.0,
    )

    np.testing.assert_allclose(result.posterior_std[0], result.posterior_std[1], atol=0.0)
    np.testing.assert_array_equal(result.selected_indices, [0, 1])
    np.testing.assert_array_equal(result.selected_candidates, [[-1.0], [1.0]])
    assert result.acquisition_scores[2] == -np.inf
    assert _diagnostics(result)["excluded_observed_candidates"] == 1
    assert result.record.seed is None


def test_gp_batch_selection_conditions_away_redundant_neighbors() -> None:
    result = select_uncertain_candidates(
        [[0.0]],
        [0.0],
        [[-3.0], [-2.9], [3.0]],
        query_count=2,
        length_scale=1.0,
        noise_std=1e-4,
    )

    # After the stable first selection, greedy conditioning favors the distant
    # candidate instead of returning a redundant near-neighbor in the batch.
    assert result.selected_indices.tolist() == [0, 2]
    assert _diagnostics(result)["batch_strategy"] == "greedy-conditional-variance"


@pytest.mark.parametrize(
    ("operation", "message"),
    [
        (
            lambda: compile_truth_table([[0, 0], [0, 1]], [0, 1]),
            "complete",
        ),
        (
            lambda: compile_truth_table(
                [[0, 0], [0, 1], [1, 0], [1, 0]], [0, 1, 1, 0]
            ),
            "every binary combination",
        ),
        (
            lambda: compile_truth_table([[0, 0], [0, 1], [1, 0], [2, 1]], [0, 1, 1, 0]),
            "binary",
        ),
        (
            lambda: synthesize_recurrent_kernel([1.0], contraction=1.0),
            "strictly between",
        ),
        (
            lambda: synthesize_low_rank_adapter([np.nan], 2, 2, rank=1),
            "finite values",
        ),
        (
            lambda: fit_physics_constrained_polynomial(
                [0.0, 1.0],
                [0.0, 1.0],
                second_derivative=0.0,
                first_derivative=0.0,
                value_coefficient=0.0,
            ),
            "non-zero",
        ),
        (
            lambda: fit_extreme_learning_machine([[0.0], [1.0]], [0.0], hidden_units=2),
            "one row",
        ),
        (
            lambda: select_uncertain_candidates([[0.0]], [0.0], [[0.0]], query_count=1),
            "Not enough unobserved",
        ),
    ],
)
def test_algorithms_reject_invalid_contracts(
    operation: Callable[[], object], message: str
) -> None:
    with pytest.raises(WeightToolError, match=message):
        operation()


@pytest.mark.parametrize(
    ("operation", "message"),
    [
        (
            lambda: synthesize_low_rank_adapter(
                np.ones(512), 4_096, 4_096, rank=256, seed=47
            ),
            "controller-parameter safety bound",
        ),
        (
            lambda: compile_truth_table(np.zeros((1, 13)), [0.0]),
            "capped at 12 inputs",
        ),
        (
            lambda: synthesize_recurrent_kernel(np.ones(4_000), state_size=512),
            "state size exceeds",
        ),
        (
            lambda: fit_physics_constrained_polynomial(
                [0.0, 1.0], [0.0, 1.0], collocation_count=20_001
            ),
            "cannot exceed 20,000",
        ),
        (
            lambda: fit_extreme_learning_machine(
                np.zeros((1_000, 1)), np.zeros(1_000), hidden_units=8_192
            ),
            "ELM safety bound",
        ),
        (
            lambda: select_uncertain_candidates(
                np.zeros((1_025, 1)), np.zeros(1_025), np.ones((1, 1))
            ),
            "cannot exceed 1,024",
        ),
    ],
)
def test_algorithms_stop_at_resource_bounds_before_large_allocations(
    operation: Callable[[], object], message: str
) -> None:
    with pytest.raises(WeightToolError, match=message):
        operation()


@pytest.mark.parametrize(
    "operation",
    [
        lambda: synthesize_low_rank_adapter([1.0], 1, 1, rank=1, seed="not-a-seed"),
        lambda: synthesize_recurrent_kernel([1.0], seed="not-a-seed"),
        lambda: fit_extreme_learning_machine([[1.0]], [1.0], seed="not-a-seed"),
    ],
)
def test_seeded_tools_report_bad_seed_as_weight_tool_error(
    operation: Callable[[], object],
) -> None:
    with pytest.raises(WeightToolError, match="seed"):
        operation()
