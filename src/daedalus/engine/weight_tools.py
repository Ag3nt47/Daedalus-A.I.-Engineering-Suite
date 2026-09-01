"""Bounded, deterministic prototypes for dynamic neural-weight workflows.

The six tools in this module are intentionally explicit about their contracts.
They are useful numerical building blocks and teaching instruments, not claims
that task knowledge can be created without data, priors, or validation.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

import numpy as np

Assurance = Literal[
    "exact_for_declared_contract",
    "numerical_fit",
    "approximation",
    "initialized_only",
]
HintSeverity = Literal["success", "info", "warning"]

MAX_MATRIX_ELEMENTS = 2_000_000
MAX_CONTROLLER_PARAMETERS = 4_000_000
MAX_LOGIC_INPUTS = 12
MAX_SEQUENCE_STATE_ELEMENTS = 2_000_000
MAX_ELM_FEATURE_ELEMENTS = 5_000_000
MAX_GP_TRAINING_ROWS = 1_024
MAX_GP_CROSS_ELEMENTS = 4_000_000


class WeightToolError(ValueError):
    """Raised when a Weight Lab request is unsafe or mathematically invalid."""


@dataclass(frozen=True, slots=True)
class ToolHint:
    severity: HintSeverity
    code: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
        }


@dataclass(frozen=True, slots=True)
class ArrayDescriptor:
    name: str
    role: str
    shape: tuple[int, ...]
    dtype: str
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "role": self.role,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class WeightToolRecord:
    """JSON-safe deterministic evidence for one computation.

    Raw arrays remain on the concrete result object.  The record contains only
    descriptors and hashes, so it can be displayed or persisted safely.
    """

    schema_version: int
    tool_key: str
    algorithm: str
    assurance: Assurance
    seed: int | None
    input_sha256: str
    config: tuple[tuple[str, Any], ...]
    arrays: tuple[ArrayDescriptor, ...]
    diagnostics: tuple[tuple[str, Any], ...]
    hints: tuple[ToolHint, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "tool_key": self.tool_key,
            "algorithm": self.algorithm,
            "assurance": self.assurance,
            "seed": self.seed,
            "input_sha256": self.input_sha256,
            "config": dict(self.config),
            "arrays": [item.to_dict() for item in self.arrays],
            "diagnostics": dict(self.diagnostics),
            "hints": [item.to_dict() for item in self.hints],
        }


@dataclass(frozen=True, slots=True)
class WeightToolSpec:
    key: str
    title: str
    concept: str
    maturity: str
    assurance: Assurance
    formula: str
    use_when: str
    avoid_when: str
    output: str
    youtube_query: str
    primary_source_title: str
    primary_source_url: str


WEIGHT_TOOL_SPECS: tuple[WeightToolSpec, ...] = (
    WeightToolSpec(
        "meta_weight",
        "Meta-Weight Synthesizer",
        "A compact hypernetwork maps a numeric context vector to low-rank target weights.",
        "Untrained scaffold",
        "initialized_only",
        "z=tanh(c W_h+b_h);  delta_W=(scale/sqrt(r)) A(z) B(z)",
        "You have a trained controller objective and need small context-conditioned adapters.",
        "You expect an untrained generator to contain task knowledge or replace all checkpoints.",
        "A bounded low-rank adapter, factors, hashes, and norm diagnostics.",
        "hypernetworks dynamic weight generation neural networks tutorial",
        "HyperNetworks (Ha, Dai, and Le)",
        "https://arxiv.org/abs/1609.09106",
    ),
    WeightToolSpec(
        "logic_compiler",
        "Direct Logic Compiler",
        "A complete binary truth table is compiled into an exact threshold lookup network.",
        "Exact bounded compiler",
        "exact_for_declared_contract",
        "h_s(x)=1[(2s-1)^T x + 0.5-||s||_1 > 0];  y=sum_s h_s(x)y_s",
        "Your rule domain is a small, complete, explicit binary grid.",
        "You need arbitrary program-to-network compilation or a large exponential rule domain.",
        "Hidden threshold weights, biases, output weights, and exactness evidence.",
        "compile boolean truth table into neural network weights",
        "Learning to Compile Programs to Neural Networks (Weber et al.)",
        "https://proceedings.mlr.press/v235/weber24b.html",
    ),
    WeightToolSpec(
        "recurrent_kernel",
        "Recurrent Kernel Engine",
        "A seeded controller produces input-selective, contractive diagonal state transitions.",
        "Selective SSM prototype",
        "approximation",
        "h_t=a_t*h_(t-1)+g_t*(B x_t),  |a_t|<rho;  y_t=C h_t+D x_t",
        "You need an inspectable local recurrence for a scalar stream.",
        "You need Mamba-compatible training kernels, hardware fusion, or benchmark claims.",
        "Transition, gate, state, output, and reference-kernel traces.",
        "selective state space models Mamba SSM tutorial",
        "Mamba: Linear-Time Sequence Modeling with Selective State Spaces",
        "https://arxiv.org/abs/2312.00752",
    ),
    WeightToolSpec(
        "constraint_optimizer",
        "Constraint Optimizer",
        "A polynomial basis is fitted to observations and a typed differential-operator residual.",
        "Physics-informed finite-basis prototype",
        "numerical_fit",
        "min_w mean||Phi w-y||^2 + lambda mean||L(Phi)w-f||^2 + boundary loss",
        "You know a 1D linear ODE/PDE residual and have sparse numeric observations.",
        "You need a full neural PINN with higher-order automatic differentiation.",
        "Polynomial weights, predictions, residuals, and loss decomposition.",
        "physics informed neural networks PINN differential equations tutorial",
        "Physics-informed neural networks (Raissi, Perdikaris, Karniadakis)",
        "https://doi.org/10.1016/j.jcp.2018.10.045",
    ),
    WeightToolSpec(
        "matrix_inverter",
        "Matrix Inverter",
        "An Extreme Learning Machine freezes random hidden features and solves output weights.",
        "Closed-form random-feature fit",
        "numerical_fit",
        "H=g(X W_in+b);  beta=argmin ||H beta-Y||^2 + lambda||beta||^2",
        "You need a fast, deterministic regression baseline on a bounded numeric dataset.",
        "You have no labels, need online deep features, or cannot tolerate random-feature variance.",
        "Hidden parameters, solved output weights, predictions, rank, and conditioning.",
        "extreme learning machine pseudoinverse random hidden layer tutorial",
        "Extreme learning machine: Theory and applications",
        "https://doi.org/10.1016/j.neucom.2005.12.126",
    ),
    WeightToolSpec(
        "uncertainty_sampler",
        "Uncertainty Sampler",
        "A small RBF Gaussian process ranks an explicit candidate pool by posterior uncertainty.",
        "Bounded GP active-learning prototype",
        "approximation",
        "mu*=K_*^T K^-1 y;  var*=k(x*,x*)-K_*^T K^-1 K_*",
        "Labels are expensive and you have a small numeric candidate pool.",
        "You need calibrated safety guarantees or uncertainty over the entire parameter space.",
        "Posterior mean/std, acquisition scores, and selected candidate indices.",
        "Gaussian process active learning maximum variance sampling tutorial",
        "Gaussian Processes for Machine Learning",
        "https://gaussianprocess.org/gpml/chapters/",
    ),
)

_SPECS_BY_KEY = {item.key: item for item in WEIGHT_TOOL_SPECS}


def get_weight_tool_spec(key: str) -> WeightToolSpec:
    try:
        return _SPECS_BY_KEY[str(key)]
    except KeyError as exc:
        raise WeightToolError(f"Unknown Weight Lab tool: {key!r}") from exc


def recommend_weight_tool(goal: str) -> WeightToolSpec:
    """Map a stable design-goal key or a short natural phrase to a tool."""

    normalized = str(goal).strip().casefold().replace("-", "_").replace(" ", "_")
    aliases = {
        "context_adapter": "meta_weight",
        "generate_weights_from_context": "meta_weight",
        "binary_rules": "logic_compiler",
        "deterministic_rules": "logic_compiler",
        "streaming_sequence": "recurrent_kernel",
        "known_equation": "constraint_optimizer",
        "physics_constraint": "constraint_optimizer",
        "instant_numeric_fit": "matrix_inverter",
        "closed_form_fit": "matrix_inverter",
        "choose_next_sample": "uncertainty_sampler",
        "expensive_labels": "uncertainty_sampler",
    }
    key = aliases.get(normalized, normalized)
    return get_weight_tool_spec(key)


def _numeric_array(value: Any, name: str, *, ndim: int) -> np.ndarray:
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise WeightToolError(f"{name} must be a rectangular numeric array") from exc
    if raw.dtype.kind not in "biuf":
        raise WeightToolError(f"{name} must contain real numeric values only")
    try:
        array = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise WeightToolError(f"{name} must contain real numeric values only") from exc
    if array.ndim != ndim:
        raise WeightToolError(f"{name} must be {ndim}-dimensional")
    if not array.size:
        raise WeightToolError(f"{name} cannot be empty")
    if not np.all(np.isfinite(array)):
        raise WeightToolError(f"{name} must contain finite values")
    return array


def _vector(value: Any, name: str) -> np.ndarray:
    return _numeric_array(value, name, ndim=1)


def _matrix(value: Any, name: str) -> np.ndarray:
    array = _numeric_array(value, name, ndim=2)
    if array.size > MAX_MATRIX_ELEMENTS:
        raise WeightToolError(
            f"{name} exceeds the {MAX_MATRIX_ELEMENTS:,}-element Weight Lab safety bound"
        )
    return array


def _positive_int(name: str, value: Any, *, maximum: int) -> int:
    if isinstance(value, bool):
        raise WeightToolError(f"{name} must be a positive integer")
    try:
        integer = int(value)
    except (TypeError, ValueError) as exc:
        raise WeightToolError(f"{name} must be a positive integer") from exc
    if integer != value or integer <= 0:
        raise WeightToolError(f"{name} must be a positive integer")
    if integer > maximum:
        raise WeightToolError(f"{name} cannot exceed {maximum:,}")
    return integer


def _seed(value: Any) -> int:
    """Validate a NumPy-compatible non-negative 32-bit seed without pre-arithmetic."""

    if isinstance(value, bool):
        raise WeightToolError("seed must be a non-negative integer")
    try:
        integer = int(value)
    except (TypeError, ValueError) as exc:
        raise WeightToolError("seed must be a non-negative integer") from exc
    if integer != value or integer < 0 or integer > 2_147_483_647:
        raise WeightToolError("seed must be a non-negative integer no greater than 2147483647")
    return integer


def _finite_float(name: str, value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise WeightToolError(f"{name} must be a finite number") from exc
    if not math.isfinite(number):
        raise WeightToolError(f"{name} must be a finite number")
    return number


def _nonnegative_float(name: str, value: Any) -> float:
    number = _finite_float(name, value)
    if number < 0:
        raise WeightToolError(f"{name} cannot be negative")
    return number


def _readonly(value: Any, *, dtype: Any | None = None) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True, order="C")
    array.setflags(write=False)
    return array


def _canonical_array(array: np.ndarray) -> np.ndarray:
    value = np.ascontiguousarray(array)
    dtype = value.dtype
    if dtype.byteorder == ">" or (dtype.byteorder == "=" and not np.little_endian):
        value = value.byteswap().view(dtype.newbyteorder("<"))
    elif dtype.byteorder == "=":
        value = value.view(dtype.newbyteorder("<"))
    return np.ascontiguousarray(value)


def _array_descriptor(name: str, role: str, array: np.ndarray) -> ArrayDescriptor:
    canonical = _canonical_array(np.asarray(array))
    digest = hashlib.sha256(canonical.tobytes(order="C")).hexdigest()
    return ArrayDescriptor(name, role, tuple(canonical.shape), canonical.dtype.str, digest)


def _input_digest(arrays: Mapping[str, np.ndarray], config: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    digest.update(
        json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    )
    for name in sorted(arrays):
        value = _canonical_array(np.asarray(arrays[name]))
        digest.update(name.encode("utf-8"))
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _record(
    *,
    tool_key: str,
    algorithm: str,
    assurance: Assurance,
    seed: int | None,
    inputs: Mapping[str, np.ndarray],
    outputs: Mapping[str, tuple[str, np.ndarray]],
    config: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
    hints: Sequence[ToolHint],
) -> WeightToolRecord:
    return WeightToolRecord(
        1,
        tool_key,
        algorithm,
        assurance,
        seed,
        _input_digest(inputs, config),
        tuple(sorted(config.items())),
        tuple(
            _array_descriptor(name, role, array)
            for name, (role, array) in sorted(outputs.items())
        ),
        tuple(sorted(diagnostics.items())),
        tuple(hints),
    )


@dataclass(frozen=True, slots=True)
class HypernetworkResult:
    weights: np.ndarray
    delta: np.ndarray
    left_factor: np.ndarray
    right_factor: np.ndarray
    latent: np.ndarray
    record: WeightToolRecord

    def project(self, inputs: Any) -> np.ndarray:
        values = np.asarray(inputs, dtype=np.float64)
        if values.shape[-1:] != (self.weights.shape[1],):
            raise WeightToolError(
                f"inputs must end with {self.weights.shape[1]} features for this adapter"
            )
        return values @ self.weights.T


def synthesize_low_rank_adapter(
    context: Any,
    input_dim: int,
    output_dim: int,
    *,
    rank: int = 4,
    hidden_dim: int | None = None,
    scale: float = 1.0,
    max_delta_norm: float = 1.0,
    seed: int = 47,
    base_weights: Any | None = None,
) -> HypernetworkResult:
    """Generate a bounded low-rank adapter from numeric context.

    The controller parameters are seeded initial values.  Until they are trained
    against a task loss or target-weight corpus, the result carries
    ``initialized_only`` assurance.
    """

    values = _vector(context, "context")
    if values.size > 512:
        raise WeightToolError("context cannot exceed 512 numeric values")
    inputs = _positive_int("input_dim", input_dim, maximum=4_096)
    outputs = _positive_int("output_dim", output_dim, maximum=4_096)
    adapter_rank = _positive_int("rank", rank, maximum=min(inputs, outputs, 256))
    hidden = (
        min(64, max(4, 2 * adapter_rank))
        if hidden_dim is None
        else _positive_int("hidden_dim", hidden_dim, maximum=512)
    )
    adapter_scale = _nonnegative_float("scale", scale)
    norm_limit = _finite_float("max_delta_norm", max_delta_norm)
    if norm_limit <= 0:
        raise WeightToolError("max_delta_norm must be positive")
    seed_value = _seed(seed)
    generated = adapter_rank * (outputs + inputs)
    parameter_count = values.size * hidden + hidden + hidden * generated + generated
    if parameter_count > MAX_CONTROLLER_PARAMETERS:
        raise WeightToolError(
            "The requested hypernetwork exceeds the controller-parameter safety bound; "
            "reduce target dimensions, rank, or hidden width."
        )
    if inputs * outputs > MAX_MATRIX_ELEMENTS:
        raise WeightToolError("The requested target weight matrix is too large for Weight Lab")

    if base_weights is None:
        base = np.zeros((outputs, inputs), dtype=np.float64)
    else:
        base = _matrix(base_weights, "base_weights")
        if base.shape != (outputs, inputs):
            raise WeightToolError(
                f"base_weights must have shape {(outputs, inputs)}, got {base.shape}"
            )

    generator = np.random.default_rng(seed_value)
    normalized = values / max(float(np.linalg.norm(values)), 1.0)
    hidden_weights = generator.normal(0.0, 1.0 / math.sqrt(values.size), (values.size, hidden))
    hidden_bias = generator.normal(0.0, 0.05, hidden)
    output_weights = generator.normal(0.0, 1.0 / math.sqrt(hidden), (hidden, generated))
    output_bias = generator.normal(0.0, 0.02, generated)
    latent = np.tanh(normalized @ hidden_weights + hidden_bias)
    generated_values = latent @ output_weights + output_bias
    split = outputs * adapter_rank
    left = generated_values[:split].reshape(outputs, adapter_rank)
    right = generated_values[split:].reshape(adapter_rank, inputs)
    delta = adapter_scale * (left @ right) / math.sqrt(adapter_rank)
    raw_norm = float(np.linalg.norm(delta))
    clip_factor = min(1.0, norm_limit / max(raw_norm, np.finfo(np.float64).tiny))
    left = left * clip_factor
    delta = delta * clip_factor
    weights = base + delta

    weights_ro = _readonly(weights, dtype=np.float64)
    delta_ro = _readonly(delta, dtype=np.float64)
    left_ro = _readonly(left, dtype=np.float64)
    right_ro = _readonly(right, dtype=np.float64)
    latent_ro = _readonly(latent, dtype=np.float64)
    record = _record(
        tool_key="meta_weight",
        algorithm="seeded-low-rank-hypernetwork-v1",
        assurance="initialized_only",
        seed=seed_value,
        inputs={"context": values, "base_weights": base},
        outputs={
            "weights": ("generated target weight block", weights_ro),
            "delta": ("low-rank adapter delta", delta_ro),
            "left_factor": ("left low-rank factor", left_ro),
            "right_factor": ("right low-rank factor", right_ro),
            "latent": ("context latent", latent_ro),
        },
        config={
            "input_dim": inputs,
            "output_dim": outputs,
            "rank": adapter_rank,
            "hidden_dim": hidden,
            "scale": adapter_scale,
            "max_delta_norm": norm_limit,
        },
        diagnostics={
            "controller_parameter_count": int(parameter_count),
            "delta_frobenius_norm": float(np.linalg.norm(delta)),
            "raw_delta_frobenius_norm": raw_norm,
            "rank_upper_bound": adapter_rank,
            "was_norm_clipped": bool(clip_factor < 1.0),
        },
        hints=(
            ToolHint(
                "warning",
                "hypernetwork.untrained",
                "These seeded controller weights demonstrate the data flow; train and evaluate "
                "the controller before treating a generated adapter as functional.",
            ),
            ToolHint(
                "info",
                "hypernetwork.low_rank",
                "Validate the adapter on held-out task evidence and compare it with a zero-delta baseline.",
            ),
        ),
    )
    return HypernetworkResult(weights_ro, delta_ro, left_ro, right_ro, latent_ro, record)


@dataclass(frozen=True, slots=True)
class TruthTableCompilation:
    hidden_weights: np.ndarray
    hidden_bias: np.ndarray
    output_weights: np.ndarray
    input_rows: np.ndarray
    targets: np.ndarray
    target_was_1d: bool
    record: WeightToolRecord

    def predict(self, inputs: Any) -> np.ndarray:
        raw = np.asarray(inputs)
        single = raw.ndim == 1
        values = raw.reshape(1, -1) if single else raw
        values = _matrix(values, "inputs")
        if values.shape[1] != self.hidden_weights.shape[0]:
            raise WeightToolError(
                f"inputs must have {self.hidden_weights.shape[0]} binary features"
            )
        if not np.all((values == 0.0) | (values == 1.0)):
            raise WeightToolError("inputs must contain binary values 0 or 1")
        hidden = (values @ self.hidden_weights + self.hidden_bias > 0.0).astype(np.float64)
        predictions = hidden @ self.output_weights
        if self.target_was_1d:
            predictions = predictions[:, 0]
        return predictions[0] if single else predictions


def compile_truth_table(
    input_rows: Any,
    targets: Any,
    *,
    tolerance: float = 1e-12,
) -> TruthTableCompilation:
    """Compile a complete binary truth table into an exact threshold network."""

    raw_inputs = _matrix(input_rows, "input_rows")
    if not np.all((raw_inputs == 0.0) | (raw_inputs == 1.0)):
        raise WeightToolError("input_rows must contain binary values 0 or 1")
    row_count, input_count = raw_inputs.shape
    if input_count > MAX_LOGIC_INPUTS:
        raise WeightToolError(
            f"truth tables are capped at {MAX_LOGIC_INPUTS} inputs because their size is exponential"
        )
    expected_rows = 1 << input_count
    if row_count != expected_rows:
        raise WeightToolError(
            f"A complete {input_count}-input truth table requires {expected_rows} rows"
        )
    raw_targets = np.asarray(targets)
    target_was_1d = raw_targets.ndim == 1
    if target_was_1d:
        target_matrix = _vector(raw_targets, "targets")[:, None]
    else:
        target_matrix = _matrix(raw_targets, "targets")
    if target_matrix.shape[0] != row_count:
        raise WeightToolError("targets must contain one row for every truth-table input row")
    error_tolerance = _nonnegative_float("tolerance", tolerance)

    powers = (1 << np.arange(input_count - 1, -1, -1, dtype=np.int64)).astype(np.float64)
    codes = np.asarray(raw_inputs @ powers, dtype=np.int64)
    if np.unique(codes).size != row_count or set(codes.tolist()) != set(range(expected_rows)):
        raise WeightToolError("input_rows must contain every binary combination exactly once")
    order = np.argsort(codes, kind="stable")
    canonical_inputs = raw_inputs[order]
    canonical_targets = target_matrix[order]

    hidden_weights = (2.0 * canonical_inputs - 1.0).T
    hidden_bias = 0.5 - np.sum(canonical_inputs, axis=1)
    output_weights = canonical_targets.copy()
    hidden = (
        canonical_inputs @ hidden_weights + hidden_bias > 0.0
    ).astype(np.float64)
    predictions = hidden @ output_weights
    max_error = float(np.max(np.abs(predictions - canonical_targets)))
    exact = max_error <= error_tolerance
    if not exact:
        raise WeightToolError("The compiled threshold network failed its exactness self-check")

    hidden_weights_ro = _readonly(hidden_weights, dtype=np.float64)
    hidden_bias_ro = _readonly(hidden_bias, dtype=np.float64)
    output_weights_ro = _readonly(output_weights, dtype=np.float64)
    inputs_ro = _readonly(canonical_inputs, dtype=np.float64)
    targets_ro = _readonly(canonical_targets, dtype=np.float64)
    record = _record(
        tool_key="logic_compiler",
        algorithm="complete-binary-threshold-compiler-v1",
        assurance="exact_for_declared_contract",
        seed=None,
        inputs={"input_rows": inputs_ro, "targets": targets_ro},
        outputs={
            "hidden_weights": ("binary pattern detector weights", hidden_weights_ro),
            "hidden_bias": ("binary pattern detector bias", hidden_bias_ro),
            "output_weights": ("truth-table output weights", output_weights_ro),
        },
        config={"input_count": input_count, "tolerance": error_tolerance},
        diagnostics={
            "compiled_rows": row_count,
            "hidden_units": row_count,
            "max_abs_error": max_error,
            "exact_on_declared_domain": exact,
        },
        hints=(
            ToolHint(
                "success",
                "logic.exact",
                "The compiled network reproduces every row in the complete declared binary domain.",
            ),
            ToolHint(
                "warning",
                "logic.contract",
                "Exactness does not extend to non-binary inputs or arbitrary program semantics.",
            ),
        ),
    )
    return TruthTableCompilation(
        hidden_weights_ro,
        hidden_bias_ro,
        output_weights_ro,
        inputs_ro,
        targets_ro,
        target_was_1d,
        record,
    )


@dataclass(frozen=True, slots=True)
class RecurrentKernelResult:
    output: np.ndarray
    reference_kernel: np.ndarray
    transition_trace: np.ndarray
    gate_trace: np.ndarray
    state_trace: np.ndarray
    context_summary: np.ndarray
    record: WeightToolRecord


def _sigmoid(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(value, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def synthesize_recurrent_kernel(
    stream: Any,
    *,
    state_size: int = 8,
    kernel_length: int = 16,
    contraction: float = 0.95,
    seed: int = 47,
) -> RecurrentKernelResult:
    """Run a stable, input-selective diagonal state-space prototype."""

    values = _vector(stream, "stream")
    states = _positive_int("state_size", state_size, maximum=512)
    length = _positive_int("kernel_length", kernel_length, maximum=100_000)
    rho = _finite_float("contraction", contraction)
    if not 0.0 < rho < 1.0:
        raise WeightToolError("contraction must be strictly between 0 and 1")
    seed_value = _seed(seed)
    if values.size * states > MAX_SEQUENCE_STATE_ELEMENTS:
        raise WeightToolError("stream length times state size exceeds the Weight Lab safety bound")
    if length * states > MAX_SEQUENCE_STATE_ELEMENTS:
        raise WeightToolError("kernel length times state size exceeds the Weight Lab safety bound")

    rms = max(float(np.sqrt(np.mean(values * values))), np.finfo(np.float64).eps)
    scale = max(rms, float(np.std(values)), 1.0)
    trend = 0.0 if values.size == 1 else float(values[-1] - values[0]) / (values.size - 1)
    summary = np.array(
        [
            float(np.mean(values)) / scale,
            float(np.std(values)) / scale,
            math.tanh(math.log1p(rms)),
            trend / scale,
            float(values[-1]) / scale,
        ],
        dtype=np.float64,
    )
    generator = np.random.default_rng(seed_value)
    controller_width = summary.size + 1
    transition_controller = generator.normal(
        0.0, 1.0 / math.sqrt(controller_width), (states, controller_width)
    )
    gate_controller = generator.normal(
        0.0, 1.0 / math.sqrt(controller_width), (states, controller_width)
    )
    transition_bias = np.linspace(-0.35, 0.35, states)
    gate_bias = generator.normal(0.0, 0.1, states)
    input_projection = generator.normal(0.0, 1.0 / math.sqrt(states), states)
    output_projection = generator.normal(0.0, 1.0 / math.sqrt(states), states)
    direct = float(generator.normal(0.0, 0.05))

    transitions = np.empty((values.size, states), dtype=np.float64)
    gates = np.empty_like(transitions)
    state_trace = np.empty_like(transitions)
    output = np.empty(values.size, dtype=np.float64)
    state = np.zeros(states, dtype=np.float64)
    for index, sample in enumerate(values):
        feature = np.concatenate((summary, [math.tanh(float(sample) / scale)]))
        transition = rho * np.tanh(transition_controller @ feature + transition_bias)
        transition = np.clip(transition, -rho + 1e-12, rho - 1e-12)
        gate = _sigmoid(gate_controller @ feature + gate_bias)
        state = transition * state + gate * input_projection * sample
        transitions[index] = transition
        gates[index] = gate
        state_trace[index] = state
        output[index] = float(output_projection @ state + direct * sample)

    reference_feature = np.concatenate((summary, [math.tanh(float(np.mean(values)) / scale)]))
    reference_transition = rho * np.tanh(
        transition_controller @ reference_feature + transition_bias
    )
    reference_transition = np.clip(reference_transition, -rho + 1e-12, rho - 1e-12)
    reference_gate = _sigmoid(gate_controller @ reference_feature + gate_bias)
    powers = np.arange(length, dtype=np.int64)[:, None]
    kernel = np.sum(
        (reference_transition[None, :] ** powers)
        * (reference_gate * input_projection * output_projection)[None, :],
        axis=1,
    )
    kernel[0] += direct

    output_ro = _readonly(output, dtype=np.float64)
    kernel_ro = _readonly(kernel, dtype=np.float64)
    transition_ro = _readonly(transitions, dtype=np.float64)
    gate_ro = _readonly(gates, dtype=np.float64)
    state_ro = _readonly(state_trace, dtype=np.float64)
    summary_ro = _readonly(summary, dtype=np.float64)
    spectral_radius = float(np.max(np.abs(transitions)))
    record = _record(
        tool_key="recurrent_kernel",
        algorithm="selective-diagonal-recurrence-v1",
        assurance="approximation",
        seed=seed_value,
        inputs={"stream": values},
        outputs={
            "output": ("selective recurrence output", output_ro),
            "reference_kernel": ("context-frozen impulse response", kernel_ro),
            "transition_trace": ("input-selective diagonal transitions", transition_ro),
            "gate_trace": ("input-selective write gates", gate_ro),
            "state_trace": ("recurrent state trace", state_ro),
        },
        config={
            "state_size": states,
            "kernel_length": length,
            "contraction": rho,
        },
        diagnostics={
            "max_transition_magnitude": spectral_radius,
            "minimum_gate": float(np.min(gates)),
            "maximum_gate": float(np.max(gates)),
            "contractive_autonomous_transition": bool(spectral_radius < 1.0),
        },
        hints=(
            ToolHint(
                "success",
                "ssm.contractive",
                "Every autonomous diagonal transition is inside the requested contraction radius.",
            ),
            ToolHint(
                "warning",
                "ssm.prototype",
                "This is an inspectable selective recurrence, not a trained Mamba block or a "
                "hardware-fused throughput implementation.",
            ),
        ),
    )
    return RecurrentKernelResult(
        output_ro,
        kernel_ro,
        transition_ro,
        gate_ro,
        state_ro,
        summary_ro,
        record,
    )


def _polynomial_basis(
    coordinates: np.ndarray,
    degree: int,
    minimum: float,
    maximum: float,
    *,
    derivative_order: int = 0,
) -> np.ndarray:
    span = maximum - minimum
    normalized = 2.0 * (coordinates - minimum) / span - 1.0
    basis = np.zeros((coordinates.size, degree + 1), dtype=np.float64)
    chain = 2.0 / span
    for power in range(degree + 1):
        if derivative_order == 0:
            basis[:, power] = normalized**power
        elif derivative_order == 1 and power >= 1:
            basis[:, power] = power * normalized ** (power - 1) * chain
        elif derivative_order == 2 and power >= 2:
            basis[:, power] = power * (power - 1) * normalized ** (power - 2) * chain**2
    return basis


@dataclass(frozen=True, slots=True)
class ConstraintFitResult:
    coefficients: np.ndarray
    data_predictions: np.ndarray
    collocation_points: np.ndarray
    collocation_predictions: np.ndarray
    residuals: np.ndarray
    domain: tuple[float, float]
    target_was_1d: bool
    record: WeightToolRecord

    def predict(self, coordinates: Any) -> np.ndarray:
        values = _vector(coordinates, "coordinates")
        basis = _polynomial_basis(
            values,
            self.coefficients.shape[0] - 1,
            self.domain[0],
            self.domain[1],
        )
        result = basis @ self.coefficients
        return result[:, 0] if self.target_was_1d else result


def fit_physics_constrained_polynomial(
    coordinates: Any,
    observations: Any,
    *,
    degree: int = 5,
    second_derivative: float = 0.0,
    first_derivative: float = 1.0,
    value_coefficient: float = 1.0,
    source: Any = 0.0,
    physics_weight: float = 1.0,
    boundary_coordinates: Any | None = None,
    boundary_values: Any | None = None,
    boundary_weight: float = 1.0,
    collocation_count: int = 64,
    ridge: float = 1e-10,
) -> ConstraintFitResult:
    """Fit polynomial weights under a typed linear differential residual.

    The operator is ``a2*u'' + a1*u' + a0*u = source``.  This is a
    finite-basis physics-informed solver, not a higher-order-autodiff PINN.
    """

    x_data = _vector(coordinates, "coordinates")
    raw_observations = np.asarray(observations)
    target_was_1d = raw_observations.ndim == 1
    y_data = (
        _vector(raw_observations, "observations")[:, None]
        if target_was_1d
        else _matrix(raw_observations, "observations")
    )
    if y_data.shape[0] != x_data.size:
        raise WeightToolError("observations must contain one row per coordinate")
    polynomial_degree = _positive_int("degree", degree, maximum=24)
    collocation_rows = _positive_int("collocation_count", collocation_count, maximum=20_000)
    a2 = _finite_float("second_derivative", second_derivative)
    a1 = _finite_float("first_derivative", first_derivative)
    a0 = _finite_float("value_coefficient", value_coefficient)
    if a2 == 0.0 and a1 == 0.0 and a0 == 0.0:
        raise WeightToolError("At least one differential-operator coefficient must be non-zero")
    physics = _nonnegative_float("physics_weight", physics_weight)
    boundary_strength = _nonnegative_float("boundary_weight", boundary_weight)
    ridge_value = _nonnegative_float("ridge", ridge)

    if (boundary_coordinates is None) != (boundary_values is None):
        raise WeightToolError("boundary_coordinates and boundary_values must be supplied together")
    boundary_x: np.ndarray | None = None
    boundary_y: np.ndarray | None = None
    if boundary_coordinates is not None:
        boundary_x = _vector(boundary_coordinates, "boundary_coordinates")
        raw_boundary = np.asarray(boundary_values)
        if raw_boundary.ndim == 0:
            raw_boundary = np.full((boundary_x.size, y_data.shape[1]), float(raw_boundary))
        elif raw_boundary.ndim == 1:
            if y_data.shape[1] == 1 and raw_boundary.size == boundary_x.size:
                raw_boundary = raw_boundary[:, None]
            elif boundary_x.size == 1 and raw_boundary.size == y_data.shape[1]:
                raw_boundary = raw_boundary[None, :]
        boundary_y = _matrix(raw_boundary, "boundary_values")
        if boundary_y.shape != (boundary_x.size, y_data.shape[1]):
            raise WeightToolError(
                "boundary_values must match the boundary-coordinate and target dimensions"
            )

    domain_values = x_data if boundary_x is None else np.concatenate((x_data, boundary_x))
    minimum = float(np.min(domain_values))
    maximum = float(np.max(domain_values))
    if not maximum > minimum:
        raise WeightToolError("coordinates must span a non-zero domain")
    collocation = np.linspace(minimum, maximum, collocation_rows, dtype=np.float64)
    phi_data = _polynomial_basis(x_data, polynomial_degree, minimum, maximum)
    phi_collocation = _polynomial_basis(collocation, polynomial_degree, minimum, maximum)
    first = _polynomial_basis(
        collocation,
        polynomial_degree,
        minimum,
        maximum,
        derivative_order=1,
    )
    second = _polynomial_basis(
        collocation,
        polynomial_degree,
        minimum,
        maximum,
        derivative_order=2,
    )
    operator = a2 * second + a1 * first + a0 * phi_collocation

    source_array = np.asarray(source)
    if source_array.ndim == 0:
        source_matrix = np.full(
            (collocation_rows, y_data.shape[1]), float(source_array), dtype=np.float64
        )
    elif source_array.ndim == 1 and y_data.shape[1] == 1:
        source_matrix = _vector(source_array, "source")[:, None]
    else:
        source_matrix = _matrix(source_array, "source")
    if source_matrix.shape != (collocation_rows, y_data.shape[1]):
        raise WeightToolError(
            "source must be a scalar or match collocation_count by target dimensions"
        )

    design_parts = [phi_data / math.sqrt(x_data.size)]
    target_parts = [y_data / math.sqrt(x_data.size)]
    if physics > 0.0:
        factor = math.sqrt(physics / collocation_rows)
        design_parts.append(factor * operator)
        target_parts.append(factor * source_matrix)
    if boundary_x is not None and boundary_y is not None and boundary_strength > 0.0:
        phi_boundary = _polynomial_basis(
            boundary_x, polynomial_degree, minimum, maximum
        )
        factor = math.sqrt(boundary_strength / boundary_x.size)
        design_parts.append(factor * phi_boundary)
        target_parts.append(factor * boundary_y)
    if ridge_value > 0.0:
        regularizer = np.eye(polynomial_degree + 1, dtype=np.float64)
        regularizer[0, 0] = 0.0
        design_parts.append(math.sqrt(ridge_value) * regularizer)
        target_parts.append(np.zeros((polynomial_degree + 1, y_data.shape[1])))

    design = np.vstack(design_parts)
    right_hand = np.vstack(target_parts)
    coefficients, _residual_values, rank, singular_values = np.linalg.lstsq(
        design, right_hand, rcond=None
    )
    data_predictions = phi_data @ coefficients
    collocation_predictions = phi_collocation @ coefficients
    residuals = operator @ coefficients - source_matrix
    data_loss = float(np.mean((data_predictions - y_data) ** 2))
    physics_loss = float(np.mean(residuals**2))
    boundary_loss = 0.0
    if boundary_x is not None and boundary_y is not None:
        phi_boundary = _polynomial_basis(
            boundary_x, polynomial_degree, minimum, maximum
        )
        boundary_loss = float(np.mean((phi_boundary @ coefficients - boundary_y) ** 2))
    ridge_loss = float(ridge_value * np.mean(coefficients[1:] ** 2))
    total_loss = data_loss + physics * physics_loss + boundary_strength * boundary_loss + ridge_loss
    condition = (
        float(singular_values[0] / singular_values[-1])
        if singular_values.size and singular_values[-1] > np.finfo(np.float64).eps
        else math.inf
    )

    coefficients_ro = _readonly(coefficients, dtype=np.float64)
    data_predictions_ro = _readonly(data_predictions, dtype=np.float64)
    collocation_ro = _readonly(collocation, dtype=np.float64)
    collocation_predictions_ro = _readonly(collocation_predictions, dtype=np.float64)
    residuals_ro = _readonly(residuals, dtype=np.float64)
    output_arrays = {
        "coefficients": ("polynomial model weights", coefficients_ro),
        "data_predictions": ("predictions at observations", data_predictions_ro),
        "collocation_predictions": (
            "predictions at physics collocation points",
            collocation_predictions_ro,
        ),
        "residuals": ("differential-operator residuals", residuals_ro),
    }
    input_arrays: dict[str, np.ndarray] = {
        "coordinates": x_data,
        "observations": y_data,
        "source": source_matrix,
    }
    if boundary_x is not None and boundary_y is not None:
        input_arrays["boundary_coordinates"] = boundary_x
        input_arrays["boundary_values"] = boundary_y
    record = _record(
        tool_key="constraint_optimizer",
        algorithm="physics-constrained-polynomial-lstsq-v1",
        assurance="numerical_fit",
        seed=None,
        inputs=input_arrays,
        outputs=output_arrays,
        config={
            "degree": polynomial_degree,
            "second_derivative": a2,
            "first_derivative": a1,
            "value_coefficient": a0,
            "physics_weight": physics,
            "boundary_weight": boundary_strength,
            "collocation_count": collocation_rows,
            "ridge": ridge_value,
        },
        diagnostics={
            "data_loss": data_loss,
            "physics_loss": physics_loss,
            "boundary_loss": boundary_loss,
            "ridge_loss": ridge_loss,
            "total_loss": total_loss,
            "design_rank": int(rank),
            "condition_number": condition,
        },
        hints=(
            ToolHint(
                "info",
                "constraint.loss_balance",
                "Compare data and physics losses separately; a small total can hide a poorly "
                "balanced objective.",
            ),
            ToolHint(
                "warning",
                "constraint.finite_basis",
                "This bounded linear-basis solver does not provide a full neural PINN or "
                "higher-order automatic differentiation.",
            ),
        ),
    )
    return ConstraintFitResult(
        coefficients_ro,
        data_predictions_ro,
        collocation_ro,
        collocation_predictions_ro,
        residuals_ro,
        (minimum, maximum),
        target_was_1d,
        record,
    )


def _activation(values: np.ndarray, name: str) -> np.ndarray:
    normalized = str(name).strip().casefold()
    if normalized == "tanh":
        return np.tanh(values)
    if normalized == "relu":
        return np.maximum(values, 0.0)
    if normalized == "sigmoid":
        return _sigmoid(values)
    raise WeightToolError("activation must be tanh, relu, or sigmoid")


@dataclass(frozen=True, slots=True)
class ExtremeLearningMachineResult:
    input_weights: np.ndarray
    hidden_bias: np.ndarray
    output_weights: np.ndarray
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    predictions: np.ndarray
    activation: str
    target_was_1d: bool
    record: WeightToolRecord

    def predict(self, features: Any) -> np.ndarray:
        values = _matrix(features, "features")
        if values.shape[1] != self.input_weights.shape[0]:
            raise WeightToolError(
                f"features must have {self.input_weights.shape[0]} columns"
            )
        normalized = (values - self.feature_mean) / self.feature_scale
        hidden = _activation(normalized @ self.input_weights + self.hidden_bias, self.activation)
        predictions = hidden @ self.output_weights
        return predictions[:, 0] if self.target_was_1d else predictions


def fit_extreme_learning_machine(
    features: Any,
    targets: Any,
    *,
    hidden_units: int = 32,
    activation: str = "tanh",
    ridge: float = 1e-6,
    standardize: bool = True,
    seed: int = 47,
) -> ExtremeLearningMachineResult:
    """Fit a single-hidden-layer random-feature model by stable least squares."""

    x_values = _matrix(features, "features")
    raw_targets = np.asarray(targets)
    target_was_1d = raw_targets.ndim == 1
    y_values = (
        _vector(raw_targets, "targets")[:, None]
        if target_was_1d
        else _matrix(raw_targets, "targets")
    )
    if y_values.shape[0] != x_values.shape[0]:
        raise WeightToolError("targets must contain one row for every feature row")
    hidden_count = _positive_int("hidden_units", hidden_units, maximum=8_192)
    if x_values.shape[0] * hidden_count > MAX_ELM_FEATURE_ELEMENTS:
        raise WeightToolError("sample count times hidden units exceeds the ELM safety bound")
    ridge_value = _nonnegative_float("ridge", ridge)
    seed_value = _seed(seed)
    activation_name = str(activation).strip().casefold()
    _activation(np.zeros(1), activation_name)

    if standardize:
        mean = np.mean(x_values, axis=0)
        scale = np.std(x_values, axis=0)
        scale = np.where(scale > np.finfo(np.float64).eps, scale, 1.0)
    else:
        mean = np.zeros(x_values.shape[1], dtype=np.float64)
        scale = np.ones(x_values.shape[1], dtype=np.float64)
    normalized = (x_values - mean) / scale
    generator = np.random.default_rng(seed_value)
    input_weights = generator.normal(
        0.0, 1.0 / math.sqrt(x_values.shape[1]), (x_values.shape[1], hidden_count)
    )
    hidden_bias = generator.uniform(-1.0, 1.0, hidden_count)
    hidden = _activation(normalized @ input_weights + hidden_bias, activation_name)
    if ridge_value > 0.0:
        design = np.vstack((hidden, math.sqrt(ridge_value) * np.eye(hidden_count)))
        right_hand = np.vstack((y_values, np.zeros((hidden_count, y_values.shape[1]))))
    else:
        design = hidden
        right_hand = y_values
    output_weights, _residuals, rank, singular_values = np.linalg.lstsq(
        design, right_hand, rcond=None
    )
    predictions = hidden @ output_weights
    rmse = float(np.sqrt(np.mean((predictions - y_values) ** 2)))
    condition = (
        float(singular_values[0] / singular_values[-1])
        if singular_values.size and singular_values[-1] > np.finfo(np.float64).eps
        else math.inf
    )
    hints = [
        ToolHint(
            "info",
            "elm.validation",
            "Closed-form training error is not generalization evidence; reserve held-out rows.",
        )
    ]
    if condition > 1e8:
        hints.append(
            ToolHint(
                "warning",
                "elm.conditioning",
                "The hidden design is poorly conditioned; increase ridge, reduce hidden units, "
                "or improve feature scaling.",
            )
        )

    input_weights_ro = _readonly(input_weights, dtype=np.float64)
    hidden_bias_ro = _readonly(hidden_bias, dtype=np.float64)
    output_weights_ro = _readonly(output_weights, dtype=np.float64)
    mean_ro = _readonly(mean, dtype=np.float64)
    scale_ro = _readonly(scale, dtype=np.float64)
    predictions_ro = _readonly(predictions, dtype=np.float64)
    record = _record(
        tool_key="matrix_inverter",
        algorithm="extreme-learning-machine-lstsq-v1",
        assurance="numerical_fit",
        seed=seed_value,
        inputs={"features": x_values, "targets": y_values},
        outputs={
            "input_weights": ("frozen random hidden weights", input_weights_ro),
            "hidden_bias": ("frozen random hidden bias", hidden_bias_ro),
            "output_weights": ("least-squares output weights", output_weights_ro),
            "predictions": ("training predictions", predictions_ro),
        },
        config={
            "hidden_units": hidden_count,
            "activation": activation_name,
            "ridge": ridge_value,
            "standardize": bool(standardize),
        },
        diagnostics={
            "training_rmse": rmse,
            "design_rank": int(rank),
            "condition_number": condition,
            "sample_count": int(x_values.shape[0]),
            "feature_count": int(x_values.shape[1]),
            "target_count": int(y_values.shape[1]),
        },
        hints=hints,
    )
    return ExtremeLearningMachineResult(
        input_weights_ro,
        hidden_bias_ro,
        output_weights_ro,
        mean_ro,
        scale_ro,
        predictions_ro,
        activation_name,
        target_was_1d,
        record,
    )


def _squared_distances(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    distances = (
        np.sum(left * left, axis=1)[:, None]
        + np.sum(right * right, axis=1)[None, :]
        - 2.0 * (left @ right.T)
    )
    return np.maximum(distances, 0.0)


@dataclass(frozen=True, slots=True)
class UncertaintySampleResult:
    posterior_mean: np.ndarray
    posterior_std: np.ndarray
    acquisition_scores: np.ndarray
    selected_indices: np.ndarray
    selected_candidates: np.ndarray
    record: WeightToolRecord


def select_uncertain_candidates(
    labeled_features: Any,
    labeled_targets: Any,
    candidate_features: Any,
    *,
    query_count: int = 1,
    length_scale: float = 1.0,
    signal_std: float = 1.0,
    noise_std: float = 1e-4,
) -> UncertaintySampleResult:
    """Rank an explicit candidate pool by RBF-GP posterior standard deviation."""

    x_train = _matrix(labeled_features, "labeled_features")
    y_train = _vector(labeled_targets, "labeled_targets")
    x_pool = _matrix(candidate_features, "candidate_features")
    if x_train.shape[0] != y_train.size:
        raise WeightToolError("labeled_targets must match labeled_features rows")
    if x_pool.shape[1] != x_train.shape[1]:
        raise WeightToolError("candidate_features must use the same columns as labeled_features")
    if x_train.shape[0] > MAX_GP_TRAINING_ROWS:
        raise WeightToolError(
            f"Gaussian-process training rows cannot exceed {MAX_GP_TRAINING_ROWS:,}"
        )
    if x_train.shape[0] * x_pool.shape[0] > MAX_GP_CROSS_ELEMENTS:
        raise WeightToolError("labeled rows times candidate rows exceeds the GP safety bound")
    queries = _positive_int("query_count", query_count, maximum=x_pool.shape[0])
    if queries * x_pool.shape[0] > MAX_GP_CROSS_ELEMENTS:
        raise WeightToolError(
            "query_count times candidate rows exceeds the greedy-batch safety bound"
        )
    length = _finite_float("length_scale", length_scale)
    signal = _finite_float("signal_std", signal_std)
    noise = _nonnegative_float("noise_std", noise_std)
    if length <= 0.0:
        raise WeightToolError("length_scale must be positive")
    if signal <= 0.0:
        raise WeightToolError("signal_std must be positive")

    kernel_train = signal**2 * np.exp(
        -0.5 * _squared_distances(x_train, x_train) / length**2
    )
    cross = signal**2 * np.exp(-0.5 * _squared_distances(x_train, x_pool) / length**2)
    base_diagonal = noise**2
    jitter = max(signal**2 * 1e-12, np.finfo(np.float64).eps)
    identity = np.eye(x_train.shape[0], dtype=np.float64)
    cholesky: np.ndarray | None = None
    for _attempt in range(9):
        try:
            cholesky = np.linalg.cholesky(
                kernel_train + (base_diagonal + jitter) * identity
            )
            break
        except np.linalg.LinAlgError:
            jitter *= 10.0
    if cholesky is None:
        raise WeightToolError(
            "The Gaussian-process kernel remained singular after bounded jitter escalation"
        )

    target_mean = float(np.mean(y_train))
    centered_targets = y_train - target_mean
    alpha = np.linalg.solve(cholesky.T, np.linalg.solve(cholesky, centered_targets))
    posterior_mean = target_mean + cross.T @ alpha
    triangular = np.linalg.solve(cholesky, cross)
    variance = signal**2 - np.sum(triangular * triangular, axis=0)
    variance = np.maximum(variance, 0.0)
    posterior_std = np.sqrt(variance)
    scores = posterior_std.copy()

    observed = np.zeros(x_pool.shape[0], dtype=bool)
    for index, candidate in enumerate(x_pool):
        observed[index] = bool(
            np.any(np.all(np.isclose(x_train, candidate, rtol=0.0, atol=1e-12), axis=1))
        )
    scores[observed] = -math.inf
    eligible = np.flatnonzero(~observed)
    if eligible.size < queries:
        raise WeightToolError("Not enough unobserved candidate rows remain for query_count")
    # Greedy conditional-variance selection avoids returning a batch of near-
    # duplicate high-variance points.  Each selected latent value is treated as
    # a bounded fantasy observation and conditions the remaining pool without
    # materializing the full pool-by-pool covariance matrix.
    working_variance = variance.copy()
    selected_mask = observed.copy()
    variance_updates: list[np.ndarray] = []
    selected: list[int] = []
    for _query in range(queries):
        remaining = np.flatnonzero(~selected_mask)
        current_scores = np.sqrt(np.maximum(working_variance[remaining], 0.0))
        chosen = int(remaining[int(np.argmax(current_scores))])
        selected.append(chosen)
        selected_mask[chosen] = True

        pool_cross = signal**2 * np.exp(
            -0.5
            * _squared_distances(x_pool[chosen : chosen + 1], x_pool)[0]
            / length**2
        )
        conditional_covariance = pool_cross - triangular[:, chosen] @ triangular
        for update in variance_updates:
            conditional_covariance -= update[chosen] * update
        denominator = math.sqrt(
            max(float(conditional_covariance[chosen]) + noise**2, jitter)
        )
        update = conditional_covariance / denominator
        variance_updates.append(update)
        working_variance = np.maximum(working_variance - update * update, 0.0)

    selected_indices = np.asarray(selected, dtype=np.int64)
    selected_candidates = x_pool[selected_indices]

    mean_ro = _readonly(posterior_mean, dtype=np.float64)
    std_ro = _readonly(posterior_std, dtype=np.float64)
    scores_ro = _readonly(scores, dtype=np.float64)
    indices_ro = _readonly(selected_indices, dtype=np.int64)
    candidates_ro = _readonly(selected_candidates, dtype=np.float64)
    record = _record(
        tool_key="uncertainty_sampler",
        algorithm="rbf-gp-max-variance-v1",
        assurance="approximation",
        seed=None,
        inputs={
            "labeled_features": x_train,
            "labeled_targets": y_train,
            "candidate_features": x_pool,
        },
        outputs={
            "posterior_mean": ("candidate posterior mean", mean_ro),
            "posterior_std": ("candidate posterior standard deviation", std_ro),
            "acquisition_scores": ("maximum-variance acquisition score", scores_ro),
            "selected_indices": ("selected candidate row indices", indices_ro),
            "selected_candidates": ("selected candidate feature rows", candidates_ro),
        },
        config={
            "query_count": queries,
            "length_scale": length,
            "signal_std": signal,
            "noise_std": noise,
        },
        diagnostics={
            "labeled_count": int(x_train.shape[0]),
            "candidate_count": int(x_pool.shape[0]),
            "excluded_observed_candidates": int(np.sum(observed)),
            "effective_jitter": jitter,
            "maximum_posterior_std": float(np.max(posterior_std)),
            "batch_strategy": "greedy-conditional-variance",
        },
        hints=(
            ToolHint(
                "info",
                "gp.next_sample",
                "Acquire the selected label, append it to the labeled set, then recompute the posterior.",
            ),
            ToolHint(
                "warning",
                "gp.scope",
                "This uncertainty is over the explicit candidate inputs under one RBF kernel; "
                "it is not uncertainty over the full weight space or a safety guarantee.",
            ),
        ),
    )
    return UncertaintySampleResult(
        mean_ro,
        std_ro,
        scores_ro,
        indices_ro,
        candidates_ro,
        record,
    )


__all__ = [
    "ArrayDescriptor",
    "ConstraintFitResult",
    "ExtremeLearningMachineResult",
    "HypernetworkResult",
    "RecurrentKernelResult",
    "ToolHint",
    "TruthTableCompilation",
    "UncertaintySampleResult",
    "WEIGHT_TOOL_SPECS",
    "WeightToolError",
    "WeightToolRecord",
    "WeightToolSpec",
    "compile_truth_table",
    "fit_extreme_learning_machine",
    "fit_physics_constrained_polynomial",
    "get_weight_tool_spec",
    "recommend_weight_tool",
    "select_uncertain_candidates",
    "synthesize_low_rank_adapter",
    "synthesize_recurrent_kernel",
]
