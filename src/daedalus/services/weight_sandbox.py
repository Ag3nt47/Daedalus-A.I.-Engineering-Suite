"""Private-project sandbox drafts for Weight Lab experiments.

Verified Weight Lab algorithms are immutable application code.  These drafts
are deliberately separate, visible project files executed only by the existing
constrained subprocess runner.  The runner limits ordinary mistakes; hostile
code still belongs in a disposable virtual machine.
"""

from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass
from pathlib import Path

from daedalus.engine.weight_tools import WEIGHT_TOOL_SPECS, WeightToolError
from daedalus.services.sandbox import RunResult, SandboxRunner, inspect_source
from daedalus.workspace.manager import WorkspaceManager

MAX_SANDBOX_SOURCE_BYTES = 512 * 1024
SANDBOX_RELATIVE_DIRECTORY = Path("experiments") / "weight_lab"

_HEADER = '''"""Private Daedalus Weight Lab sandbox.

This file is an experimental extension, not a verified built-in.  It runs in
the constrained project subprocess.  Do not use it for hostile code.
"""

from __future__ import annotations

import numpy as np

TOOL_ID = {tool_id!r}
API_VERSION = 1

'''

_BODIES: dict[str, str] = {
    "meta_weight": '''def extend(context: np.ndarray, rows: int = 3, columns: int = 4) -> np.ndarray:
    """Prototype a context-to-low-rank delta; replace this with your controller."""
    values = np.asarray(context, dtype=float).reshape(-1)
    rng = np.random.default_rng(47)
    rank = min(2, rows, columns)
    gates = np.tanh(values @ rng.normal(size=(values.size, rank)))
    left = rng.normal(size=(rows, rank)) * gates
    right = rng.normal(size=(rank, columns))
    return left @ right / np.sqrt(rank)


def main() -> None:
    delta = extend(np.array([1.0, 0.5, -0.25]))
    print({"tool": TOOL_ID, "shape": list(delta.shape), "norm": float(np.linalg.norm(delta))})
''',
    "logic_compiler": '''def extend(rows: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compile binary pattern detectors; keep inputs as data and never evaluate code strings."""
    table = np.asarray(rows, dtype=float)
    output = np.asarray(targets, dtype=float).reshape(table.shape[0], -1)
    hidden_weights = (2.0 * table - 1.0).T
    hidden_bias = 0.5 - table.sum(axis=1)
    return hidden_weights, hidden_bias, output


def main() -> None:
    rows = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    weights, bias, output = extend(rows, np.array([0, 1, 1, 0]))
    prediction = ((rows @ weights + bias) > 0.0).astype(float) @ output
    print({"tool": TOOL_ID, "prediction": prediction[:, 0].tolist()})
''',
    "recurrent_kernel": '''def extend(stream: np.ndarray, state_size: int = 4, contraction: float = 0.9) -> np.ndarray:
    """Prototype a contractive diagonal recurrence; return one output per input."""
    values = np.asarray(stream, dtype=float).reshape(-1)
    transition = np.linspace(0.2, contraction, state_size)
    state = np.zeros(state_size)
    output = []
    for sample in values:
        state = transition * state + sample / np.sqrt(state_size)
        output.append(float(state.mean()))
    return np.asarray(output)


def main() -> None:
    output = extend(np.array([1.0, 0.5, 0.25, 0.125]))
    print({"tool": TOOL_ID, "output": output.round(6).tolist()})
''',
    "constraint_optimizer": '''def extend(x: np.ndarray, y: np.ndarray, physics_weight: float = 5.0) -> np.ndarray:
    """Fit u' + u = 0 with a polynomial basis and a typed residual matrix."""
    coordinates = np.asarray(x, dtype=float).reshape(-1)
    targets = np.asarray(y, dtype=float).reshape(-1, 1)
    degree = 4
    basis = np.vander(coordinates, degree + 1, increasing=True)
    derivative = np.zeros_like(basis)
    for power in range(1, degree + 1):
        derivative[:, power] = power * coordinates ** (power - 1)
    residual = derivative + basis
    design = np.vstack([basis, np.sqrt(physics_weight) * residual])
    right = np.vstack([targets, np.zeros_like(targets)])
    coefficients, *_ = np.linalg.lstsq(design, right, rcond=None)
    return coefficients[:, 0]


def main() -> None:
    x = np.array([0.0, 0.5, 1.0])
    coefficients = extend(x, np.exp(-x))
    print({"tool": TOOL_ID, "coefficients": coefficients.round(6).tolist()})
''',
    "matrix_inverter": '''def extend(features: np.ndarray, targets: np.ndarray, hidden_units: int = 8) -> np.ndarray:
    """Solve ELM output weights with least squares; avoid an explicit matrix inverse."""
    x = np.asarray(features, dtype=float)
    y = np.asarray(targets, dtype=float).reshape(x.shape[0], -1)
    rng = np.random.default_rng(47)
    hidden = np.tanh(x @ rng.normal(size=(x.shape[1], hidden_units)) + rng.normal(size=hidden_units))
    beta, *_ = np.linalg.lstsq(hidden, y, rcond=None)
    return beta


def main() -> None:
    x = np.array([[0.0], [0.5], [1.0], [1.5]])
    beta = extend(x, np.sin(x[:, 0]))
    print({"tool": TOOL_ID, "output_weight_shape": list(beta.shape)})
''',
    "uncertainty_sampler": '''def extend(labeled: np.ndarray, candidates: np.ndarray, length_scale: float = 1.0) -> int:
    """Return the candidate with greatest geometry-only RBF posterior variance."""
    x = np.asarray(labeled, dtype=float)
    pool = np.asarray(candidates, dtype=float)
    distance = lambda a, b: np.maximum((a * a).sum(1)[:, None] + (b * b).sum(1)[None, :] - 2 * a @ b.T, 0.0)
    kernel = np.exp(-0.5 * distance(x, x) / length_scale**2) + 1e-8 * np.eye(x.shape[0])
    cross = np.exp(-0.5 * distance(x, pool) / length_scale**2)
    factor = np.linalg.cholesky(kernel)
    variance = np.maximum(1.0 - (np.linalg.solve(factor, cross) ** 2).sum(0), 0.0)
    return int(np.argmax(variance))


def main() -> None:
    index = extend(np.array([[0.0], [1.0]]), np.array([[-1.0], [0.5], [2.0]]))
    print({"tool": TOOL_ID, "selected_index": index})
''',
}

_VALID_TOOL_KEYS = frozenset(item.key for item in WEIGHT_TOOL_SPECS)
if set(_BODIES) != _VALID_TOOL_KEYS:  # pragma: no cover - import-time developer invariant
    raise RuntimeError("Weight Lab sandbox templates do not match the built-in tool catalog")


def sandbox_template(tool_key: str) -> str:
    key = str(tool_key)
    if key not in _VALID_TOOL_KEYS:
        raise WeightToolError(f"Unknown Weight Lab sandbox tool: {tool_key!r}")
    return _HEADER.format(tool_id=key) + _BODIES[key] + "\n\nif __name__ == \"__main__\":\n    main()\n"


def sandbox_template_sha256(tool_key: str) -> str:
    return hashlib.sha256(sandbox_template(tool_key).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class SandboxDraft:
    tool_key: str
    project: Path
    path: Path
    source: str
    exists: bool
    template_sha256: str


class WeightSandboxService:
    """Resolve, create, save, and run one visible draft per project/tool."""

    def __init__(self, manager: WorkspaceManager, *, timeout_seconds: float = 15.0) -> None:
        self.manager = manager
        self.runner = SandboxRunner(manager, timeout_seconds=timeout_seconds)

    def _project(self, project: str | os.PathLike[str]) -> Path:
        original = Path(project)
        if original.is_symlink() or self.manager.projects_dir.is_symlink():
            raise PermissionError("Weight Lab sandbox projects cannot be symbolic links.")
        resolved = self.manager.resolve_user_path(original, must_exist=True)
        projects = self.manager.projects_dir.resolve(strict=True)
        if resolved.parent != projects or not resolved.is_dir():
            raise PermissionError("Weight Lab sandboxes require a direct private project directory.")
        return resolved

    @staticmethod
    def _key(tool_key: str) -> str:
        key = str(tool_key)
        if key not in _VALID_TOOL_KEYS:
            raise WeightToolError(f"Unknown Weight Lab sandbox tool: {tool_key!r}")
        return key

    def draft_path(self, project: str | os.PathLike[str], tool_key: str) -> Path:
        resolved = self._project(project)
        key = self._key(tool_key)
        return resolved / SANDBOX_RELATIVE_DIRECTORY / f"{key}.py"

    def load(self, project: str | os.PathLike[str], tool_key: str) -> SandboxDraft:
        resolved = self._project(project)
        key = self._key(tool_key)
        path = resolved / SANDBOX_RELATIVE_DIRECTORY / f"{key}.py"
        if path.is_symlink():
            raise PermissionError("Weight Lab sandbox files cannot be symbolic links.")
        exists = path.exists()
        if exists:
            if not path.is_file():
                raise ValueError("The Weight Lab sandbox path is not a regular file.")
            if path.stat().st_size > MAX_SANDBOX_SOURCE_BYTES:
                raise ValueError("The Weight Lab sandbox source limit is 512 KiB.")
            source = path.read_text(encoding="utf-8")
        else:
            source = sandbox_template(key)
        return SandboxDraft(key, resolved, path, source, exists, sandbox_template_sha256(key))

    def _ensure_directory(self, project: Path) -> Path:
        experiments = project / "experiments"
        directory = project / SANDBOX_RELATIVE_DIRECTORY
        for item in (experiments, directory):
            if item.is_symlink():
                raise PermissionError("Weight Lab sandbox directories cannot be symbolic links.")
            item.mkdir(exist_ok=True)
            if not item.is_dir():
                raise ValueError("Weight Lab sandbox directory is not a directory.")
        return directory

    def create(self, project: str | os.PathLike[str], tool_key: str, source: str | None = None) -> Path:
        resolved = self._project(project)
        key = self._key(tool_key)
        directory = self._ensure_directory(resolved)
        path = directory / f"{key}.py"
        if path.is_symlink():
            raise PermissionError("Weight Lab sandbox files cannot be symbolic links.")
        text = sandbox_template(key) if source is None else str(source)
        self._validate_source(text)
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        return path

    def save(self, project: str | os.PathLike[str], tool_key: str, source: str) -> Path:
        path = self.draft_path(project, tool_key)
        if not path.exists():
            raise FileNotFoundError("Create the private sandbox draft before saving it.")
        if path.is_symlink() or not path.is_file():
            raise PermissionError("Weight Lab sandbox files must be regular non-symlink files.")
        text = str(source)
        self._validate_source(text)
        temporary = path.with_name(f".{path.name}.daedalus-{time.time_ns()}.tmp")
        try:
            temporary.write_text(text, encoding="utf-8", newline="\n")
            temporary.replace(path)
        except OSError:
            temporary.unlink(missing_ok=True)
            raise
        return path

    @staticmethod
    def _validate_source(source: str) -> None:
        encoded = source.encode("utf-8")
        if len(encoded) > MAX_SANDBOX_SOURCE_BYTES:
            raise ValueError("The Weight Lab sandbox source limit is 512 KiB.")
        inspect_source(source)

    def run(self, project: str | os.PathLike[str], tool_key: str) -> RunResult:
        path = self.draft_path(project, tool_key)
        return self.runner.run_file(path)


__all__ = [
    "MAX_SANDBOX_SOURCE_BYTES",
    "SANDBOX_RELATIVE_DIRECTORY",
    "SandboxDraft",
    "WeightSandboxService",
    "sandbox_template",
    "sandbox_template_sha256",
]
