"""Local project setup, toolchain, and reproducibility evidence.

The service inspects metadata and hashes bounded source files.  It never imports
project code, installs a dependency, contacts a package index, or sends project
information over the network.  Optional professional tools are capabilities,
not requirements: their absence is reported without making a project unhealthy.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import shutil
import sys
import tomllib
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import StrEnum
from functools import lru_cache
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import TYPE_CHECKING, Any, Mapping

from daedalus import __version__ as DAEDALUS_VERSION

if TYPE_CHECKING:
    from daedalus.workspace.manager import WorkspaceManager


SNAPSHOT_SCHEMA = 1
RUN_MANIFEST_SCHEMA = 1
MAX_SOURCE_FILES = 1_000
MAX_SOURCE_ENTRIES = 10_000
MAX_SOURCE_FILE_BYTES = 2 * 1024 * 1024
MAX_SOURCE_BYTES = 16 * 1024 * 1024
MAX_DEPENDENCIES = 2_000
MAX_JSON_NODES = 20_000
MAX_JSON_BYTES = 2 * 1024 * 1024

ENVIRONMENT_SNAPSHOT = "ENVIRONMENT_SNAPSHOT.json"
ENVIRONMENT_LOCK = "environment.lock.json"

_SOURCE_SUFFIXES = {".py", ".json", ".toml", ".yaml", ".yml"}
_SKIP_DIRECTORIES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "checkpoints",
    "data",
    "logs",
    "node_modules",
    "runs",
}
_EXCLUDED_SNAPSHOT_NAMES = {ENVIRONMENT_LOCK.casefold(), ENVIRONMENT_SNAPSHOT.casefold()}
_SAFE_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,99}\Z")
_SECRET_PATTERNS = (
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----", re.IGNORECASE),
    re.compile(r"\bAKIA[A-Z0-9]{16}\b"),
    re.compile(r"\b(?:sk-[A-Za-z0-9_-]{12,}|ghp_[A-Za-z0-9]{12,})\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{12,}\b"),
)
_SECRET_KEYS = ("api_key", "password", "secret", "token", "credential")


class StandardStatus(StrEnum):
    """Severity/readiness state for one standards finding."""

    READY = "ready"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class StandardFinding:
    code: str
    status: StandardStatus
    summary: str
    location: str = ""
    action: str = ""


@dataclass(frozen=True, slots=True)
class ToolCapability:
    key: str
    label: str
    category: str
    available: bool
    version: str | None
    optional: bool
    purpose: str


@dataclass(frozen=True, slots=True)
class DependencyVersion:
    name: str
    version: str


@dataclass(frozen=True, slots=True)
class EnvironmentSnapshot:
    """Redacted, JSON-safe description of the active Daedalus environment."""

    schema: int
    kind: str
    captured_utc: str
    project_name: str
    python_implementation: str
    python_version: str
    daedalus_version: str
    operating_system: str
    operating_system_release: str
    architecture: str
    isolated_environment: bool
    environment_kind: str
    logical_cpu_count: int
    memory_bytes: int | None
    device_capability: str
    dependencies: tuple[DependencyVersion, ...]
    dependency_inventory_truncated: bool
    source_sha256: str
    source_files_hashed: int
    source_bytes_hashed: int
    source_inventory_truncated: bool
    entrypoint_sha256: str | None
    pyproject_sha256: str | None
    dependency_lock_sha256: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ProjectStandardsReport:
    project_name: str
    inspected_utc: str
    environment: EnvironmentSnapshot
    findings: tuple[StandardFinding, ...]
    tools: tuple[ToolCapability, ...]

    @property
    def ok(self) -> bool:
        return not any(item.status == StandardStatus.ERROR for item in self.findings)

    @property
    def error_count(self) -> int:
        return sum(item.status == StandardStatus.ERROR for item in self.findings)

    @property
    def warning_count(self) -> int:
        return sum(item.status == StandardStatus.WARNING for item in self.findings)

    def format_text(self) -> str:
        lines = [
            f"Project setup & reproducibility — {self.project_name}",
            "=" * 58,
            (
                f"Python {self.environment.python_version} · "
                f"Daedalus {self.environment.daedalus_version} · "
                f"{len(self.environment.dependencies)} installed distribution(s) inventoried"
            ),
            "No project code was imported or executed; no network request was made.",
            "",
        ]
        if not self.findings:
            lines.append("READY  No project-standard problem was found.")
        for finding in self.findings:
            location = f" [{finding.location}]" if finding.location else ""
            lines.append(f"{finding.status.value.upper()}  {finding.summary}{location}")
            if finding.action:
                lines.append(f"        Next: {finding.action}")
        lines.extend(("", "OPTIONAL CAPABILITIES"))
        for tool in self.tools:
            state = "available" if tool.available else "not detected"
            version = f" {tool.version}" if tool.version else ""
            lines.append(f"- {tool.label}: {state}{version} — {tool.purpose}")
        return "\n".join(lines).rstrip() + "\n"


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _canonical_package_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.strip().casefold()).strip("-")[:200]


@lru_cache(maxsize=1)
def _installed_distributions() -> tuple[tuple[DependencyVersion, ...], bool]:
    versions: dict[str, str] = {}
    try:
        distributions = importlib.metadata.distributions()
        for distribution in distributions:
            try:
                raw_name = str(distribution.metadata.get("Name") or "").strip()
                raw_version = str(distribution.version or "").strip()
            except (KeyError, OSError, TypeError, UnicodeError):
                continue
            name = _canonical_package_name(raw_name)
            if not name or not raw_version or any(ord(character) < 32 for character in raw_version):
                continue
            safe_version = re.sub(r"[^A-Za-z0-9!+._-]+", "-", raw_version)[:200]
            if safe_version:
                versions.setdefault(name, safe_version)
    except (OSError, TypeError):
        return (), True
    ordered = tuple(
        DependencyVersion(name, version)
        for name, version in sorted(versions.items(), key=lambda item: item[0])
    )
    return ordered[:MAX_DEPENDENCIES], len(ordered) > MAX_DEPENDENCIES


def _sha256_file(path: Path, *, maximum_bytes: int | None = None) -> str | None:
    if path.is_symlink() or not path.is_file():
        return None
    try:
        size = path.stat().st_size
        if maximum_bytes is not None and size > maximum_bytes:
            return None
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
    except OSError:
        return None


def _source_fingerprint(project: Path) -> tuple[str, int, int, bool]:
    """Hash a bounded, name-delimited inventory without exposing source or paths."""

    candidates: list[Path] = []
    stack = [project]
    entries_seen = 0
    truncated = False
    while stack and entries_seen < MAX_SOURCE_ENTRIES and len(candidates) < MAX_SOURCE_FILES:
        directory = stack.pop()
        try:
            entries = sorted(directory.iterdir(), key=lambda item: item.name.casefold())
        except OSError:
            truncated = True
            continue
        for entry in entries:
            entries_seen += 1
            if entries_seen > MAX_SOURCE_ENTRIES:
                truncated = True
                break
            if entry.is_symlink():
                truncated = True
                continue
            if entry.is_dir():
                if entry.name.casefold() not in _SKIP_DIRECTORIES:
                    stack.append(entry)
                continue
            if (
                entry.suffix.casefold() in _SOURCE_SUFFIXES
                and entry.name.casefold() not in _EXCLUDED_SNAPSHOT_NAMES
                and not (
                    entry.parent.name.casefold() == "runs"
                    and entry.name.casefold().endswith(".manifest.json")
                )
            ):
                candidates.append(entry)
                if len(candidates) >= MAX_SOURCE_FILES:
                    truncated = True
                    break
    if stack or entries_seen >= MAX_SOURCE_ENTRIES:
        truncated = True

    digest = hashlib.sha256()
    files_hashed = 0
    bytes_hashed = 0
    for path in sorted(candidates, key=lambda item: item.relative_to(project).as_posix()):
        try:
            size = path.stat().st_size
        except OSError:
            truncated = True
            continue
        if size > MAX_SOURCE_FILE_BYTES or bytes_hashed + size > MAX_SOURCE_BYTES:
            truncated = True
            continue
        try:
            content = path.read_bytes()
        except OSError:
            truncated = True
            continue
        relative = path.relative_to(project).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
        files_hashed += 1
        bytes_hashed += len(content)
    return digest.hexdigest(), files_hashed, bytes_hashed, truncated


def _memory_bytes() -> int | None:
    try:
        import psutil

        return int(psutil.virtual_memory().total)
    except (ImportError, OSError, RuntimeError, ValueError):
        return None


def _environment_kind() -> tuple[bool, str]:
    if sys.prefix != getattr(sys, "base_prefix", sys.prefix):
        return True, "virtual-environment"
    if bool(os.getenv("CONDA_PREFIX")):
        return True, "conda-environment"
    return False, "system-interpreter"


def _runtime_snapshot_unvalidated(project: Path, *, project_name: str) -> EnvironmentSnapshot:
    dependencies, dependency_truncated = _installed_distributions()
    isolated, environment_kind = _environment_kind()
    source_hash, source_files, source_bytes, source_truncated = _source_fingerprint(project)
    lock = project / ENVIRONMENT_LOCK
    nvidia_cli = shutil.which("nvidia-smi")
    return EnvironmentSnapshot(
        schema=SNAPSHOT_SCHEMA,
        kind="daedalus-environment-snapshot",
        captured_utc=_utc_now(),
        project_name=project_name,
        python_implementation=platform.python_implementation(),
        python_version=platform.python_version(),
        daedalus_version=DAEDALUS_VERSION,
        operating_system=platform.system() or "unknown",
        operating_system_release=platform.release() or "unknown",
        architecture=platform.machine() or "unknown",
        isolated_environment=isolated,
        environment_kind=environment_kind,
        logical_cpu_count=int(os.cpu_count() or 1),
        memory_bytes=_memory_bytes(),
        device_capability=("NVIDIA tooling detected" if nvidia_cli else "CPU / no GPU CLI detected"),
        dependencies=dependencies,
        dependency_inventory_truncated=dependency_truncated,
        source_sha256=source_hash,
        source_files_hashed=source_files,
        source_bytes_hashed=source_bytes,
        source_inventory_truncated=source_truncated,
        entrypoint_sha256=_sha256_file(project / "main.py", maximum_bytes=MAX_SOURCE_FILE_BYTES),
        pyproject_sha256=_sha256_file(
            project / "pyproject.toml", maximum_bytes=MAX_SOURCE_FILE_BYTES
        ),
        dependency_lock_sha256=_sha256_file(lock, maximum_bytes=MAX_SOURCE_FILE_BYTES),
    )


def _toml_project_name(project_name: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", project_name.casefold()).strip("-")[:200]
    return normalized or "daedalus-project"


def _baseline_files(
    project: Path,
    *,
    project_name: str,
    template: str,
) -> dict[PurePosixPath, str]:
    package_name = _toml_project_name(project_name)
    dependencies = (
        '["numpy>=2,<3"]'
        if template == "minimal"
        else '["daedalus-ai-suite>=0.1,<0.2", "numpy>=2,<3"]'
    )
    pyproject = (
        "[project]\n"
        f'name = "{package_name}"\n'
        'version = "0.1.0"\n'
        'requires-python = ">=3.11,<3.15"\n'
        f"dependencies = {dependencies}\n\n"
        "[tool.daedalus]\n"
        "schema = 1\n"
        f'template = "{template}"\n'
        'entrypoint = "main.py"\n'
        'environment-snapshot = "ENVIRONMENT_SNAPSHOT.json"\n'
        'environment-lock = "environment.lock.json"\n'
        "default-seed = 47\n\n"
        "[tool.pytest.ini_options]\n"
        'testpaths = ["tests"]\n\n'
        "[tool.ruff]\n"
        'target-version = "py311"\n'
        "line-length = 100\n"
    )
    gitignore = (
        ".env\n"
        ".venv/\n"
        "__pycache__/\n"
        ".pytest_cache/\n"
        ".ruff_cache/\n"
        "*.py[cod]\n"
        "logs/*\n"
        "!logs/README.txt\n"
        "data/*\n"
        "!data/.gitkeep\n"
        "checkpoints/*\n"
        "!checkpoints/.gitkeep\n"
        "*.npz\n"
        "*.pt\n"
        "*.pth\n"
        "*.safetensors\n"
    )
    smoke_test = (
        '"""Starter contract tests that inspect code without executing a training run."""\n\n'
        "import ast\n"
        "from pathlib import Path\n\n\n"
        "def test_entrypoint_is_valid_python() -> None:\n"
        '    entrypoint = Path(__file__).parents[1] / "main.py"\n'
        "    ast.parse(entrypoint.read_text(encoding=\"utf-8\"), filename=str(entrypoint))\n"
    )
    card_guide = (
        "# Evidence cards\n\n"
        "Use these templates while gathering evidence. A template is not a completed gate. "
        "The AI Developer Bot generates the canonical root-level cards after the required "
        "answers and evidence are available.\n"
    )
    dataset_card = (
        "# Dataset card template\n\n"
        "Status: DRAFT TEMPLATE — not release evidence.\n\n"
        "Document origin and owner, permitted use, target definition, population, time range, "
        "missingness, sensitive fields, leakage risks, split policy, checksum, and limitations.\n"
    )
    model_card = (
        "# Model card template\n\n"
        "Status: DRAFT TEMPLATE — not release evidence.\n\n"
        "Document intended and out-of-scope use, architecture, training data identity, held-out "
        "metrics, slices, robustness checks, limitations, human oversight, and rollback owner.\n"
    )
    default_config = json.dumps(
        {
            "schema": 1,
            "kind": "daedalus-project-config",
            "entrypoint": "main.py",
            "seed": 47,
            "training": {
                "batch_size": 32,
                "early_stopping_patience": 25,
                "epochs": 500,
            },
        },
        indent=2,
        sort_keys=True,
    ) + "\n"
    deployment = (
        "# Deployment baseline\n\n"
        "Before release, define the target runtime, input/output schema, latency and memory "
        "budgets, health check, human owner, rollback command, and a known-good artifact hash.\n\n"
        "This baseline is guidance, not a completed deployment runbook.\n"
    )
    observability = (
        "# Observability baseline\n\n"
        "Track operational failures, latency, input validity, data drift, output distribution, "
        "and the model/config version. Define redacted logs, alert thresholds, retention, review "
        "cadence, and an accountable responder before deployment.\n"
    )
    lock_payload = {
        **_runtime_snapshot_unvalidated(project, project_name=project_name).to_dict(),
        "kind": "daedalus-installed-environment-lock",
        "lock_scope": "captured-active-environment",
        "portable_resolver_lock": False,
        "note": (
            "This exact installed-version inventory records the creation environment. "
            "Use uv.lock or pylock.toml when a portable resolver lock is required."
        ),
    }
    return {
        PurePosixPath("pyproject.toml"): pyproject,
        PurePosixPath("environment.lock.json"): json.dumps(
            lock_payload, indent=2, sort_keys=True
        )
        + "\n",
        PurePosixPath(".gitignore"): gitignore,
        PurePosixPath("data/.gitkeep"): "",
        PurePosixPath("checkpoints/.gitkeep"): "",
        PurePosixPath("runs/.gitkeep"): "",
        PurePosixPath("tests/test_smoke.py"): smoke_test,
        PurePosixPath("cards/README.md"): card_guide,
        PurePosixPath("cards/DATASET_CARD.template.md"): dataset_card,
        PurePosixPath("cards/MODEL_CARD.template.md"): model_card,
        PurePosixPath("configs/default.json"): default_config,
        PurePosixPath("deployment/README.md"): deployment,
        PurePosixPath("observability/README.md"): observability,
    }


def _publish_new_text(path: Path, content: str) -> None:
    """Publish one completed file without a check-then-overwrite race."""

    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path, follow_symlinks=False)
    finally:
        temporary.unlink(missing_ok=True)


def _replace_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def initialize_staged_project(project: Path, *, project_name: str, template: str) -> None:
    """Populate a not-yet-published creation directory used by WorkspaceManager."""

    project = Path(project)
    if not project.is_dir() or project.is_symlink():
        raise ValueError("staged project must be a regular directory")
    for relative, content in _baseline_files(
        project, project_name=project_name, template=template
    ).items():
        destination = project.joinpath(*relative.parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(content)


def _validate_direct_project(manager: WorkspaceManager, project: str | os.PathLike[str]) -> Path:
    manager.bootstrap()
    original = Path(project)
    if original.is_symlink() or manager.projects_dir.is_symlink():
        raise PermissionError("project-standard roots cannot be symbolic links")
    try:
        resolved = manager.resolve_user_path(original, must_exist=True)
    except (FileNotFoundError, PermissionError) as exc:
        raise PermissionError(
            "project standards require a direct private project directory"
        ) from exc
    projects = manager.projects_dir.resolve(strict=True)
    if resolved.parent != projects or not resolved.is_dir() or resolved.name.startswith("."):
        raise PermissionError("project standards require a direct private project directory")
    return resolved


def _safe_standard_destination(project: Path, relative: PurePosixPath) -> Path:
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise PermissionError("standard-file destination escapes the project")
    destination = project.joinpath(*relative.parts)
    current = project
    for part in relative.parts[:-1]:
        current = current / part
        if current.is_symlink():
            raise PermissionError("standard-file parent cannot be a symbolic link")
        if current.exists() and not current.is_dir():
            raise ValueError("standard-file parent is not a directory")
    if destination.is_symlink():
        raise PermissionError("standard files cannot be symbolic links")
    if destination.exists() and not destination.is_file():
        raise ValueError("standard-file destination is not a regular file")
    return destination


def _project_identity(project: Path) -> tuple[str, str]:
    manifest = project / "project.json"
    if not manifest.is_file() or manifest.is_symlink():
        return project.name, "minimal"
    try:
        if manifest.stat().st_size > MAX_SOURCE_FILE_BYTES:
            return project.name, "minimal"
        raw = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError, TypeError, json.JSONDecodeError):
        return project.name, "minimal"
    if not isinstance(raw, dict):
        return project.name, "minimal"
    name = str(raw.get("name") or project.name).strip() or project.name
    template = str(raw.get("template") or "minimal").strip()
    if template not in {"minimal", "xor", "regression"}:
        template = "minimal"
    return name[:80], template


def _requires_python_compatible(requirement: str) -> bool | None:
    """Evaluate the simple ordered-version clauses emitted by Daedalus.

    Complex PEP 440 expressions are left visibly unverified instead of being
    guessed. Project package managers remain the authority for those forms.
    """

    current = tuple(sys.version_info[:3])
    clauses = [item.strip() for item in requirement.split(",") if item.strip()]
    if not clauses:
        return None
    for clause in clauses:
        match = re.fullmatch(r"(<=|>=|==|!=|<|>)\s*(\d+(?:\.\d+){0,2})", clause)
        if match is None:
            return None
        operator, raw_version = match.groups()
        pieces = tuple(int(item) for item in raw_version.split("."))
        target = pieces + (0,) * (3 - len(pieces))
        passed = {
            "<": current < target,
            "<=": current <= target,
            ">": current > target,
            ">=": current >= target,
            "==": current[: len(pieces)] == pieces,
            "!=": current[: len(pieces)] != pieces,
        }[operator]
        if not passed:
            return False
    return True


class ProjectStandardsService:
    """Write narrowly scoped, reviewable standards artifacts for one workspace."""

    def __init__(self, manager: WorkspaceManager) -> None:
        self.manager = manager

    def runtime_snapshot(self, project: str | os.PathLike[str]) -> dict[str, Any]:
        resolved = _validate_direct_project(self.manager, project)
        name, _template = _project_identity(resolved)
        return _runtime_snapshot_unvalidated(resolved, project_name=name).to_dict()

    def initialize_missing(self, project: str | os.PathLike[str]) -> tuple[Path, ...]:
        """Create missing baseline files, never replacing user-owned content.

        Publication is transactional for files: if one creation fails, files
        created by this call are removed. Existing files are never touched.
        """

        resolved = _validate_direct_project(self.manager, project)
        name, template = _project_identity(resolved)
        mapping = _baseline_files(resolved, project_name=name, template=template)
        manifest = resolved / "project.json"
        if not manifest.exists() and not manifest.is_symlink():
            mapping = {
                PurePosixPath("project.json"): json.dumps(
                    {
                        "schema": 1,
                        "kind": "daedalus-ai-project",
                        "id": str(uuid.uuid4()),
                        "name": name,
                        "template": template,
                        "created_utc": _utc_now(),
                        "entrypoint": "main.py",
                        "standards": {
                            "project_manifest": "pyproject.toml",
                            "environment_lock": ENVIRONMENT_LOCK,
                            "environment_snapshot": ENVIRONMENT_SNAPSHOT,
                            "run_manifest_pattern": "runs/<run-id>.manifest.json",
                        },
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                **mapping,
            }
        destinations = {
            relative: _safe_standard_destination(resolved, relative) for relative in mapping
        }
        missing = [relative for relative, path in destinations.items() if not path.exists()]
        created_files: list[Path] = []
        created_directories: list[Path] = []
        try:
            for relative in missing:
                destination = destinations[relative]
                parents: list[Path] = []
                current = destination.parent
                while current != resolved and not current.exists():
                    parents.append(current)
                    current = current.parent
                for directory in reversed(parents):
                    directory.mkdir()
                    created_directories.append(directory)
                _publish_new_text(destination, mapping[relative])
                created_files.append(destination)
        except Exception:
            for path in reversed(created_files):
                path.unlink(missing_ok=True)
            for directory in reversed(created_directories):
                try:
                    directory.rmdir()
                except OSError:
                    pass
            raise
        return tuple(created_files)

    def capture_environment(
        self,
        project: str | os.PathLike[str],
        destination: str | os.PathLike[str] | None = None,
    ) -> Path:
        resolved = _validate_direct_project(self.manager, project)
        payload = self.runtime_snapshot(resolved)
        target = resolved / ENVIRONMENT_SNAPSHOT if destination is None else Path(destination)
        if not target.is_absolute():
            target = resolved / target
        if target.is_symlink():
            raise PermissionError("environment snapshot destination cannot be a symbolic link")
        target = target.resolve(strict=False)
        try:
            relative = target.relative_to(resolved)
        except ValueError as exc:
            raise PermissionError("environment snapshot destination escapes the project") from exc
        if target.suffix.casefold() != ".json" or len(relative.parts) > 3:
            raise ValueError("environment snapshot must be a shallow JSON file in the project")
        current = resolved
        for part in relative.parts[:-1]:
            current = current / part
            if current.is_symlink():
                raise PermissionError("environment snapshot parent cannot be a symbolic link")
            current.mkdir(exist_ok=True)
        _replace_json(target, payload)
        return target

    def write_run_manifest(
        self,
        project: str | os.PathLike[str],
        run_id: str,
        payload: Mapping[str, Any],
    ) -> Path:
        resolved = _validate_direct_project(self.manager, project)
        clean_id = str(run_id).strip()
        if not _SAFE_RUN_ID.fullmatch(clean_id) or clean_id in {".", ".."}:
            raise ValueError("run_id must be a bounded filesystem-safe identifier")
        safe_payload = _redacted_json_value(payload)
        manifest = {
            "schema": RUN_MANIFEST_SCHEMA,
            "kind": "daedalus-run-manifest",
            "created_utc": _utc_now(),
            "run_id": clean_id,
            "environment": self.runtime_snapshot(resolved),
            "record": safe_payload,
        }
        encoded = json.dumps(manifest, sort_keys=True, allow_nan=False)
        if len(encoded.encode("utf-8")) > MAX_JSON_BYTES:
            raise ValueError("run manifest exceeds the bounded JSON size limit")
        runs = resolved / "runs"
        if runs.is_symlink():
            raise PermissionError("run manifest directory cannot be a symbolic link")
        runs.mkdir(exist_ok=True)
        destination = runs / f"{clean_id}.manifest.json"
        if destination.exists() or destination.is_symlink():
            raise FileExistsError("run manifests are immutable and already exist")
        _publish_new_text(destination, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        return destination


class ProjectStandardsInspector:
    """Read-only project and environment readiness assessment."""

    def __init__(self, manager: WorkspaceManager) -> None:
        self.manager = manager

    def inspect(self, project: str | os.PathLike[str]) -> ProjectStandardsReport:
        resolved = _validate_direct_project(self.manager, project)
        name, _template = _project_identity(resolved)
        findings: list[StandardFinding] = []
        self._inspect_project_manifest(resolved, findings)
        declared_dependencies = self._inspect_pyproject(resolved, findings)
        self._inspect_lock(resolved, findings)
        self._inspect_baseline(resolved, findings)
        environment = _runtime_snapshot_unvalidated(resolved, project_name=name)
        installed = {item.name for item in environment.dependencies}
        for dependency in declared_dependencies:
            if dependency not in installed:
                findings.append(
                    StandardFinding(
                        "environment.dependency-missing",
                        StandardStatus.ERROR,
                        f"Declared dependency {dependency} is not installed in the active environment.",
                        "project/pyproject.toml",
                        "Synchronize the isolated project environment from its reviewed lock.",
                    )
                )
        if not environment.isolated_environment:
            findings.append(
                StandardFinding(
                    "environment.isolation",
                    StandardStatus.WARNING,
                    "The active interpreter is not identified as an isolated environment.",
                    action="Run Daedalus and project checks from a dedicated virtual environment.",
                )
            )
        if environment.dependency_inventory_truncated:
            findings.append(
                StandardFinding(
                    "environment.dependency-limit",
                    StandardStatus.WARNING,
                    "The installed dependency inventory exceeded the bounded capture limit.",
                    action="Review the environment and remove unrelated packages.",
                )
            )
        if environment.source_inventory_truncated:
            findings.append(
                StandardFinding(
                    "project.source-limit",
                    StandardStatus.WARNING,
                    "Some source files were outside the bounded reproducibility fingerprint.",
                    action="Remove generated files or split the project before recapturing evidence.",
                )
            )
        return ProjectStandardsReport(
            name,
            _utc_now(),
            environment,
            tuple(findings),
            _tool_capabilities(environment.dependencies),
        )

    @staticmethod
    def _inspect_project_manifest(project: Path, findings: list[StandardFinding]) -> None:
        path = project / "project.json"
        try:
            if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_SOURCE_FILE_BYTES:
                raise ValueError
            raw = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict) or raw.get("schema") != 1 or not str(
                raw.get("name", "")
            ).strip():
                raise ValueError
        except (OSError, UnicodeError, ValueError, TypeError, json.JSONDecodeError):
            findings.append(
                StandardFinding(
                    "project.manifest",
                    StandardStatus.ERROR,
                    "project.json is missing or invalid.",
                    "project/project.json",
                    "Restore the Daedalus project identity before running or releasing it.",
                )
            )

    @staticmethod
    def _inspect_pyproject(
        project: Path, findings: list[StandardFinding]
    ) -> tuple[str, ...]:
        path = project / "pyproject.toml"
        try:
            if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_SOURCE_FILE_BYTES:
                raise ValueError
            raw = tomllib.loads(path.read_text(encoding="utf-8"))
            project_table = raw.get("project")
            if not isinstance(project_table, dict):
                raise ValueError
            if not str(project_table.get("name", "")).strip():
                raise ValueError
            requires_python = str(project_table.get("requires-python", "")).strip()
            if not requires_python:
                raise ValueError
            dependency_values = project_table.get("dependencies")
            if not isinstance(dependency_values, list) or not all(
                isinstance(item, str) for item in dependency_values
            ):
                raise ValueError
        except (OSError, UnicodeError, ValueError, TypeError, tomllib.TOMLDecodeError):
            findings.append(
                StandardFinding(
                    "project.pyproject",
                    StandardStatus.ERROR,
                    "pyproject.toml is missing or does not declare runtime requirements.",
                    "project/pyproject.toml",
                    "Initialize the missing project standards or repair the project table.",
                )
            )
            return ()
        compatible = _requires_python_compatible(requires_python)
        if compatible is False:
            findings.append(
                StandardFinding(
                    "environment.python-version",
                    StandardStatus.ERROR,
                    "The active Python version is outside the project's declared range.",
                    "project/pyproject.toml",
                    "Use a supported isolated interpreter or update the reviewed runtime contract.",
                )
            )
        elif compatible is None:
            findings.append(
                StandardFinding(
                    "environment.python-range",
                    StandardStatus.WARNING,
                    "The Python requirement uses syntax this bounded audit could not verify.",
                    "project/pyproject.toml",
                    "Verify the requirement with the selected package manager before running.",
                )
            )
        dependencies: list[str] = []
        for requirement in dependency_values:
            match = re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)", requirement)
            if match:
                dependencies.append(_canonical_package_name(match.group(1)))
        return tuple(dict.fromkeys(dependencies))

    @staticmethod
    def _inspect_lock(project: Path, findings: list[StandardFinding]) -> None:
        portable = [project / "uv.lock", project / "pylock.toml"]
        invalid_portable = False
        for portable_path in portable:
            if not portable_path.exists() and not portable_path.is_symlink():
                continue
            try:
                if (
                    portable_path.is_symlink()
                    or not portable_path.is_file()
                    or not 0 < portable_path.stat().st_size <= MAX_SOURCE_FILE_BYTES
                ):
                    raise ValueError
                parsed = tomllib.loads(portable_path.read_text(encoding="utf-8"))
                if not isinstance(parsed, dict) or not parsed:
                    raise ValueError
                return
            except (OSError, UnicodeError, ValueError, tomllib.TOMLDecodeError):
                invalid_portable = True
        path = project / ENVIRONMENT_LOCK
        try:
            if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_SOURCE_FILE_BYTES:
                raise ValueError
            raw = json.loads(path.read_text(encoding="utf-8"))
            dependencies = raw.get("dependencies") if isinstance(raw, dict) else None
            if (
                not isinstance(raw, dict)
                or raw.get("schema") != SNAPSHOT_SCHEMA
                or raw.get("kind") != "daedalus-installed-environment-lock"
                or not isinstance(dependencies, list)
            ):
                raise ValueError
        except (OSError, UnicodeError, ValueError, TypeError, json.JSONDecodeError):
            findings.append(
                StandardFinding(
                    "environment.lock",
                    StandardStatus.ERROR,
                    "No valid dependency resolution or installed-environment lock was found.",
                    f"project/{ENVIRONMENT_LOCK}",
                    "Initialize standards, then create a portable uv.lock when sharing machines.",
                )
            )
            return
        if invalid_portable:
            findings.append(
                StandardFinding(
                    "environment.portable-lock-invalid",
                    StandardStatus.WARNING,
                    "A portable lock file exists but failed bounded TOML validation.",
                    "project",
                    "Regenerate the lock from the reviewed pyproject instead of editing it by hand.",
                )
            )
        findings.append(
            StandardFinding(
                "environment.portable-lock",
                StandardStatus.INFO,
                "The exact creation environment is recorded, but no portable resolver lock is present.",
                f"project/{ENVIRONMENT_LOCK}",
                "For cross-machine work, create uv.lock or pylock.toml and keep it with the project.",
            )
        )

    @staticmethod
    def _inspect_baseline(project: Path, findings: list[StandardFinding]) -> None:
        expected = (
            ".gitignore",
            "tests/test_smoke.py",
            "cards/README.md",
            "cards/DATASET_CARD.template.md",
            "cards/MODEL_CARD.template.md",
            "configs/default.json",
            "deployment/README.md",
            "observability/README.md",
        )
        for relative_text in expected:
            relative = PurePosixPath(relative_text)
            path = project.joinpath(*relative.parts)
            if path.is_symlink() or not path.is_file():
                findings.append(
                    StandardFinding(
                        f"baseline.{relative.stem.casefold()}",
                        StandardStatus.WARNING,
                        f"The standard baseline file {relative.as_posix()} is missing.",
                        f"project/{relative.as_posix()}",
                        "Initialize missing standards; existing project files will not be replaced.",
                    )
                )


def _tool_capabilities(
    dependencies: tuple[DependencyVersion, ...],
) -> tuple[ToolCapability, ...]:
    versions = {item.name: item.version for item in dependencies}

    def package(
        key: str,
        label: str,
        category: str,
        purpose: str,
        *,
        distribution: str | None = None,
    ) -> ToolCapability:
        name = _canonical_package_name(distribution or key)
        version = versions.get(name)
        return ToolCapability(key, label, category, version is not None, version, True, purpose)

    def command(key: str, label: str, category: str, purpose: str) -> ToolCapability:
        available = shutil.which(key) is not None
        version = versions.get(_canonical_package_name(key))
        return ToolCapability(key, label, category, available, version, True, purpose)

    return (
        command("uv", "uv", "environment", "Create and verify portable dependency locks."),
        command("git", "Git", "source", "Version code and review changes."),
        package("pytest", "pytest", "quality", "Run automated tests."),
        package("ruff", "Ruff", "quality", "Lint and format Python source."),
        package("pip-audit", "pip-audit", "security", "Check Python dependency advisories."),
        package("torch", "PyTorch", "framework", "Train accelerated production-scale models."),
        package("tensorflow", "TensorFlow", "framework", "Build and deploy tensor models."),
        package("jax", "JAX", "framework", "Compile numerical and accelerator workloads."),
        package(
            "transformers",
            "Hugging Face Transformers",
            "framework",
            "Use pretrained transformer architectures and tokenizers.",
        ),
        package(
            "scikit-learn",
            "scikit-learn",
            "framework",
            "Build classical baselines and preprocessing pipelines.",
        ),
        package("mlflow", "MLflow", "tracking", "Compare runs and manage model artifacts."),
        package("dvc", "DVC", "data", "Version large data and pipeline dependencies."),
        command("docker", "Docker", "deployment", "Package reproducible runtime images."),
        command(
            "nvidia-smi",
            "NVIDIA GPU tooling",
            "hardware",
            "Inspect compatible NVIDIA accelerators and drivers.",
        ),
    )


def _looks_absolute_path(value: str) -> bool:
    if re.match(r"^[A-Za-z]:[\\/]", value):
        return True
    return value.startswith(("/", "\\\\"))


def _redacted_json_value(value: Any, *, key: str = "", depth: int = 0, nodes: list[int] | None = None) -> Any:
    if nodes is None:
        nodes = [0]
    nodes[0] += 1
    if nodes[0] > MAX_JSON_NODES or depth > 12:
        raise ValueError("run manifest contains too many or too-deep values")
    if any(marker in key.casefold() for marker in _SECRET_KEYS):
        return "<redacted>"
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("run manifest numbers must be finite")
        return value
    if isinstance(value, Path):
        value = str(value)
    if isinstance(value, str):
        if any(pattern.search(value) for pattern in _SECRET_PATTERNS):
            return "<redacted>"
        if _looks_absolute_path(value):
            name = (
                PureWindowsPath(value).name
                if re.match(r"^[A-Za-z]:[\\/]", value) or value.startswith("\\\\")
                else PurePosixPath(value).name
            )
            return f"<private-path>/{name}" if name else "<private-path>"
        return value[:8_000]
    if isinstance(value, Mapping):
        if len(value) > 1_000:
            raise ValueError("run manifest object contains too many fields")
        cleaned: dict[str, Any] = {}
        for child_key, child in value.items():
            raw_key = str(child_key)
            if any(pattern.search(raw_key) for pattern in _SECRET_PATTERNS):
                raise ValueError("run manifest contains a credential-like field name")
            clean_key = re.sub(r"[\x00-\x1f]", "", raw_key)[:200]
            if not clean_key:
                raise ValueError("run manifest field names cannot be empty")
            cleaned[clean_key] = _redacted_json_value(
                child, key=clean_key, depth=depth + 1, nodes=nodes
            )
        return cleaned
    if isinstance(value, (list, tuple)):
        if len(value) > 2_000:
            raise ValueError("run manifest list contains too many items")
        return [
            _redacted_json_value(child, key=key, depth=depth + 1, nodes=nodes)
            for child in value
        ]
    raise TypeError(f"run manifest value is not JSON-compatible: {type(value).__name__}")


def initialize_missing(
    manager: WorkspaceManager, project: str | os.PathLike[str]
) -> tuple[Path, ...]:
    return ProjectStandardsService(manager).initialize_missing(project)


def runtime_snapshot(
    manager: WorkspaceManager, project: str | os.PathLike[str]
) -> dict[str, Any]:
    return ProjectStandardsService(manager).runtime_snapshot(project)


def capture_environment(
    manager: WorkspaceManager,
    project: str | os.PathLike[str],
    destination: str | os.PathLike[str] | None = None,
) -> Path:
    return ProjectStandardsService(manager).capture_environment(project, destination)


def write_run_manifest(
    manager: WorkspaceManager,
    project: str | os.PathLike[str],
    run_id: str,
    payload: Mapping[str, Any],
) -> Path:
    return ProjectStandardsService(manager).write_run_manifest(project, run_id, payload)


__all__ = [
    "DependencyVersion",
    "EnvironmentSnapshot",
    "ProjectStandardsInspector",
    "ProjectStandardsReport",
    "ProjectStandardsService",
    "StandardFinding",
    "StandardStatus",
    "ToolCapability",
    "capture_environment",
    "initialize_missing",
    "runtime_snapshot",
    "write_run_manifest",
]
