"""Backup, sandbox, release, training, and project-standard services."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_MODULE_EXPORTS = {
    ".project_standards": (
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
    ),
    ".weight_sandbox": (
        "SandboxDraft",
        "WeightSandboxService",
        "sandbox_template",
        "sandbox_template_sha256",
    ),
}
_EXPORT_MODULE = {
    name: module_name
    for module_name, names in _MODULE_EXPORTS.items()
    for name in names
}

__all__ = [
    "DependencyVersion",
    "EnvironmentSnapshot",
    "ProjectStandardsInspector",
    "ProjectStandardsReport",
    "ProjectStandardsService",
    "StandardFinding",
    "StandardStatus",
    "SandboxDraft",
    "ToolCapability",
    "WeightSandboxService",
    "capture_environment",
    "initialize_missing",
    "runtime_snapshot",
    "sandbox_template",
    "sandbox_template_sha256",
    "write_run_manifest",
]


def __getattr__(name: str) -> Any:
    """Load only the service family that owns the requested public symbol."""

    module_name = _EXPORT_MODULE.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
