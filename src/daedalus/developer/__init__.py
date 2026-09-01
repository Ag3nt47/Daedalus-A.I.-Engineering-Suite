"""Offline, deterministic AI Developer Bot domain API."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_MODULE_EXPORTS = {
    ".advisor": ("DeveloperAdvisor",),
    ".artifacts": ("ArtifactGenerator",),
    ".diagnostics": (
        "DiagnosticFinding",
        "DiagnosticSeverity",
        "ProjectDiagnosticReport",
        "ProjectDiagnosticsScanner",
    ),
    ".health": (
        "FindingSeverity",
        "HealthFinding",
        "HealthReport",
        "ProjectHealthInspector",
    ),
    ".models": (
        "ARTIFACT_FILENAMES",
        "NON_WAIVABLE_STAGES",
        "STAGE_ORDER",
        "AdvisorTurn",
        "ArtifactKind",
        "ArtifactRef",
        "BuildPlan",
        "BuildStep",
        "DeveloperSession",
        "ExperienceMode",
        "GateResult",
        "GateState",
        "ProjectBrief",
        "ProjectEvidence",
        "Question",
        "Stage",
        "TaskKind",
        "ToolIntent",
        "ToolKey",
    ),
    ".progress": (
        "GateProgressSummary",
        "ProjectProgressInspector",
        "ProjectProgressSnapshot",
        "calculate_project_progress",
    ),
    ".recovery": (
        "DEFAULT_MAX_BACKUP_AGE",
        "RecoveryInventory",
        "RecoveryPlanner",
        "RecoveryProposal",
        "validate_restore_destination",
    ),
    ".store": (
        "ConcurrentSessionUpdate",
        "DeveloperSessionStore",
        "SessionCatalogEntry",
        "SessionCatalogState",
        "SessionIntegrityError",
        "SessionStoreError",
        "StoredRevision",
        "session_from_dict",
        "session_from_json",
        "session_to_dict",
        "session_to_json",
    ),
}
_EXPORT_MODULE = {
    name: module_name
    for module_name, names in _MODULE_EXPORTS.items()
    for name in names
}

__all__ = [
    "ARTIFACT_FILENAMES",
    "NON_WAIVABLE_STAGES",
    "STAGE_ORDER",
    "AdvisorTurn",
    "ArtifactGenerator",
    "ArtifactKind",
    "ArtifactRef",
    "BuildPlan",
    "BuildStep",
    "ConcurrentSessionUpdate",
    "DeveloperAdvisor",
    "DeveloperSession",
    "DeveloperSessionStore",
    "DiagnosticFinding",
    "DiagnosticSeverity",
    "DEFAULT_MAX_BACKUP_AGE",
    "ExperienceMode",
    "FindingSeverity",
    "GateResult",
    "GateProgressSummary",
    "GateState",
    "HealthFinding",
    "HealthReport",
    "ProjectBrief",
    "ProjectEvidence",
    "ProjectDiagnosticReport",
    "ProjectDiagnosticsScanner",
    "ProjectHealthInspector",
    "ProjectProgressInspector",
    "ProjectProgressSnapshot",
    "Question",
    "RecoveryInventory",
    "RecoveryPlanner",
    "RecoveryProposal",
    "SessionCatalogEntry",
    "SessionCatalogState",
    "SessionIntegrityError",
    "SessionStoreError",
    "Stage",
    "StoredRevision",
    "TaskKind",
    "ToolIntent",
    "ToolKey",
    "calculate_project_progress",
    "session_from_dict",
    "session_from_json",
    "session_to_dict",
    "session_to_json",
    "validate_restore_destination",
]


def __getattr__(name: str) -> Any:
    """Import only the Developer Bot domain area requested by the caller."""

    module_name = _EXPORT_MODULE.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
