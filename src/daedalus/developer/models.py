"""Typed, JSON-safe domain models for the offline AI Developer Bot.

The developer package deliberately contains no GUI, network, process, or model
provider integration.  Its values are plain dataclasses so the same workflow can
be driven by the desktop UI, tests, or a future command-line adapter.
"""

from __future__ import annotations

import math
import re
import uuid
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Mapping

SCHEMA_VERSION = 1
MAX_TEXT_LENGTH = 8_000
MAX_COLLECTION_ITEMS = 256


class ExperienceMode(StrEnum):
    BEGINNER = "beginner"
    BUILDER = "builder"
    EXPERT = "expert"


class TaskKind(StrEnum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    GENERATION = "generation"
    RANKING = "ranking"
    ANOMALY = "anomaly"
    CUSTOM = "custom"


class Stage(StrEnum):
    DISCOVERY = "discovery"
    RECOVERY = "recovery"
    DATA = "data"
    BASELINE = "baseline"
    ARCHITECTURE = "architecture"
    EXPERIMENT = "experiment"
    EVALUATION = "evaluation"
    DEPLOYMENT = "deployment"
    SECURITY = "security"
    RELEASE = "release"


STAGE_ORDER = tuple(Stage)
NON_WAIVABLE_STAGES = frozenset(
    {
        Stage.DISCOVERY,
        Stage.RECOVERY,
        Stage.DATA,
        Stage.SECURITY,
        Stage.RELEASE,
    }
)


class GateState(StrEnum):
    UNKNOWN = "unknown"
    BLOCKED = "blocked"
    PASSED = "passed"
    WAIVED = "waived"


class ToolKey(StrEnum):
    LEARN = "learn"
    ARCHITECTURE = "architecture"
    CALCULATOR = "calculator"
    TRAINING = "training"
    WORKSHOP = "workshop"
    EVALUATE = "evaluate"
    VAULT = "vault"
    GUARD = "guard"


class ArtifactKind(StrEnum):
    PROJECT_SPEC = "project_spec"
    DATASET_CARD = "dataset_card"
    EXPERIMENT_PLAN = "experiment_plan"
    EVALUATION_PLAN = "evaluation_plan"
    MODEL_CARD = "model_card"
    THREAT_MODEL = "threat_model"
    DEPLOYMENT_RUNBOOK = "deployment_runbook"
    REPRODUCIBILITY = "reproducibility"


ARTIFACT_FILENAMES: Mapping[ArtifactKind, str] = {
    ArtifactKind.PROJECT_SPEC: "AI_PROJECT_SPEC.md",
    ArtifactKind.DATASET_CARD: "DATASET_CARD.md",
    ArtifactKind.EXPERIMENT_PLAN: "EXPERIMENT_PLAN.md",
    ArtifactKind.EVALUATION_PLAN: "EVALUATION_PLAN.md",
    ArtifactKind.MODEL_CARD: "MODEL_CARD.md",
    ArtifactKind.THREAT_MODEL: "THREAT_MODEL.md",
    ArtifactKind.DEPLOYMENT_RUNBOOK: "DEPLOYMENT_RUNBOOK.md",
    ArtifactKind.REPRODUCIBILITY: "REPRODUCIBILITY.json",
}


_SECRET_PATTERNS = (
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----", re.IGNORECASE),
    re.compile(r"\bAKIA[A-Z0-9]{16}\b"),
    re.compile(r"\b(?:sk-[A-Za-z0-9_-]{12,}|ghp_[A-Za-z0-9]{12,})\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{12,}\b"),
)


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def validate_safe_text(value: str, *, field_name: str = "text", allow_empty: bool = False) -> str:
    """Validate bounded human text without ever echoing suspected credentials."""

    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be text")
    cleaned = value.strip()
    if not cleaned and not allow_empty:
        raise ValueError(f"{field_name} cannot be empty")
    if len(cleaned) > MAX_TEXT_LENGTH:
        raise ValueError(f"{field_name} exceeds the {MAX_TEXT_LENGTH:,}-character limit")
    if "\x00" in cleaned:
        raise ValueError(f"{field_name} contains a null character")
    if any(pattern.search(cleaned) for pattern in _SECRET_PATTERNS):
        raise ValueError("Possible credential detected; remove credentials before saving")
    return cleaned


def validate_json_value(value: Any, *, depth: int = 0) -> Any:
    """Validate a small JSON-compatible value and reject likely secret material."""

    if depth > 8:
        raise ValueError("JSON value is nested too deeply")
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("JSON numbers must be finite")
        return value
    if isinstance(value, str):
        return validate_safe_text(value, field_name="answer", allow_empty=True)
    if isinstance(value, (list, tuple)):
        if len(value) > MAX_COLLECTION_ITEMS:
            raise ValueError("JSON list contains too many items")
        return [validate_json_value(item, depth=depth + 1) for item in value]
    if isinstance(value, dict):
        if len(value) > MAX_COLLECTION_ITEMS:
            raise ValueError("JSON object contains too many fields")
        result: dict[str, Any] = {}
        for key, item in value.items():
            clean_key = validate_safe_text(str(key), field_name="JSON key")
            if any(word in clean_key.casefold() for word in ("password", "api_key", "token", "secret")):
                raise ValueError("Credential-like fields are not accepted by the developer session")
            result[clean_key] = validate_json_value(item, depth=depth + 1)
        return result
    raise TypeError(f"Unsupported JSON value type: {type(value).__name__}")


@dataclass(frozen=True, slots=True)
class ProjectBrief:
    project_name: str
    outcome: str
    users: str
    task_kind: TaskKind
    inputs: str
    outputs: str
    success_metric: str
    constraints: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("project_name", "outcome", "users", "inputs", "outputs", "success_metric"):
            object.__setattr__(self, name, validate_safe_text(getattr(self, name), field_name=name))
        if not isinstance(self.task_kind, TaskKind):
            object.__setattr__(self, "task_kind", TaskKind(self.task_kind))
        if len(self.constraints) > 32:
            raise ValueError("constraints contains too many items")
        object.__setattr__(
            self,
            "constraints",
            tuple(validate_safe_text(item, field_name="constraint") for item in self.constraints),
        )


@dataclass(frozen=True, slots=True)
class Question:
    id: str
    stage: Stage
    prompt: str
    explanation: str
    required: bool = True
    value_type: str = "text"
    example: str = ""
    choices: tuple[str, ...] = ()
    recommended_answer: Any = None


@dataclass(frozen=True, slots=True)
class ToolIntent:
    tool_key: ToolKey
    label: str
    reason: str
    payload: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ArtifactRef:
    kind: ArtifactKind
    relative_path: str
    sha256: str
    created_utc: str


@dataclass(frozen=True, slots=True)
class ProjectEvidence:
    workspace_ready: bool | None = None
    project_manifest_valid: bool | None = None
    session_inventory_complete: bool | None = None
    run_inventory_complete: bool | None = None
    checkpoint_inventory_complete: bool | None = None
    restore_target_safe: bool | None = None
    dataset_present: bool | None = None
    dataset_integrity: bool | None = None
    split_documented: bool | None = None
    baseline_recorded: bool | None = None
    architecture_validated: bool | None = None
    experiment_plan_present: bool | None = None
    run_completed: bool | None = None
    heldout_metrics_present: bool | None = None
    checkpoint_valid: bool | None = None
    model_card_present: bool | None = None
    deployment_plan_present: bool | None = None
    threat_model_present: bool | None = None
    secret_scan_passed: bool | None = None
    dependency_scan_passed: bool | None = None
    backup_current: bool | None = None
    release_guard_passed: bool | None = None


@dataclass(frozen=True, slots=True)
class GateResult:
    stage: Stage
    state: GateState
    reasons: tuple[str, ...]
    missing: tuple[str, ...] = ()
    waiver_reason: str | None = None


@dataclass(frozen=True, slots=True)
class BuildStep:
    stage: Stage
    title: str
    objective: str
    gate: GateResult
    tool_intents: tuple[ToolIntent, ...]


@dataclass(frozen=True, slots=True)
class BuildPlan:
    session_id: str
    mode: ExperienceMode
    task_kind: TaskKind
    steps: tuple[BuildStep, ...]
    generated_utc: str
    schema_version: int = SCHEMA_VERSION


@dataclass(frozen=True, slots=True)
class AdvisorTurn:
    stage: Stage
    headline: str
    summary: str
    reasons: tuple[str, ...]
    questions: tuple[Question, ...]
    gate: GateResult
    tool_intents: tuple[ToolIntent, ...]


@dataclass(frozen=True, slots=True)
class DeveloperSession:
    id: str
    project_root: str
    mode: ExperienceMode
    brief: ProjectBrief
    answers: Mapping[str, Any]
    waivers: Mapping[str, str] = field(default_factory=dict)
    artifacts: tuple[ArtifactRef, ...] = ()
    created_utc: str = field(default_factory=utc_now)
    updated_utc: str = field(default_factory=utc_now)
    revision: int = 0
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        try:
            uuid.UUID(self.id)
        except (ValueError, TypeError) as exc:
            raise ValueError("session id must be a UUID") from exc
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported developer session schema: {self.schema_version}")
        if self.revision < 0:
            raise ValueError("session revision cannot be negative")
        root = Path(self.project_root)
        if not root.is_absolute():
            raise ValueError("project_root must be absolute")
        if not isinstance(self.mode, ExperienceMode):
            object.__setattr__(self, "mode", ExperienceMode(self.mode))
        object.__setattr__(self, "answers", validate_json_value(dict(self.answers)))
        clean_waivers: dict[str, str] = {}
        for stage, reason in self.waivers.items():
            resolved_stage = Stage(stage)
            if resolved_stage in NON_WAIVABLE_STAGES:
                raise ValueError(f"the {resolved_stage.value} gate cannot carry a waiver")
            stage_key = resolved_stage.value
            clean_waivers[stage_key] = validate_safe_text(reason, field_name="waiver reason")
        object.__setattr__(self, "waivers", clean_waivers)

    @classmethod
    def create(
        cls,
        project_root: Path,
        brief: ProjectBrief,
        mode: ExperienceMode = ExperienceMode.BEGINNER,
        *,
        session_id: str | None = None,
    ) -> "DeveloperSession":
        root = project_root.resolve(strict=False)
        timestamp = utc_now()
        answers = {
            "outcome": brief.outcome,
            "users": brief.users,
            "inputs": brief.inputs,
            "outputs": brief.outputs,
            "success_metric": brief.success_metric,
        }
        return cls(
            id=session_id or str(uuid.uuid4()),
            project_root=str(root),
            mode=ExperienceMode(mode),
            brief=brief,
            answers=answers,
            created_utc=timestamp,
            updated_utc=timestamp,
        )

    def with_answer(self, question_id: str, value: Any) -> "DeveloperSession":
        updated = dict(self.answers)
        updated[question_id] = validate_json_value(value)
        return replace(self, answers=updated, updated_utc=utc_now())

    def with_waiver(self, stage: Stage, reason: str) -> "DeveloperSession":
        resolved_stage = Stage(stage)
        if resolved_stage in NON_WAIVABLE_STAGES:
            raise ValueError(f"the {resolved_stage.value} gate cannot be waived")
        waivers = dict(self.waivers)
        waivers[resolved_stage.value] = validate_safe_text(reason, field_name="waiver reason")
        return replace(self, waivers=waivers, updated_utc=utc_now())

    def with_artifacts(self, refs: tuple[ArtifactRef, ...]) -> "DeveloperSession":
        by_kind = {item.kind: item for item in self.artifacts}
        by_kind.update({item.kind: item for item in refs})
        ordered = tuple(by_kind[kind] for kind in ArtifactKind if kind in by_kind)
        return replace(self, artifacts=ordered, updated_utc=utc_now())


def evidence_to_dict(evidence: ProjectEvidence) -> dict[str, Any]:
    return asdict(evidence)


__all__ = [
    "ARTIFACT_FILENAMES",
    "NON_WAIVABLE_STAGES",
    "SCHEMA_VERSION",
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
    "evidence_to_dict",
    "utc_now",
    "validate_json_value",
    "validate_safe_text",
]
