"""Deterministic, non-overwriting project artifact generation."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Callable, Iterable

from daedalus.developer.models import (
    ARTIFACT_FILENAMES,
    ArtifactKind,
    ArtifactRef,
    BuildPlan,
    DeveloperSession,
    ProjectEvidence,
    evidence_to_dict,
)


def _answer(session: DeveloperSession, key: str, fallback: str = "Not yet recorded") -> str:
    value = session.answers.get(key)
    if value is None or value == "":
        return fallback
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, (list, tuple)):
        return ", ".join(str(item) for item in value)
    return str(value).replace("\r\n", "\n").replace("\r", "\n")


def _line(value: str) -> str:
    return " ".join(value.split())


def _bullet(label: str, value: str) -> str:
    indented = value.replace("\n", "\n  ")
    return f"- **{label}:** {indented}"


def _header(session: DeveloperSession, title: str) -> str:
    return (
        f"# {title}\n\n"
        f"Project: **{_line(session.brief.project_name)}**  \n"
        f"Task: **{session.brief.task_kind.value}**  \n"
        f"Developer session: `{session.id}`  \n"
        f"Schema: `{session.schema_version}`\n\n"
        "> Generated locally from explicit project answers. Review and edit this artifact; "
        "it is not an autonomous approval.\n"
    )


def _project_spec(session: DeveloperSession, plan: BuildPlan, _evidence: ProjectEvidence) -> str:
    constraints = "\n".join(f"- {item}" for item in session.brief.constraints) or "- None recorded"
    stages = "\n".join(
        f"- **{step.title}:** {step.objective} Current gate: `{step.gate.state.value}`."
        for step in plan.steps
    )
    return (
        _header(session, "AI Project Specification")
        + "\n## Problem contract\n\n"
        + "\n".join(
            (
                _bullet("Outcome", session.brief.outcome),
                _bullet("Users and affected people", session.brief.users),
                _bullet("Inputs at inference time", session.brief.inputs),
                _bullet("Outputs", session.brief.outputs),
                _bullet("Success metric", session.brief.success_metric),
            )
        )
        + f"\n\n## Constraints\n\n{constraints}\n\n## Build stages\n\n{stages}\n"
    )


def _dataset_card(session: DeveloperSession, _plan: BuildPlan, evidence: ProjectEvidence) -> str:
    return (
        _header(session, "Dataset Card")
        + "\n## Origin and permission\n\n"
        + "\n".join(
            (
                _bullet("Source and owner", _answer(session, "data_source")),
                _bullet("Rights confirmed", _answer(session, "data_rights_confirmed")),
                _bullet("Target definition", _answer(session, "target_definition")),
                _bullet("Split strategy", _answer(session, "split_strategy")),
            )
        )
        + "\n\n## Machine-verifiable evidence\n\n"
        + "\n".join(
            (
                _bullet("Dataset registered", str(evidence.dataset_present)),
                _bullet("Checksum/integrity verified", str(evidence.dataset_integrity)),
                _bullet("Split evidence found", str(evidence.split_documented)),
            )
        )
        + "\n\n## Required review\n\n"
        "Document collection gaps, missingness, duplicates, imbalance, leakage risks, sensitive "
        "fields, retention, and representative limitations before training.\n"
    )


def _experiment_plan(session: DeveloperSession, _plan: BuildPlan, _evidence: ProjectEvidence) -> str:
    return (
        _header(session, "Experiment Plan")
        + "\n## Baseline\n\n"
        + "\n".join(
            (
                _bullet("Baseline", _answer(session, "baseline_choice")),
                _bullet("Threshold to beat", _answer(session, "baseline_success_threshold")),
            )
        )
        + "\n\n## Model contract\n\n"
        + "\n".join(
            (
                _bullet("Architecture", _answer(session, "architecture_summary")),
                _bullet("Shape contract", _answer(session, "shape_contract")),
                _bullet("Loss", _answer(session, "loss_function")),
                _bullet("Primary metric", _answer(session, "primary_metric")),
            )
        )
        + "\n\n## Reproducible run\n\n"
        + "\n".join(
            (
                _bullet("Seed", _answer(session, "seed")),
                _bullet("Budget", _answer(session, "training_budget")),
                _bullet("Stop/reject rule", _answer(session, "stop_rule")),
            )
        )
        + "\n"
    )


def _evaluation_plan(session: DeveloperSession, _plan: BuildPlan, evidence: ProjectEvidence) -> str:
    return (
        _header(session, "Evaluation Plan")
        + "\n## Acceptance claim\n\n"
        + _bullet("Primary success metric", session.brief.success_metric)
        + "\n"
        + _bullet("Primary model metric", _answer(session, "primary_metric"))
        + "\n\n## Recorded evaluation\n\n"
        + "\n".join(
            (
                _bullet("Hold-out result", _answer(session, "holdout_result")),
                _bullet("Failure and slice analysis", _answer(session, "failure_analysis")),
                _bullet("Robustness checks", _answer(session, "robustness_checks")),
                _bullet("Completed run observed", str(evidence.run_completed)),
                _bullet("Held-out metrics observed", str(evidence.heldout_metrics_present)),
            )
        )
        + "\n\n## Guardrail\n\nDo not tune against the untouched test set. Preserve poor examples and "
        "negative results as evidence.\n"
    )


def _model_card(session: DeveloperSession, _plan: BuildPlan, evidence: ProjectEvidence) -> str:
    return (
        _header(session, "Model Card")
        + "\n## Intended use\n\n"
        + "\n".join(
            (
                _bullet("Outcome", session.brief.outcome),
                _bullet("Users and affected people", session.brief.users),
                _bullet("Input", session.brief.inputs),
                _bullet("Output", session.brief.outputs),
            )
        )
        + "\n\n## Model and evidence\n\n"
        + "\n".join(
            (
                _bullet("Architecture", _answer(session, "architecture_summary")),
                _bullet("Primary metric", _answer(session, "primary_metric")),
                _bullet("Hold-out result", _answer(session, "holdout_result")),
                _bullet("Failure analysis", _answer(session, "failure_analysis")),
                _bullet("Checkpoint integrity", str(evidence.checkpoint_valid)),
            )
        )
        + "\n\n## Limitations and human oversight\n\n"
        + _bullet("Robustness evidence", _answer(session, "robustness_checks"))
        + "\n"
        + _bullet("Monitoring", _answer(session, "monitoring_plan"))
        + "\n"
        + _bullet("Rollback", _answer(session, "rollback_plan"))
        + "\n\nAdd explicit out-of-scope uses, known weak populations, and accountable approval before release.\n"
    )


def _threat_model(session: DeveloperSession, _plan: BuildPlan, evidence: ProjectEvidence) -> str:
    return (
        _header(session, "Threat Model")
        + "\n## Assets and boundaries\n\n"
        + "\n".join(
            (
                _bullet("Private-data review", _answer(session, "privacy_review")),
                _bullet("Threats and misuse", _answer(session, "threats")),
                _bullet("Licenses reviewed", _answer(session, "licenses_reviewed")),
            )
        )
        + "\n\n## Gate evidence\n\n"
        + "\n".join(
            (
                _bullet("Secret scan", str(evidence.secret_scan_passed)),
                _bullet("Dependency scan", str(evidence.dependency_scan_passed)),
                _bullet("Verified backup", str(evidence.backup_current)),
            )
        )
        + "\n\n## Required controls\n\n"
        "Record mitigations, residual risk, access ownership, incident response, and conditions "
        "that disable the system. Never include real credentials in this document.\n"
    )


def _deployment_runbook(session: DeveloperSession, _plan: BuildPlan, _evidence: ProjectEvidence) -> str:
    return (
        _header(session, "Deployment Runbook")
        + "\n## Runtime contract\n\n"
        + "\n".join(
            (
                _bullet("Target", _answer(session, "deployment_target")),
                _bullet("Latency and memory budget", _answer(session, "latency_budget")),
                _bullet("Input", session.brief.inputs),
                _bullet("Output", session.brief.outputs),
            )
        )
        + "\n\n## Operations\n\n"
        + "\n".join(
            (
                _bullet("Monitoring and owner", _answer(session, "monitoring_plan")),
                _bullet("Rollback", _answer(session, "rollback_plan")),
                _bullet("Proposed version", _answer(session, "release_version")),
            )
        )
        + "\n\n## Drill\n\nBefore release, exercise startup, health checks, invalid input, "
        "dependency failure, rollback, and restore using the proposed artifact.\n"
    )


def _reproducibility(session: DeveloperSession, plan: BuildPlan, evidence: ProjectEvidence) -> str:
    payload = {
        "schema_version": 1,
        "kind": "daedalus-reproducibility-manifest",
        "session_id": session.id,
        "session_revision": session.revision,
        "project_name": session.brief.project_name,
        "task_kind": session.brief.task_kind.value,
        "mode": session.mode.value,
        "seed": session.answers.get("seed"),
        "data_source_description": session.answers.get("data_source"),
        "split_strategy": session.answers.get("split_strategy"),
        "architecture": session.answers.get("architecture_summary"),
        "loss": session.answers.get("loss_function"),
        "primary_metric": session.answers.get("primary_metric"),
        "training_budget": session.answers.get("training_budget"),
        "stop_rule": session.answers.get("stop_rule"),
        "gate_states": {step.stage.value: step.gate.state.value for step in plan.steps},
        "evidence": evidence_to_dict(evidence),
        "generated_from_session_utc": session.updated_utc,
    }
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


_RENDERERS: dict[
    ArtifactKind, Callable[[DeveloperSession, BuildPlan, ProjectEvidence], str]
] = {
    ArtifactKind.PROJECT_SPEC: _project_spec,
    ArtifactKind.DATASET_CARD: _dataset_card,
    ArtifactKind.EXPERIMENT_PLAN: _experiment_plan,
    ArtifactKind.EVALUATION_PLAN: _evaluation_plan,
    ArtifactKind.MODEL_CARD: _model_card,
    ArtifactKind.THREAT_MODEL: _threat_model,
    ArtifactKind.DEPLOYMENT_RUNBOOK: _deployment_runbook,
    ArtifactKind.REPRODUCIBILITY: _reproducibility,
}


class ArtifactGenerator:
    """Generate only known artifacts beneath one selected private project."""

    def __init__(self, project_root: Path, projects_root: Path) -> None:
        original_project = Path(project_root)
        original_projects = Path(projects_root)
        if original_project.is_symlink() or original_projects.is_symlink():
            raise PermissionError("artifact roots cannot be symbolic links")
        resolved_projects = original_projects.resolve(strict=True)
        resolved_project = original_project.resolve(strict=True)
        if not resolved_projects.is_dir() or not resolved_project.is_dir():
            raise ValueError("artifact roots must be existing directories")
        try:
            relative = resolved_project.relative_to(resolved_projects)
        except ValueError as exc:
            raise PermissionError("selected project escapes the private projects root") from exc
        if not relative.parts:
            raise PermissionError("select a project, not the projects collection root")
        self.projects_root = resolved_projects
        self.project_root = resolved_project

    def generate(
        self,
        session: DeveloperSession,
        plan: BuildPlan,
        evidence: ProjectEvidence | None = None,
        *,
        kinds: Iterable[ArtifactKind] | None = None,
    ) -> tuple[ArtifactRef, ...]:
        if Path(session.project_root).resolve(strict=False) != self.project_root:
            raise PermissionError("developer session belongs to a different private project")
        if plan.session_id != session.id:
            raise ValueError("build plan belongs to a different developer session")
        requested = tuple(ArtifactKind) if kinds is None else kinds
        selected = tuple(ArtifactKind(item) for item in requested)
        if len(set(selected)) != len(selected):
            raise ValueError("artifact selection contains duplicates")
        destinations = {kind: self.project_root / ARTIFACT_FILENAMES[kind] for kind in selected}
        for destination in destinations.values():
            if destination.exists() or destination.is_symlink():
                raise FileExistsError(f"artifact already exists and will not be overwritten: {destination.name}")
            if destination.parent.resolve(strict=True) != self.project_root:
                raise PermissionError("artifact destination escapes the selected project")

        observed = evidence or ProjectEvidence()
        rendered = {kind: _RENDERERS[kind](session, plan, observed) for kind in selected}
        references: list[ArtifactRef] = []
        for kind in selected:
            destination = destinations[kind]
            content = rendered[kind]
            temporary = self.project_root / f".{destination.name}.{uuid.uuid4().hex}.tmp"
            try:
                with temporary.open("x", encoding="utf-8", newline="\n") as handle:
                    handle.write(content)
                    handle.flush()
                    os.fsync(handle.fileno())
                # A hard link makes publication of the completed temporary file atomic and
                # fails when the destination appears concurrently; it never overwrites.
                os.link(temporary, destination, follow_symlinks=False)
            finally:
                temporary.unlink(missing_ok=True)
            digest = hashlib.sha256(destination.read_bytes()).hexdigest()
            references.append(
                ArtifactRef(kind, destination.name, digest, session.updated_utc)
            )
        return tuple(references)


__all__ = ["ArtifactGenerator"]
