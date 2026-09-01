"""Read-only, evidence-backed progress snapshots for one private project.

The desktop shell can render this module's immutable snapshot without knowing
how developer sessions, health evidence, or gates are persisted.  Progress is
deliberately coarse: a gate contributes only when the canonical advisor marks
it passed or explicitly waived.  Missing evidence therefore cannot inflate the
percentage.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from daedalus.developer.advisor import DeveloperAdvisor
from daedalus.developer.health import HealthFinding, HealthReport, ProjectHealthInspector
from daedalus.developer.models import (
    STAGE_ORDER,
    DeveloperSession,
    GateState,
    ProjectEvidence,
    Stage,
)
from daedalus.developer.store import (
    DeveloperSessionStore,
    SessionCatalogState,
)
from daedalus.workspace.manager import WorkspaceManager

_COMPLETE_STATES = frozenset({GateState.PASSED, GateState.WAIVED})
_NO_SESSION_MESSAGE = "Start a guided developer session for this project."


@dataclass(frozen=True, slots=True)
class GateProgressSummary:
    """Small UI-safe view of one canonical advisor gate."""

    stage: Stage
    title: str
    state: GateState
    reasons: tuple[str, ...]
    missing: tuple[str, ...]
    waiver_reason: str | None = None

    @property
    def complete(self) -> bool:
        return self.state in _COMPLETE_STATES


@dataclass(frozen=True, slots=True)
class ProjectProgressSnapshot:
    """Immutable progress and diagnostic evidence for an active project."""

    project_root: Path
    project_name: str
    session_id: str | None
    session_revision: int | None
    percent: int
    completed_gates: int
    waived_gates: int
    total_gates: int
    next_gate: Stage | None
    next_gate_title: str | None
    gate_summaries: tuple[GateProgressSummary, ...]
    evidence: ProjectEvidence
    health_findings: tuple[HealthFinding, ...]
    files_checked: int
    bytes_checked: int

    @property
    def complete(self) -> bool:
        return self.total_gates > 0 and self.completed_gates == self.total_gates

    @property
    def gates(self) -> tuple[GateProgressSummary, ...]:
        """Concise alias for consumers that display the gate matrix."""

        return self.gate_summaries

    @property
    def completed(self) -> int:
        """Compatibility-friendly concise count for presentation layers."""

        return self.completed_gates

    @property
    def total(self) -> int:
        """Compatibility-friendly concise total for presentation layers."""

        return self.total_gates

    @property
    def findings(self) -> tuple[HealthFinding, ...]:
        """Expose the underlying read-only health findings without copying them."""

        return self.health_findings


def calculate_project_progress(
    session: DeveloperSession,
    health_report: HealthReport,
    *,
    advisor: DeveloperAdvisor | None = None,
) -> ProjectProgressSnapshot:
    """Calculate bounded progress from a session and already-observed evidence."""

    resolved_advisor = advisor or DeveloperAdvisor()
    plan = resolved_advisor.build_plan(session, health_report.evidence)
    summaries = tuple(
        GateProgressSummary(
            stage=step.stage,
            title=step.title,
            state=step.gate.state,
            reasons=step.gate.reasons,
            missing=step.gate.missing,
            waiver_reason=step.gate.waiver_reason,
        )
        for step in plan.steps
    )
    if tuple(summary.stage for summary in summaries) != STAGE_ORDER:
        raise RuntimeError("developer advisor returned an unexpected gate order")

    completed = sum(summary.complete for summary in summaries)
    waived = sum(summary.state == GateState.WAIVED for summary in summaries)
    total = len(summaries)
    percent = min(100, max(0, (completed * 100) // total)) if total else 0
    next_summary = next((summary for summary in summaries if not summary.complete), None)
    return ProjectProgressSnapshot(
        project_root=Path(session.project_root).resolve(strict=False),
        project_name=session.brief.project_name,
        session_id=session.id,
        session_revision=session.revision,
        percent=percent,
        completed_gates=completed,
        waived_gates=waived,
        total_gates=total,
        next_gate=next_summary.stage if next_summary else None,
        next_gate_title=next_summary.title if next_summary else None,
        gate_summaries=summaries,
        evidence=health_report.evidence,
        health_findings=health_report.findings,
        files_checked=health_report.files_checked,
        bytes_checked=health_report.bytes_checked,
    )


def _empty_snapshot(project_root: Path, health_report: HealthReport) -> ProjectProgressSnapshot:
    summaries = tuple(
        GateProgressSummary(
            stage=stage,
            title=stage.value.replace("_", " ").title(),
            state=GateState.UNKNOWN,
            reasons=(_NO_SESSION_MESSAGE,),
            missing=(_NO_SESSION_MESSAGE,),
        )
        for stage in STAGE_ORDER
    )
    first = summaries[0] if summaries else None
    return ProjectProgressSnapshot(
        project_root=project_root,
        project_name=project_root.name,
        session_id=None,
        session_revision=None,
        percent=0,
        completed_gates=0,
        waived_gates=0,
        total_gates=len(summaries),
        next_gate=first.stage if first else None,
        next_gate_title=first.title if first else None,
        gate_summaries=summaries,
        evidence=health_report.evidence,
        health_findings=health_report.findings,
        files_checked=health_report.files_checked,
        bytes_checked=health_report.bytes_checked,
    )


class ProjectProgressInspector:
    """Load the newest healthy matching session and inspect its evidence."""

    def __init__(
        self,
        manager: WorkspaceManager,
        session_store: DeveloperSessionStore | None = None,
        *,
        advisor: DeveloperAdvisor | None = None,
    ) -> None:
        self.manager = manager
        self.advisor = advisor or DeveloperAdvisor()

        workspace = Path(manager.workspace_root).resolve(strict=False)
        if session_store is None:
            database = Path(manager.settings_dir) / "developer-sessions.sqlite3"
            # Progress inspection must not create a session database merely by
            # viewing a project. The normal Developer Bot owns initialization.
            session_store = (
                DeveloperSessionStore(database, allowed_root=workspace)
                if database.is_file()
                else None
            )
        self.session_store = session_store
        if session_store is None:
            return

        database = Path(session_store.database).resolve(strict=False)
        try:
            database.relative_to(workspace)
        except ValueError as exc:
            raise PermissionError("developer session store escapes the private workspace") from exc

    def _resolve_project(self, project_root: Path) -> Path:
        original = Path(project_root)
        projects_root = Path(self.manager.projects_dir)
        if original.is_symlink() or projects_root.is_symlink():
            raise PermissionError("progress-inspection roots cannot be symbolic links")
        resolved_projects = projects_root.resolve(strict=True)
        resolved_project = original.resolve(strict=True)
        if not resolved_project.is_dir():
            raise ValueError("progress inspection requires an existing project directory")
        try:
            relative = resolved_project.relative_to(resolved_projects)
        except ValueError as exc:
            raise PermissionError("selected project escapes the private projects root") from exc
        if len(relative.parts) != 1 or relative.name.startswith("."):
            raise PermissionError("select a direct private project directory")
        return resolved_project

    def _latest_session(self, project_root: Path) -> DeveloperSession | None:
        if self.session_store is None:
            return None
        for entry in self.session_store.list_catalog():
            session = entry.session
            if entry.state != SessionCatalogState.HEALTHY or session is None:
                continue
            if Path(session.project_root).resolve(strict=False) == project_root:
                return session
        return None

    def inspect(self, project_root: Path) -> ProjectProgressSnapshot:
        """Return one bounded snapshot without modifying project or artifact state."""

        project = self._resolve_project(project_root)
        session = self._latest_session(project)
        health_report = ProjectHealthInspector(
            project,
            Path(self.manager.workspace_root),
            source_root=Path(self.manager.source_root),
            backup_root=Path(self.manager.backup_root),
        ).inspect(session)
        if session is None:
            return _empty_snapshot(project, health_report)
        return calculate_project_progress(session, health_report, advisor=self.advisor)


__all__ = [
    "GateProgressSummary",
    "ProjectProgressInspector",
    "ProjectProgressSnapshot",
    "calculate_project_progress",
]
