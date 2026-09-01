from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import pytest

from daedalus.developer import (
    STAGE_ORDER,
    DeveloperAdvisor,
    DeveloperSessionStore,
    GateState,
    HealthReport,
    ProjectBrief,
    ProjectEvidence,
    ProjectProgressInspector,
    Stage,
    TaskKind,
    calculate_project_progress,
)
from daedalus.workspace.manager import WorkspaceManager


@pytest.fixture()
def manager(tmp_path: Path) -> WorkspaceManager:
    source = tmp_path / "source"
    source.mkdir()
    value = WorkspaceManager(source, tmp_path / "workspace", tmp_path / "backup")
    value.bootstrap()
    return value


def _brief(name: str) -> ProjectBrief:
    return ProjectBrief(
        name,
        "Flag a bounded teaching pattern before it causes harm",
        "Local learners and accountable reviewers",
        TaskKind.CLASSIFICATION,
        "Two finite numeric features",
        "One reviewed class label",
        "Held-out F1 of at least 0.80",
    )


def _answer_all(advisor: DeveloperAdvisor, session):
    for question in advisor.questions(session):
        if question.id in session.answers:
            continue
        if question.value_type == "bool":
            value = True
        elif question.value_type == "int":
            value = 47
        else:
            value = question.recommended_answer or f"Recorded {question.id} evidence"
        session = advisor.answer(session, question.id, value)
    return session


def _store(manager: WorkspaceManager) -> DeveloperSessionStore:
    return DeveloperSessionStore(
        manager.settings_dir / "developer-sessions.sqlite3",
        allowed_root=manager.workspace_root,
    )


def test_progress_calculation_uses_all_ten_canonical_gates(tmp_path: Path) -> None:
    advisor = DeveloperAdvisor()
    project = tmp_path / "project"
    session = _answer_all(advisor, advisor.start(project, _brief("project")))
    evidence = ProjectEvidence(**{item.name: True for item in fields(ProjectEvidence)})

    snapshot = calculate_project_progress(
        session,
        HealthReport(evidence, (), files_checked=7, bytes_checked=1234),
        advisor=advisor,
    )

    assert snapshot.percent == 100
    assert snapshot.completed_gates == snapshot.total_gates == len(STAGE_ORDER) == 10
    assert snapshot.waived_gates == 0
    assert snapshot.next_gate is None
    assert snapshot.next_gate_title is None
    assert snapshot.complete
    assert tuple(summary.stage for summary in snapshot.gate_summaries) == STAGE_ORDER
    assert all(summary.state == GateState.PASSED for summary in snapshot.gates)
    assert snapshot.files_checked == 7
    assert snapshot.bytes_checked == 1234


def test_inspector_selects_matching_session_and_counts_explicit_waiver(
    manager: WorkspaceManager,
) -> None:
    advisor = DeveloperAdvisor()
    store = _store(manager)
    project = manager.create_project("active-project")
    other = manager.create_project("newer-unrelated-project")
    session = advisor.start(project, _brief(project.name))
    session = advisor.waive(session, Stage.BASELINE, "Use a reviewed prototype waiver for now")
    persisted = store.save(session)
    store.save(advisor.start(other, _brief(other.name)))

    snapshot = ProjectProgressInspector(manager, advisor=advisor).inspect(project)

    assert snapshot.project_root == project.resolve()
    assert snapshot.project_name == project.name
    assert snapshot.session_id == persisted.id
    assert snapshot.session_revision == persisted.revision
    assert snapshot.completed_gates == 2
    assert snapshot.waived_gates == 1
    assert snapshot.percent == 20
    assert snapshot.next_gate == Stage.RECOVERY
    assert snapshot.next_gate_title == "Inventory and protect recoverable work"
    baseline = snapshot.gates[STAGE_ORDER.index(Stage.BASELINE)]
    assert baseline.state == GateState.WAIVED
    assert baseline.complete
    assert baseline.missing


def test_project_without_healthy_session_has_zero_unknown_progress(
    manager: WorkspaceManager,
) -> None:
    project = manager.create_project("unstarted-project")
    database = manager.settings_dir / "developer-sessions.sqlite3"
    assert not database.exists()

    snapshot = ProjectProgressInspector(manager).inspect(project)

    assert snapshot.session_id is None
    assert snapshot.session_revision is None
    assert snapshot.percent == 0
    assert snapshot.completed_gates == 0
    assert snapshot.total_gates == len(STAGE_ORDER)
    assert snapshot.next_gate == Stage.DISCOVERY
    assert not snapshot.complete
    assert all(summary.state == GateState.UNKNOWN for summary in snapshot.gates)
    assert snapshot.evidence.workspace_ready is True
    assert snapshot.evidence.project_manifest_valid is True
    assert not database.exists()


def test_inspector_rejects_non_project_paths(manager: WorkspaceManager, tmp_path: Path) -> None:
    inspector = ProjectProgressInspector(manager, _store(manager))
    outside = tmp_path / "outside"
    outside.mkdir()
    nested = manager.create_project("parent-project") / "nested"
    nested.mkdir()

    with pytest.raises(PermissionError, match="escapes"):
        inspector.inspect(outside)
    with pytest.raises(PermissionError, match="direct"):
        inspector.inspect(nested)
    with pytest.raises(PermissionError, match="direct"):
        inspector.inspect(manager.projects_dir)
