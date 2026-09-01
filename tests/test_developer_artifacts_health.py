from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path, PurePosixPath

import numpy as np
import pytest

from daedalus.developer import (
    ARTIFACT_FILENAMES,
    ArtifactGenerator,
    ArtifactKind,
    DeveloperAdvisor,
    DeveloperSessionStore,
    ProjectBrief,
    ProjectEvidence,
    ProjectHealthInspector,
    RecoveryPlanner,
    TaskKind,
    validate_restore_destination,
)
from daedalus.layers import Parameter
from daedalus.workspace.checkpoints import save_checkpoint
from daedalus.workspace.run_registry import RunRegistry


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_workspace(tmp_path: Path):
    workspace = tmp_path / "workspace"
    projects = workspace / "projects"
    project = projects / "Bearing Watch"
    for directory in (
        project / "runs",
        project / "checkpoints",
        workspace / "datasets",
        workspace / "training-runs",
        workspace / "checkpoints",
        workspace / ".daedalus",
    ):
        directory.mkdir(parents=True, exist_ok=True)
    (workspace / ".daedalus-workspace.json").write_text(
        json.dumps(
            {
                "kind": "daedalus-user-workspace",
                "schema": 1,
                "id": "test-workspace-identity",
            }
        ),
        encoding="utf-8",
    )
    (project / "project.json").write_text(
        json.dumps({"schema": 1, "name": "Bearing Watch", "template": "minimal"}),
        encoding="utf-8",
    )
    brief = ProjectBrief(
        "Bearing Watch",
        "Flag likely bearing failures before an outage",
        "Maintenance coordinators",
        TaskKind.CLASSIFICATION,
        "Four finite vibration features",
        "A reviewed risk class",
        "Held-out F1 at least 0.9",
    )
    advisor = DeveloperAdvisor()
    session = advisor.start(project, brief)
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
    restore_target = tmp_path / "restores" / "Bearing Watch Restored"
    session = advisor.answer(session, "restore_destination", str(restore_target))
    return workspace, projects, project, advisor, session, restore_target


def make_verified_backup(tmp_path: Path, workspace: Path, *, schema: int = 2) -> Path:
    backup = tmp_path / "backup"
    backed_marker = backup / "workspace-current" / ".daedalus-workspace.json"
    backed_project = (
        backup / "workspace-current" / "projects" / "Bearing Watch" / "project.json"
    )
    backed_marker.parent.mkdir(parents=True)
    backed_project.parent.mkdir(parents=True)
    backed_marker.write_bytes((workspace / ".daedalus-workspace.json").read_bytes())
    backed_project.write_bytes(
        (workspace / "projects" / "Bearing Watch" / "project.json").read_bytes()
    )
    (backup / ".daedalus-backup-root.json").write_text(
        json.dumps({"kind": "daedalus-backup-root", "schema": 1}), encoding="utf-8"
    )
    inventory = []
    for path in (backed_marker, backed_project):
        digest = sha256(path)
        entry = {
            "path": path.relative_to(backup).as_posix(),
            "size": path.stat().st_size,
            "sha256": digest,
            "kind": "file",
        }
        if schema == 3:
            object_relative = Path("objects") / "sha256" / digest[:2] / digest
            content_object = backup / object_relative
            content_object.parent.mkdir(parents=True, exist_ok=True)
            content_object.write_bytes(path.read_bytes())
            entry["object_path"] = object_relative.as_posix()
        inventory.append(entry)
    (backup / "latest.json").write_text(
        json.dumps(
            {
                "schema": schema,
                "kind": "daedalus-backup-manifest",
                "finished_utc": datetime.now(UTC).isoformat(),
                "errors": [],
                "files_scanned": len(inventory),
                "inventory": inventory,
                "workspace": str(workspace),
            }
        ),
        encoding="utf-8",
    )
    return backup


def add_dataset(workspace: Path) -> None:
    dataset = workspace / "datasets" / "bearing.csv"
    dataset.write_text("x,target\n1,0\n2,1\n", encoding="utf-8")
    (workspace / "datasets" / "bearing.dataset.json").write_text(
        json.dumps({"file": dataset.name, "sha256": sha256(dataset)}), encoding="utf-8"
    )


def add_run_and_checkpoint(workspace: Path) -> None:
    registry = RunRegistry(workspace / "training-runs" / "runs.sqlite3")
    run_id = registry.create_run("Bearing Watch", "bearing", {"seed": 47})
    registry.transition(run_id, "running")
    registry.transition(run_id, "completed", metrics={"val_f1": 0.91})
    save_checkpoint(
        workspace / "checkpoints" / "Bearing Watch",
        "best",
        [Parameter(np.array([1.0, 2.0]))],
        metrics={"val_f1": 0.91},
    )


def test_artifacts_are_confined_canonical_and_never_overwritten(tmp_path) -> None:
    workspace, projects, project, advisor, session, _target = make_workspace(tmp_path)
    evidence = ProjectEvidence(workspace_ready=True, project_manifest_valid=True)
    plan = advisor.build_plan(session, evidence)
    generator = ArtifactGenerator(project, projects)

    references = generator.generate(session, plan, evidence)
    assert len(references) == len(ArtifactKind) == 8
    assert {item.relative_path for item in references} == set(ARTIFACT_FILENAMES.values())
    for reference in references:
        path = project / reference.relative_path
        assert path.is_file()
        assert reference.sha256 == sha256(path)
        assert path.resolve().parent == project.resolve()
    manifest = json.loads((project / "REPRODUCIBILITY.json").read_text(encoding="utf-8"))
    assert manifest["session_id"] == session.id
    assert manifest["seed"] == 47

    original = (project / "AI_PROJECT_SPEC.md").read_bytes()
    with pytest.raises(FileExistsError):
        generator.generate(
            session, plan, evidence, kinds=(ArtifactKind.PROJECT_SPEC,)
        )
    assert (project / "AI_PROJECT_SPEC.md").read_bytes() == original

    outside = tmp_path / "outside"
    outside.mkdir()
    with pytest.raises(PermissionError):
        ArtifactGenerator(outside, projects)


def test_generated_placeholder_artifacts_remain_draft_evidence(tmp_path) -> None:
    workspace, projects, project, advisor, completed_session, _target = make_workspace(tmp_path)
    session = advisor.start(project, completed_session.brief)
    ArtifactGenerator(project, projects).generate(session, advisor.build_plan(session))

    report = ProjectHealthInspector(project, workspace).inspect(session)

    assert report.evidence.experiment_plan_present is False
    assert report.evidence.model_card_present is False
    assert report.evidence.deployment_plan_present is False
    assert report.evidence.threat_model_present is False
    drafts = [finding for finding in report.findings if finding.code == "artifact.draft"]
    assert len(drafts) == 4


def test_health_recognizes_workspace_run_and_checkpoint_layouts(tmp_path) -> None:
    workspace, projects, project, advisor, session, _target = make_workspace(tmp_path)
    add_dataset(workspace)
    add_run_and_checkpoint(workspace)
    backup = make_verified_backup(tmp_path, workspace)
    (project / "runs" / "baseline.json").write_text(
        json.dumps({"metric": "f1", "value": 0.5}), encoding="utf-8"
    )
    (project / "DAEDALUS_EVIDENCE.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "architecture_validated": True,
                "dependency_scan_passed": True,
                "release_guard_passed": True,
            }
        ),
        encoding="utf-8",
    )
    plan = advisor.build_plan(session)
    ArtifactGenerator(project, projects).generate(session, plan)

    report = ProjectHealthInspector(
        project, workspace, backup_root=backup
    ).inspect(session)
    evidence = report.evidence
    assert evidence.workspace_ready is True
    assert evidence.project_manifest_valid is True
    assert evidence.session_inventory_complete is True
    assert evidence.run_inventory_complete is True
    assert evidence.checkpoint_inventory_complete is True
    assert evidence.restore_target_safe is True
    assert evidence.dataset_present is True
    assert evidence.dataset_integrity is True
    assert evidence.baseline_recorded is True
    assert evidence.run_completed is True
    assert evidence.heldout_metrics_present is True
    assert evidence.checkpoint_valid is True
    assert evidence.experiment_plan_present is True
    assert evidence.model_card_present is True
    assert evidence.deployment_plan_present is True
    assert evidence.threat_model_present is True
    assert evidence.secret_scan_passed is True
    assert evidence.dependency_scan_passed is True
    assert evidence.backup_current is True
    assert evidence.release_guard_passed is True


def test_health_reports_ignore_shadow_and_redacts_secret_value(tmp_path) -> None:
    workspace, _projects, project, _advisor, session, _target = make_workspace(tmp_path)
    source = tmp_path / "source"
    manager = source / "src" / "daedalus" / "workspace" / "manager.py"
    manager.parent.mkdir(parents=True)
    manager.write_text("# required source\n", encoding="utf-8")
    (source / ".gitignore").write_text("workspace/\n", encoding="utf-8")
    credential = "".join(("ghp", "_", "ABCDEFGHIJKLMNOPQRSTUVWXYZ123456"))
    (project / "unsafe.txt").write_text(f"token={credential}\n", encoding="utf-8")

    report = ProjectHealthInspector(project, workspace, source_root=source).inspect(session)
    codes = {item.code for item in report.findings}
    rendered = "\n".join(item.summary for item in report.findings)
    assert "source.ignore-shadow" in codes
    assert "project.possible-secret" in codes
    assert report.evidence.secret_scan_passed is False
    assert credential not in rendered


def test_recovery_inventory_and_safe_proposal_route_to_vault(tmp_path) -> None:
    workspace, _projects, project, _advisor, session, restore_target = make_workspace(tmp_path)
    add_run_and_checkpoint(workspace)
    backup = make_verified_backup(tmp_path, workspace)
    store = DeveloperSessionStore(workspace / ".daedalus" / "developer-bot.sqlite3")
    saved = store.save(session)
    planner = RecoveryPlanner(project, workspace, backup)

    inventory = planner.inventory(session_store=store, session_id=saved.id)
    assert inventory.ready
    assert inventory.session_revision == 1
    assert inventory.session_revision_count == 1
    assert inventory.run_count == 1
    assert inventory.completed_run_count == 1
    assert inventory.checkpoint_count == 1
    assert inventory.valid_checkpoint_count == 1
    assert inventory.backup_verified is True
    proposal = planner.propose_restore(restore_target, inventory)
    assert proposal.mode == "new-directory-only"
    assert proposal.requires_confirmation is True
    intent = planner.tool_intent(proposal)
    assert intent.tool_key.value == "vault"
    assert intent.payload["destination"] == str(restore_target.resolve(strict=False))

    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError):
        planner.propose_restore(existing, inventory)
    with pytest.raises(PermissionError):
        planner.propose_restore(workspace / "new-restore", inventory)
    with pytest.raises(ValueError):
        validate_restore_destination(Path("relative"), (workspace, backup))


def test_unverified_backup_cannot_produce_restore_proposal(tmp_path) -> None:
    workspace, _projects, project, _advisor, _session, restore_target = make_workspace(tmp_path)
    backup = tmp_path / "bad-backup"
    backup.mkdir()
    planner = RecoveryPlanner(project, workspace, backup)
    inventory = planner.inventory()

    assert inventory.backup_verified is False
    with pytest.raises(ValueError):
        planner.propose_restore(restore_target, inventory)


def test_recovery_rejects_wrong_schema_stale_or_foreign_workspace_backup(tmp_path) -> None:
    workspace, _projects, project, _advisor, _session, _target = make_workspace(tmp_path)
    backup = make_verified_backup(tmp_path, workspace)
    latest_path = backup / "latest.json"
    (project / "DAEDALUS_EVIDENCE.json").write_text(
        json.dumps({"schema_version": 1, "backup_current": True}), encoding="utf-8"
    )

    def health_backup_current() -> bool | None:
        return ProjectHealthInspector(
            project, workspace, backup_root=backup
        ).inspect().evidence.backup_current

    latest = json.loads(latest_path.read_text(encoding="utf-8"))
    latest["schema"] = 1
    latest_path.write_text(json.dumps(latest), encoding="utf-8")
    inventory = RecoveryPlanner(project, workspace, backup).inventory()
    assert inventory.backup_manifest_present is True
    assert inventory.backup_verified is False
    assert any("schema-2" in finding for finding in inventory.findings)
    assert health_backup_current() is False

    latest["schema"] = 2
    latest["finished_utc"] = (datetime.now(UTC) - timedelta(days=2)).isoformat()
    latest_path.write_text(json.dumps(latest), encoding="utf-8")
    inventory = RecoveryPlanner(project, workspace, backup).inventory()
    assert inventory.backup_verified is False
    assert any("freshness" in finding for finding in inventory.findings)
    assert health_backup_current() is False

    latest["finished_utc"] = datetime.now(UTC).isoformat()
    backed_marker = backup / "workspace-current" / ".daedalus-workspace.json"
    foreign = json.loads(backed_marker.read_text(encoding="utf-8"))
    foreign["id"] = "foreign-workspace"
    backed_marker.write_text(json.dumps(foreign), encoding="utf-8")
    marker_entry = next(
        item
        for item in latest["inventory"]
        if item["path"] == "workspace-current/.daedalus-workspace.json"
    )
    marker_entry["size"] = backed_marker.stat().st_size
    marker_entry["sha256"] = sha256(backed_marker)
    latest_path.write_text(json.dumps(latest), encoding="utf-8")
    inventory = RecoveryPlanner(project, workspace, backup).inventory()
    assert inventory.backup_verified is False
    assert any("identity" in finding for finding in inventory.findings)
    assert health_backup_current() is False


def test_recovery_requires_selected_project_in_fresh_backup(tmp_path) -> None:
    workspace, _projects, project, _advisor, _session, _target = make_workspace(tmp_path)
    backup = make_verified_backup(tmp_path, workspace)
    latest_path = backup / "latest.json"
    latest = json.loads(latest_path.read_text(encoding="utf-8"))
    latest["inventory"] = [
        item
        for item in latest["inventory"]
        if item["path"] != "workspace-current/projects/Bearing Watch/project.json"
    ]
    latest_path.write_text(json.dumps(latest), encoding="utf-8")

    inventory = RecoveryPlanner(project, workspace, backup).inventory()

    assert inventory.backup_verified is False
    assert any("selected project is absent" in finding.casefold() for finding in inventory.findings)
    assert (
        ProjectHealthInspector(project, workspace, backup_root=backup)
        .inspect()
        .evidence.backup_current
        is False
    )


def test_schema3_recovery_uses_verified_immutable_objects_not_mutable_mirror(
    tmp_path,
) -> None:
    workspace, _projects, project, _advisor, _session, _target = make_workspace(tmp_path)
    backup = make_verified_backup(tmp_path, workspace, schema=3)

    # A committed schema-3 manifest remains restorable even when the convenience
    # current mirror is later changed or removed.
    (backup / "workspace-current" / ".daedalus-workspace.json").write_text(
        '{"id":"foreign mutable mirror"}', encoding="utf-8"
    )
    (backup / "workspace-current" / "projects" / "Bearing Watch" / "project.json").unlink()

    inventory = RecoveryPlanner(project, workspace, backup).inventory()
    report = ProjectHealthInspector(project, workspace, backup_root=backup).inspect()

    assert inventory.backup_verified is True
    assert inventory.backup_file_count == 2
    assert report.evidence.backup_current is True

    manifest = json.loads((backup / "latest.json").read_text(encoding="utf-8"))
    project_entry = next(
        item
        for item in manifest["inventory"]
        if item["path"] == "workspace-current/projects/Bearing Watch/project.json"
    )
    project_object = backup / Path(*PurePosixPath(project_entry["object_path"]).parts)
    project_object.write_bytes(b"tampered immutable object")

    assert RecoveryPlanner(project, workspace, backup).inventory().backup_verified is False


def test_schema3_recovery_rejects_noncanonical_object_path_and_foreign_identity(
    tmp_path,
) -> None:
    workspace, _projects, project, _advisor, _session, _target = make_workspace(tmp_path)
    backup = make_verified_backup(tmp_path, workspace, schema=3)
    latest_path = backup / "latest.json"
    manifest = json.loads(latest_path.read_text(encoding="utf-8"))
    marker_entry = next(
        item
        for item in manifest["inventory"]
        if item["path"] == "workspace-current/.daedalus-workspace.json"
    )
    original_object_path = marker_entry["object_path"]
    marker_entry["object_path"] = "objects/sha256/00/not-the-entry-digest"
    latest_path.write_text(json.dumps(manifest), encoding="utf-8")

    assert RecoveryPlanner(project, workspace, backup).inventory().backup_verified is False

    marker_entry["object_path"] = original_object_path
    foreign_bytes = json.dumps(
        {"kind": "daedalus-user-workspace", "schema": 1, "id": "foreign-workspace"}
    ).encode("utf-8")
    foreign_digest = hashlib.sha256(foreign_bytes).hexdigest()
    foreign_relative = PurePosixPath(
        "objects", "sha256", foreign_digest[:2], foreign_digest
    )
    foreign_object = backup / Path(*foreign_relative.parts)
    foreign_object.parent.mkdir(parents=True, exist_ok=True)
    foreign_object.write_bytes(foreign_bytes)
    marker_entry["size"] = len(foreign_bytes)
    marker_entry["sha256"] = foreign_digest
    marker_entry["object_path"] = foreign_relative.as_posix()
    latest_path.write_text(json.dumps(manifest), encoding="utf-8")

    inventory = RecoveryPlanner(project, workspace, backup).inventory()
    assert inventory.backup_verified is False
    assert any("identity" in finding for finding in inventory.findings)


def test_health_and_recovery_close_read_only_registry_connections(
    tmp_path, monkeypatch
) -> None:
    workspace, _projects, project, _advisor, session, _target = make_workspace(tmp_path)
    database = workspace / "training-runs" / "runs.sqlite3"
    original_connect = sqlite3.connect
    connection = original_connect(database)
    try:
        connection.execute(
            "CREATE TABLE runs(project TEXT, status TEXT, metrics_json TEXT)"
        )
        connection.execute(
            "INSERT INTO runs VALUES (?, ?, ?)",
            (session.brief.project_name, "completed", '{"val_f1": 0.9}'),
        )
        connection.commit()
    finally:
        connection.close()

    opened: list[sqlite3.Connection] = []

    def tracked_connect(*args, **kwargs):
        tracked = original_connect(*args, **kwargs)
        opened.append(tracked)
        return tracked

    monkeypatch.setattr(sqlite3, "connect", tracked_connect)
    ProjectHealthInspector(project, workspace).inspect(session)
    RecoveryPlanner(project, workspace, tmp_path / "missing-backup").inventory()

    assert len(opened) == 2
    for tracked in opened:
        with pytest.raises(sqlite3.ProgrammingError):
            tracked.execute("SELECT 1")

    moved = database.with_suffix(".moved")
    database.rename(moved)
    moved.rename(database)
