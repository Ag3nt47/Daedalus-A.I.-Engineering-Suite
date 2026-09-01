"""Read-only project diagnostics and automatic project log folder coverage."""

from __future__ import annotations

from pathlib import Path

import pytest

from daedalus.developer.diagnostics import (
    DiagnosticSeverity,
    ProjectDiagnosticsScanner,
)
from daedalus.workspace.manager import WorkspaceManager
from daedalus.workspace.run_registry import RunRegistry


@pytest.fixture()
def manager(tmp_path: Path) -> WorkspaceManager:
    source = tmp_path / "source"
    source.mkdir()
    instance = WorkspaceManager(
        source,
        tmp_path / "private-workspace",
        tmp_path / "backup",
    )
    instance.bootstrap()
    return instance


def test_new_and_existing_projects_receive_safe_log_folders(
    manager: WorkspaceManager,
) -> None:
    project = manager.create_project("logged-project")
    guide = project / "logs" / "README.txt"
    assert guide.is_file()
    assert "live or on demand" in guide.read_text(encoding="utf-8")

    legacy = manager.projects_dir / "legacy-project"
    legacy.mkdir()
    (legacy / "project.json").write_text(
        '{"schema": 1, "name": "legacy-project"}', encoding="utf-8"
    )
    logs = manager.ensure_project_logs(legacy)
    assert logs == legacy / "logs"
    assert (logs / "README.txt").is_file()


def test_scanner_finds_syntax_log_run_and_artifact_problems_without_echoing_logs(
    manager: WorkspaceManager,
) -> None:
    project = manager.create_project("troubled-project")
    (project / "broken.py").write_text("def broken(:\n    pass\n", encoding="utf-8")
    private_value = "sk-do-not-repeat-this-private-value"
    (project / "logs" / "training.log").write_text(
        f"Traceback while training {private_value}\n", encoding="utf-8"
    )

    registry = RunRegistry(manager.runs_dir / "runs.sqlite3")
    run_id = registry.create_run(project.name, "missing-dataset", {"epochs": 1})
    registry.transition(run_id, "running")
    registry.transition(
        run_id,
        "failed",
        checkpoint=str(manager.checkpoints_dir / project.name / "missing.json"),
        error="redacted training failure",
    )

    report = ProjectDiagnosticsScanner(manager).scan(project)
    codes = {item.code for item in report.findings}
    assert {
        "python.syntax",
        "log.error",
        "run.failed",
        "dataset.missing",
        "checkpoint.missing",
    } <= codes
    assert report.error_count >= 4
    assert report.log_files_scanned >= 1
    assert private_value not in report.format_text()
    assert "never imported or executed" in report.format_text()
    assert any(item.severity == DiagnosticSeverity.ERROR for item in report.findings)


def test_scanner_accepts_a_clean_starter_and_rejects_outside_paths(
    manager: WorkspaceManager,
    tmp_path: Path,
) -> None:
    project = manager.create_project("clean-project")
    report = ProjectDiagnosticsScanner(manager).scan(project)
    assert report.ok
    assert not report.truncated
    assert report.files_scanned >= 1

    outside = tmp_path / "outside"
    outside.mkdir()
    with pytest.raises(PermissionError, match="direct private project"):
        ProjectDiagnosticsScanner(manager).scan(outside)


def test_scanner_reports_when_a_log_exceeds_its_bound(manager: WorkspaceManager) -> None:
    project = manager.create_project("bounded-project")
    (project / "logs" / "large.log").write_text("error\n" * 100, encoding="utf-8")
    report = ProjectDiagnosticsScanner(
        manager,
        maximum_file_bytes=32,
        maximum_total_bytes=128,
    ).scan(project)
    assert report.truncated
    assert any(item.code == "file.limit" for item in report.findings)


def test_change_token_notices_project_log_updates(manager: WorkspaceManager) -> None:
    project = manager.create_project("live-watch-project")
    scanner = ProjectDiagnosticsScanner(manager)
    before = scanner.change_token(project)
    (project / "logs" / "live.log").write_text("training started\n", encoding="utf-8")
    after_create = scanner.change_token(project)
    (project / "logs" / "live.log").write_text("training failed\n", encoding="utf-8")
    after_update = scanner.change_token(project)
    assert before != after_create
    assert after_create != after_update
