"""Project setup, environment evidence, and optional capability coverage."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import daedalus.services.project_standards as standards_module
from daedalus.services.project_standards import (
    DependencyVersion,
    ProjectStandardsInspector,
    ProjectStandardsService,
    StandardStatus,
)
from daedalus.workspace.manager import WorkspaceManager


@pytest.fixture()
def manager(tmp_path: Path) -> WorkspaceManager:
    source = tmp_path / "source"
    source.mkdir()
    instance = WorkspaceManager(source, tmp_path / "private", tmp_path / "backup")
    instance.bootstrap()
    return instance


def test_new_project_has_inspectable_professional_baseline(
    manager: WorkspaceManager,
) -> None:
    project = manager.create_project("standards-ready")
    report = ProjectStandardsInspector(manager).inspect(project)

    assert report.ok
    assert report.error_count == 0
    assert report.environment.project_name == "standards-ready"
    assert report.environment.python_version
    assert report.environment.daedalus_version
    assert report.environment.entrypoint_sha256
    assert report.environment.pyproject_sha256
    assert report.environment.dependencies == tuple(
        sorted(report.environment.dependencies, key=lambda item: item.name)
    )
    assert any(item.key == "torch" and item.optional for item in report.tools)
    assert "No project code was imported or executed" in report.format_text()
    assert not any(item.status == StandardStatus.ERROR for item in report.findings)


def test_inspection_is_bounded_to_direct_projects_and_never_imports_project_code(
    manager: WorkspaceManager,
    tmp_path: Path,
) -> None:
    project = manager.create_project("no-import")
    marker = project / "imported.txt"
    (project / "dangerous_module.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('bad')\n",
        encoding="utf-8",
    )
    ProjectStandardsInspector(manager).inspect(project)
    assert not marker.exists()

    outside = tmp_path / "outside"
    outside.mkdir()
    with pytest.raises(PermissionError, match="direct private project"):
        ProjectStandardsInspector(manager).inspect(outside)


def test_legacy_initialization_is_missing_only_and_idempotent(
    manager: WorkspaceManager,
) -> None:
    legacy = manager.projects_dir / "legacy"
    legacy.mkdir()
    (legacy / "main.py").write_text("print('legacy')\n", encoding="utf-8")
    sentinel = "# user-owned project manifest\n"
    (legacy / "pyproject.toml").write_text(sentinel, encoding="utf-8")

    service = ProjectStandardsService(manager)
    created = service.initialize_missing(legacy)
    assert (legacy / "project.json") in created
    assert (legacy / "environment.lock.json") in created
    assert (legacy / "cards" / "MODEL_CARD.template.md") in created
    assert (legacy / "configs" / "default.json") in created
    assert (legacy / "deployment" / "README.md") in created
    assert (legacy / "observability" / "README.md") in created
    assert (legacy / "pyproject.toml").read_text(encoding="utf-8") == sentinel
    assert service.initialize_missing(legacy) == ()


def test_legacy_initialization_rolls_back_files_created_by_failed_call(
    manager: WorkspaceManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    legacy = manager.projects_dir / "rollback"
    legacy.mkdir()
    (legacy / "main.py").write_text("print('safe')\n", encoding="utf-8")
    original = standards_module._publish_new_text
    calls = 0

    def fail_second(path: Path, content: str) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated baseline failure")
        original(path, content)

    monkeypatch.setattr(standards_module, "_publish_new_text", fail_second)
    with pytest.raises(OSError, match="simulated baseline failure"):
        ProjectStandardsService(manager).initialize_missing(legacy)
    assert not (legacy / "project.json").exists()
    assert not (legacy / "pyproject.toml").exists()


def test_capture_environment_is_redacted_stable_and_project_scoped(
    manager: WorkspaceManager,
    tmp_path: Path,
) -> None:
    project = manager.create_project("snapshot")
    service = ProjectStandardsService(manager)
    before = service.runtime_snapshot(project)
    destination = service.capture_environment(project)
    captured = json.loads(destination.read_text(encoding="utf-8"))
    after = service.runtime_snapshot(project)

    assert captured["kind"] == "daedalus-environment-snapshot"
    assert captured["dependencies"]
    assert before["source_sha256"] == after["source_sha256"]
    encoded = json.dumps(captured)
    assert str(manager.workspace_root) not in encoded
    assert str(Path.home()) not in encoded

    with pytest.raises(PermissionError, match="escapes"):
        service.capture_environment(project, tmp_path / "outside.json")


def test_run_manifest_is_immutable_bounded_and_redacts_paths_and_credentials(
    manager: WorkspaceManager,
) -> None:
    project = manager.create_project("run-evidence")
    service = ProjectStandardsService(manager)
    secret = "ghp_" + "A" * 36
    manifest_path = service.write_run_manifest(
        project,
        "run-001",
        {
            "checkpoint": project / "checkpoints" / "model.json",
            "metric": 0.25,
            "token": secret,
        },
    )
    raw = manifest_path.read_text(encoding="utf-8")
    manifest = json.loads(raw)
    assert manifest["kind"] == "daedalus-run-manifest"
    assert manifest["record"]["token"] == "<redacted>"
    assert manifest["record"]["checkpoint"] == "<private-path>/model.json"
    assert secret not in raw
    assert str(project) not in raw
    with pytest.raises(FileExistsError, match="immutable"):
        service.write_run_manifest(project, "run-001", {"metric": 0.1})
    with pytest.raises(ValueError, match="run_id"):
        service.write_run_manifest(project, "../escape", {})


def test_missing_optional_tools_are_capabilities_not_failures(
    manager: WorkspaceManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = manager.create_project("optional-tools")
    monkeypatch.setattr(
        standards_module,
        "_installed_distributions",
        lambda: ((DependencyVersion("numpy", "2.0.0"),), False),
    )
    monkeypatch.setattr(standards_module.shutil, "which", lambda _name: None)
    report = ProjectStandardsInspector(manager).inspect(project)
    unavailable = [tool for tool in report.tools if not tool.available]
    assert unavailable
    assert all(tool.optional for tool in unavailable)
    assert report.ok


def test_invalid_metadata_is_actionable_without_echoing_contents(
    manager: WorkspaceManager,
) -> None:
    project = manager.create_project("invalid-standard")
    private_text = "do-not-echo-private-project-text"
    (project / "pyproject.toml").write_text(private_text, encoding="utf-8")
    (project / "environment.lock.json").write_text("{}", encoding="utf-8")
    report = ProjectStandardsInspector(manager).inspect(project)
    codes = {item.code for item in report.findings}
    assert {"project.pyproject", "environment.lock"} <= codes
    assert report.error_count == 2
    assert private_text not in report.format_text()
