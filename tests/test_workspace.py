import json
import tomllib
from pathlib import Path

import pytest

import daedalus.services.project_standards as standards_module
import daedalus.workspace.manager as manager_module
from daedalus.workspace.manager import WorkspaceManager, safe_project_name


def manager(tmp_path: Path) -> WorkspaceManager:
    source = tmp_path / "source"
    source.mkdir()
    return WorkspaceManager(source, tmp_path / "private", tmp_path / "backup")


def test_only_environment_manager_requires_production_backup_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    explicit = manager(tmp_path)
    monkeypatch.setenv("DAEDALUS_WORKSPACE_ROOT", str(tmp_path / "environment-workspace"))
    monkeypatch.setenv("DAEDALUS_BACKUP_ROOT", str(tmp_path / "environment-backup"))

    environment = WorkspaceManager.from_environment()

    assert explicit.require_backup_volume_preflight is False
    assert environment.require_backup_volume_preflight is True


def test_bootstrap_creates_private_tree(tmp_path: Path) -> None:
    workspace = manager(tmp_path)
    workspace.bootstrap()
    assert workspace.marker_path.is_file()
    assert workspace.projects_dir.is_dir()
    assert workspace.checkpoints_dir.is_dir()


def test_workspace_must_be_separate_from_source(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    workspace = WorkspaceManager(source, source / "workspace", tmp_path / "backup")
    with pytest.raises(ValueError, match="outside"):
        workspace.bootstrap()


def test_resolve_rejects_escape(tmp_path: Path) -> None:
    workspace = manager(tmp_path)
    workspace.bootstrap()
    with pytest.raises(PermissionError):
        workspace.resolve_user_path(tmp_path / "somewhere-else")


def test_create_project_uses_external_private_root(tmp_path: Path) -> None:
    workspace = manager(tmp_path)
    project = workspace.create_project("My XOR", "xor")
    assert project == workspace.projects_dir / "My XOR"
    assert (project / "main.py").is_file()
    assert (project / ".gitignore").is_file()
    assert (project / "tests" / "test_smoke.py").is_file()
    assert (project / "cards" / "MODEL_CARD.template.md").is_file()
    assert (project / "configs" / "default.json").is_file()
    assert (project / "deployment" / "README.md").is_file()
    assert (project / "observability" / "README.md").is_file()
    pyproject = tomllib.loads((project / "pyproject.toml").read_text(encoding="utf-8"))
    assert pyproject["project"]["name"] == "my-xor"
    assert "daedalus-ai-suite>=0.1,<0.2" in pyproject["project"]["dependencies"]
    environment_lock = json.loads(
        (project / "environment.lock.json").read_text(encoding="utf-8")
    )
    assert environment_lock["kind"] == "daedalus-installed-environment-lock"
    assert environment_lock["portable_resolver_lock"] is False
    manifest = json.loads((project / "project.json").read_text(encoding="utf-8"))
    assert manifest["kind"] == "daedalus-ai-project"
    assert manifest["entrypoint"] == "main.py"
    assert workspace.list_projects() == [project]


def test_create_project_validates_before_touching_workspace(tmp_path: Path) -> None:
    workspace = manager(tmp_path)
    with pytest.raises(ValueError, match="Unknown project template"):
        workspace.create_project("invalid", "does-not-exist")
    assert not workspace.workspace_root.exists()


def test_create_project_failure_leaves_no_partial_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = manager(tmp_path)
    workspace.bootstrap()

    def fail_metadata(*_args, **_kwargs):
        raise OSError("simulated metadata failure")

    monkeypatch.setattr(manager_module, "_atomic_json", fail_metadata)
    with pytest.raises(OSError, match="simulated"):
        workspace.create_project("atomic", "minimal")
    assert not (workspace.projects_dir / "atomic").exists()
    assert not list(workspace.projects_dir.glob(".atomic.creating-*"))


def test_standard_baseline_failure_leaves_no_partial_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = manager(tmp_path)

    def fail_baseline(*_args, **_kwargs):
        raise OSError("simulated standards failure")

    monkeypatch.setattr(standards_module, "initialize_staged_project", fail_baseline)
    with pytest.raises(OSError, match="simulated standards failure"):
        workspace.create_project("standards-atomic", "minimal")
    assert not (workspace.projects_dir / "standards-atomic").exists()
    assert not list(workspace.projects_dir.glob(".standards-atomic.creating-*"))


@pytest.mark.parametrize("raw, expected", [(" Hello/World ", "Hello-World"), ("A.I.", "A.I")])
def test_safe_project_name(raw: str, expected: str) -> None:
    assert safe_project_name(raw) == expected
