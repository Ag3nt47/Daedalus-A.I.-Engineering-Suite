import hashlib
from pathlib import Path

import pytest

from daedalus.engine.weight_tools import WEIGHT_TOOL_SPECS, WeightToolError
from daedalus.services.sandbox import SandboxPolicyError, inspect_source
from daedalus.services.weight_sandbox import (
    MAX_SANDBOX_SOURCE_BYTES,
    SANDBOX_RELATIVE_DIRECTORY,
    WeightSandboxService,
    sandbox_template,
    sandbox_template_sha256,
)
from daedalus.workspace.manager import WorkspaceManager


@pytest.fixture()
def private_project(tmp_path: Path) -> tuple[WorkspaceManager, Path]:
    source = tmp_path / "source"
    source.mkdir()
    manager = WorkspaceManager(source, tmp_path / "workspace", tmp_path / "backup")
    return manager, manager.create_project("weight-sandbox")


def _make_symlink(link: Path, target: Path, *, directory: bool) -> None:
    try:
        link.symlink_to(target, target_is_directory=directory)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"symbolic links are unavailable in this environment: {exc}")


def test_all_starter_templates_are_parseable_warning_free_and_stable() -> None:
    for spec in WEIGHT_TOOL_SPECS:
        source = sandbox_template(spec.key)
        _tree, warnings = inspect_source(source)

        assert warnings == []
        assert f"TOOL_ID = {spec.key!r}" in source
        assert "if __name__ == \"__main__\":" in source
        assert sandbox_template_sha256(spec.key) == hashlib.sha256(
            source.encode("utf-8")
        ).hexdigest()
        assert len(sandbox_template_sha256(spec.key)) == 64

    with pytest.raises(WeightToolError, match="Unknown Weight Lab sandbox tool"):
        sandbox_template("../../escape")


def test_load_is_read_only_until_explicit_create_and_create_never_overwrites(
    private_project: tuple[WorkspaceManager, Path],
) -> None:
    manager, project = private_project
    service = WeightSandboxService(manager)

    draft = service.load(project, "logic_compiler")
    assert not draft.exists
    assert draft.source == sandbox_template("logic_compiler")
    assert draft.template_sha256 == sandbox_template_sha256("logic_compiler")
    assert draft.path == project / SANDBOX_RELATIVE_DIRECTORY / "logic_compiler.py"
    assert not (project / "experiments").exists()

    path = service.create(project, "logic_compiler")
    assert path.read_text(encoding="utf-8") == draft.source
    with pytest.raises(FileExistsError):
        service.create(project, "logic_compiler", "print('replacement')\n")
    assert path.read_text(encoding="utf-8") == draft.source


def test_save_is_atomic_and_run_uses_the_constrained_subprocess(
    private_project: tuple[WorkspaceManager, Path],
) -> None:
    manager, project = private_project
    service = WeightSandboxService(manager)
    path = service.create(project, "matrix_inverter")
    updated = "print('weight-sandbox-ok')\n"

    assert service.save(project, "matrix_inverter", updated) == path
    assert service.load(project, "matrix_inverter").source == updated
    assert not list(path.parent.glob(f".{path.name}.daedalus-*.tmp"))

    result = service.run(project, "matrix_inverter")
    assert result.ok
    assert result.stdout.strip() == "weight-sandbox-ok"
    assert result.policy_warnings == ()


def test_every_builtin_starter_executes_in_its_private_project(
    private_project: tuple[WorkspaceManager, Path],
) -> None:
    manager, project = private_project
    service = WeightSandboxService(manager, timeout_seconds=10)

    for spec in WEIGHT_TOOL_SPECS:
        service.create(project, spec.key)
        result = service.run(project, spec.key)
        assert result.ok, f"{spec.key}: {result.stderr}"
        assert spec.key in result.stdout
        assert result.policy_warnings == ()


def test_invalid_save_does_not_replace_the_existing_draft(
    private_project: tuple[WorkspaceManager, Path],
) -> None:
    manager, project = private_project
    service = WeightSandboxService(manager)
    path = service.create(project, "meta_weight", "print('original')\n")

    with pytest.raises(SandboxPolicyError, match="Syntax error"):
        service.save(project, "meta_weight", "def broken(:\n")
    assert path.read_text(encoding="utf-8") == "print('original')\n"

    with pytest.raises(ValueError, match="512 KiB"):
        service.save(project, "meta_weight", "#" * (MAX_SANDBOX_SOURCE_BYTES + 1))
    assert path.read_text(encoding="utf-8") == "print('original')\n"


def test_restricted_import_is_saved_as_visible_code_but_blocked_on_run(
    private_project: tuple[WorkspaceManager, Path],
) -> None:
    manager, project = private_project
    service = WeightSandboxService(manager)
    service.create(project, "recurrent_kernel", "import os\nprint(os.getcwd())\n")

    with pytest.raises(SandboxPolicyError, match="blocked import: os"):
        service.run(project, "recurrent_kernel")


def test_service_rejects_outside_nested_and_tool_key_traversal(
    private_project: tuple[WorkspaceManager, Path], tmp_path: Path
) -> None:
    manager, project = private_project
    service = WeightSandboxService(manager)
    outside = tmp_path / "outside-project"
    outside.mkdir()
    nested = project / "nested"
    nested.mkdir()

    with pytest.raises(PermissionError):
        service.load(outside, "meta_weight")
    with pytest.raises(PermissionError, match="direct private project"):
        service.load(nested, "meta_weight")
    original_main = (project / "main.py").read_text(encoding="utf-8")
    with pytest.raises(WeightToolError, match="Unknown Weight Lab sandbox tool"):
        service.create(project, "../../main", "print('escaped')\n")
    assert (project / "main.py").read_text(encoding="utf-8") == original_main
    assert not (project / "experiments").exists()


def test_service_rejects_a_symlinked_project(
    private_project: tuple[WorkspaceManager, Path],
) -> None:
    manager, project = private_project
    service = WeightSandboxService(manager)
    linked_project = manager.projects_dir / "linked-project"
    _make_symlink(linked_project, project, directory=True)

    with pytest.raises(PermissionError, match="symbolic links"):
        service.load(linked_project, "meta_weight")


def test_service_rejects_symlinked_sandbox_directories(
    private_project: tuple[WorkspaceManager, Path], tmp_path: Path
) -> None:
    manager, project = private_project
    service = WeightSandboxService(manager)
    outside = tmp_path / "outside-experiments"
    outside.mkdir()
    _make_symlink(project / "experiments", outside, directory=True)

    with pytest.raises(PermissionError, match="directories cannot be symbolic links"):
        service.create(project, "logic_compiler")
    assert not (outside / "weight_lab" / "logic_compiler.py").exists()


def test_service_rejects_a_symlinked_draft_file(
    private_project: tuple[WorkspaceManager, Path], tmp_path: Path
) -> None:
    manager, project = private_project
    service = WeightSandboxService(manager)
    directory = project / SANDBOX_RELATIVE_DIRECTORY
    directory.mkdir(parents=True)
    outside = tmp_path / "outside.py"
    outside.write_text("print('outside')\n", encoding="utf-8")
    draft = directory / "uncertainty_sampler.py"
    _make_symlink(draft, outside, directory=False)

    with pytest.raises(PermissionError, match="files cannot be symbolic links"):
        service.load(project, "uncertainty_sampler")
    with pytest.raises(PermissionError, match="files must be regular non-symlink files"):
        service.save(project, "uncertainty_sampler", "print('changed')\n")
    assert outside.read_text(encoding="utf-8") == "print('outside')\n"
