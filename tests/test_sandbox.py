from pathlib import Path

import pytest

from daedalus.services.sandbox import SandboxPolicyError, SandboxRunner
from daedalus.workspace.manager import WorkspaceManager


def runner(tmp_path: Path) -> tuple[SandboxRunner, Path]:
    source = tmp_path / "source"
    source.mkdir()
    manager = WorkspaceManager(source, tmp_path / "private", tmp_path / "backup")
    project = manager.create_project("test")
    # These tests exercise policy, output, and audit behavior rather than the
    # timeout boundary. Use the production default so a loaded Windows host
    # cannot turn interpreter startup latency into an unrelated failure.
    return SandboxRunner(manager), project


def test_runner_executes_simple_project_file(tmp_path: Path) -> None:
    sandbox, project = runner(tmp_path)
    script = project / "hello.py"
    script.write_text("print(6 * 7)", encoding="utf-8")
    result = sandbox.run_file(script)
    assert result.ok
    assert result.stdout.strip() == "42"
    log = next((project / "logs").glob("daedalus-*.log"))
    text = log.read_text(encoding="utf-8")
    assert "event=sandbox_completed" in text
    assert "return_code=0" in text
    assert result.stdout not in text


def test_runner_rejects_system_import_by_default(tmp_path: Path) -> None:
    sandbox, project = runner(tmp_path)
    script = project / "unsafe.py"
    script.write_text("import subprocess\n", encoding="utf-8")
    with pytest.raises(SandboxPolicyError, match="blocked import"):
        sandbox.run_file(script)


def test_runner_rejects_paths_outside_projects(tmp_path: Path) -> None:
    sandbox, _ = runner(tmp_path)
    outside = tmp_path / "private" / "datasets" / "x.py"
    outside.write_text("print('x')", encoding="utf-8")
    with pytest.raises(PermissionError):
        sandbox.run_file(outside)


def test_failed_run_writes_redacted_project_event(tmp_path: Path) -> None:
    sandbox, project = runner(tmp_path)
    script = project / "failure.py"
    private_value = "private-runtime-detail"
    script.write_text(f"raise RuntimeError('{private_value}')\n", encoding="utf-8")
    result = sandbox.run_file(script)
    assert not result.ok
    assert private_value in result.stderr

    log = next((project / "logs").glob("daedalus-*.log"))
    text = log.read_text(encoding="utf-8")
    assert "level=ERROR" in text
    assert "event=sandbox_failed" in text
    assert private_value not in text
