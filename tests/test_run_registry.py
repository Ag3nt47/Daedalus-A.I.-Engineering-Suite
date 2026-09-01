from pathlib import Path

import pytest

from daedalus.workspace.run_registry import RunRegistry


def test_run_registry_lifecycle_and_audit_events(tmp_path: Path) -> None:
    registry = RunRegistry(tmp_path / "runs.sqlite3")
    run_id = registry.create_run("xor", "builtin:xor", {"epochs": 100})
    assert registry.get(run_id).status == "queued"
    registry.transition(run_id, "running")
    registry.record_metrics(run_id, {"loss": 0.2})
    registry.transition(run_id, "completed", metrics={"loss": 0.01}, checkpoint="xor.json")
    record = registry.get(run_id)
    assert record.status == "completed"
    assert record.metrics == {"loss": 0.01}
    assert [event["event"] for event in registry.events(run_id)] == [
        "created",
        "transition",
        "metrics",
        "transition",
    ]


def test_terminal_run_cannot_restart(tmp_path: Path) -> None:
    registry = RunRegistry(tmp_path / "runs.sqlite3")
    run_id = registry.create_run("xor", "builtin:xor", {})
    registry.transition(run_id, "cancelled")
    with pytest.raises(ValueError, match="Invalid run transition"):
        registry.transition(run_id, "running")


def test_failed_run_requires_error_summary(tmp_path: Path) -> None:
    registry = RunRegistry(tmp_path / "runs.sqlite3")
    run_id = registry.create_run("xor", "builtin:xor", {})
    with pytest.raises(ValueError, match="requires"):
        registry.transition(run_id, "failed")


def test_registry_releases_database_handles_after_every_operation(tmp_path: Path) -> None:
    database = tmp_path / "runs.sqlite3"
    registry = RunRegistry(database)
    run_id = registry.create_run("xor", "builtin:xor", {})
    registry.get(run_id)
    registry.list_runs()
    registry.events(run_id)

    # Windows refuses this rename while any SQLite connection is still open.
    moved = database.with_suffix(".moved")
    database.rename(moved)
    moved.rename(database)
