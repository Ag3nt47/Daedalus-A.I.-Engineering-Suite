import json
import os
import socket
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import psutil
import pytest

import daedalus.services.backup as backup_module
from daedalus.services.backup import (
    BackupLockError,
    BackupService,
    BackupVolumePreflightError,
)
from daedalus.workspace.manager import WorkspaceManager


def test_backup_copies_source_and_private_workspace_without_deleting(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "app.py").write_text("print('hello')", encoding="utf-8")
    manager = WorkspaceManager(source, tmp_path / "private", tmp_path / "backup")
    manager.bootstrap()
    (manager.projects_dir / "note.txt").write_text("private", encoding="utf-8")

    service = BackupService(manager)
    first = service.run()
    assert first.ok
    assert (manager.backup_root / "source-current" / "app.py").is_file()
    assert (manager.backup_root / "workspace-current" / "projects" / "note.txt").is_file()

    (source / "app.py").unlink()
    second = service.run()
    assert second.ok
    assert (manager.backup_root / "source-current" / "app.py").is_file()


def test_production_preflight_failure_precedes_workspace_and_destination_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "app.py").write_text("safe", encoding="utf-8")
    workspace = tmp_path / "private"
    backup = tmp_path / "backup"
    backup.mkdir()
    (backup / ".daedalus-backup-root.json").write_text(
        json.dumps(
            {
                "kind": "daedalus-backup-root",
                "schema": 1,
                "created_utc": datetime.now(UTC).isoformat(),
            }
        ),
        encoding="utf-8",
    )
    destination_health = backup / "backup-health.json"
    destination_health.write_text('{"state":"existing"}', encoding="utf-8")
    original_destination_health = destination_health.read_bytes()
    manager = WorkspaceManager(
        source,
        workspace,
        backup,
        require_backup_volume_preflight=True,
    )
    service = BackupService(manager)

    def reject_volume() -> None:
        assert not workspace.exists()
        assert not (backup / "source-current").exists()
        raise BackupVolumePreflightError("backup-volume-dirty")

    monkeypatch.setattr(service, "_run_required_volume_preflight", reject_volume)

    with pytest.raises(BackupVolumePreflightError, match="backup-volume-dirty"):
        service.run()

    assert not manager.marker_path.exists()
    assert not manager.projects_dir.exists()
    assert not (backup / "source-current").exists()
    assert destination_health.read_bytes() == original_destination_health
    health = json.loads(service.health_path.read_text(encoding="utf-8"))
    assert health["state"] == "failed"
    assert health["failure_code"] == "backup-volume-dirty"


def test_successful_production_preflight_runs_before_bootstrap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "app.py").write_text("safe", encoding="utf-8")
    manager = WorkspaceManager(
        source,
        tmp_path / "private",
        tmp_path / "backup",
        require_backup_volume_preflight=True,
    )
    service = BackupService(manager)
    calls: list[str] = []

    def accept_volume() -> None:
        assert not manager.workspace_root.exists()
        assert not manager.backup_root.exists()
        calls.append("preflight")

    monkeypatch.setattr(service, "_run_required_volume_preflight", accept_volume)

    result = service.run()

    assert result.ok
    assert calls == ["preflight"]
    assert manager.marker_path.is_file()
    assert (manager.backup_root / "latest.json").is_file()


@pytest.mark.skipif(os.name != "nt", reason="Windows volume preflight subprocess")
def test_production_volume_preflight_invokes_read_only_launcher(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    (source / "tools").mkdir(parents=True)
    (source / "tools" / "backup.ps1").write_text("# mocked launcher\n", encoding="utf-8")
    manager = WorkspaceManager(
        source,
        tmp_path / "private",
        tmp_path / "backup",
        require_backup_volume_preflight=True,
    )
    service = BackupService(manager)
    expected_drive = service.backup_root.drive.rstrip(":\\/").upper()
    observed: dict[str, object] = {}

    def complete_preflight(command, **kwargs):
        observed["command"] = command
        observed["kwargs"] = kwargs
        return backup_module.subprocess.CompletedProcess(
            command,
            0,
            json.dumps(
                {
                    "kind": "daedalus-backup-preflight",
                    "schema": 1,
                    "state": "ready",
                    "drive": expected_drive,
                    "free_bytes": 10 * 1024**3,
                    "required_free_bytes": 5 * 1024**3,
                }
            ),
            "",
        )

    monkeypatch.setattr(backup_module.subprocess, "run", complete_preflight)

    service._run_required_volume_preflight()

    command = observed["command"]
    kwargs = observed["kwargs"]
    assert isinstance(command, list)
    assert command[-1] == "-PreflightOnly"
    assert str(source / "tools" / "backup.ps1") in command
    assert isinstance(kwargs, dict)
    assert kwargs["env"]["DAEDALUS_BACKUP_ROOT"] == str(service.backup_root)
    assert kwargs["env"]["DAEDALUS_WORKSPACE_ROOT"] == str(manager.workspace_root)
    assert kwargs["capture_output"] is True
    assert kwargs["timeout"] == 45


@pytest.mark.skipif(os.name != "nt", reason="Windows volume preflight subprocess")
def test_production_volume_preflight_maps_sanitized_failure_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    (source / "tools").mkdir(parents=True)
    (source / "tools" / "backup.ps1").write_text("# mocked launcher\n", encoding="utf-8")
    manager = WorkspaceManager(
        source,
        tmp_path / "private",
        tmp_path / "backup",
        require_backup_volume_preflight=True,
    )
    service = BackupService(manager)

    def reject_preflight(command, **_kwargs):
        return backup_module.subprocess.CompletedProcess(command, 22, "", "private details")

    monkeypatch.setattr(backup_module.subprocess, "run", reject_preflight)

    with pytest.raises(BackupVolumePreflightError) as captured:
        service._run_required_volume_preflight()

    assert captured.value.failure_code == "backup-volume-dirty"
    assert "private details" not in str(captured.value)


def test_backup_excludes_git_and_repairs_same_size_future_dated_corruption(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    original = "print('safe')\n"
    (source / "app.py").write_text(original, encoding="utf-8")
    git = source / ".git"
    git.mkdir()
    (git / "config").write_text("credential = private", encoding="utf-8")
    manager = WorkspaceManager(source, tmp_path / "private", tmp_path / "backup")
    service = BackupService(manager)
    assert service.run().ok

    assert not (manager.backup_root / "source-current" / ".git").exists()
    target = manager.backup_root / "source-current" / "app.py"
    target.write_text("X" * len(original), encoding="utf-8")
    future = datetime(2035, 1, 1, tzinfo=UTC).timestamp()
    os.utime(target, (future, future))

    repaired = service.run()
    assert repaired.ok
    assert target.read_text(encoding="utf-8") == original
    assert any(entry.path == "source-current/app.py" for entry in repaired.inventory)


def test_manifest_inventory_verifies_and_detects_tampering(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "app.py").write_text("content", encoding="utf-8")
    manager = WorkspaceManager(source, tmp_path / "private", tmp_path / "backup")
    service = BackupService(manager)
    result = service.run()
    assert result.ok
    assert len(result.inventory) == result.files_scanned

    manifest = json.loads((manager.backup_root / "latest.json").read_text(encoding="utf-8"))
    assert manifest["kind"] == "daedalus-backup-manifest"
    assert manifest["schema"] == 3
    assert manifest["inventory"]
    assert all(len(entry["sha256"]) == 64 for entry in manifest["inventory"])
    assert all(
        entry["object_path"]
        == f"objects/sha256/{entry['sha256'][:2]}/{entry['sha256']}"
        for entry in manifest["inventory"]
    )
    destination_health = manager.backup_root / "backup-health.json"
    destination_health_before_verification = destination_health.read_bytes()
    assert service.verify().ok
    assert destination_health.read_bytes() == destination_health_before_verification

    logical_target = manager.backup_root / "source-current" / "app.py"
    logical_target.write_text("mutable mirror damage", encoding="utf-8")
    assert service.verify().ok
    assert destination_health.read_bytes() == destination_health_before_verification

    app_entry = next(entry for entry in manifest["inventory"] if entry["path"].endswith("app.py"))
    target = manager.backup_root / Path(*app_entry["object_path"].split("/"))
    target.write_text("tampered object", encoding="utf-8")
    verification = service.verify()
    assert not verification.ok
    assert "source-current/app.py" in verification.mismatched
    assert destination_health.read_bytes() == destination_health_before_verification
    status = service.latest_status()
    assert status is not None
    assert status["health"]["state"] == "failed"  # type: ignore[index]
    assert status["health"]["failure_code"] == "verification-failed"  # type: ignore[index]


def test_verify_rejects_an_empty_inventory(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "app.py").write_text("content", encoding="utf-8")
    manager = WorkspaceManager(source, tmp_path / "private", tmp_path / "backup")
    service = BackupService(manager)
    assert service.run().ok

    latest = manager.backup_root / "latest.json"
    payload = json.loads(latest.read_text(encoding="utf-8"))
    payload["inventory"] = []
    payload["files_scanned"] = 0
    latest.write_text(json.dumps(payload), encoding="utf-8")

    verification = service.verify()
    assert not verification.ok
    assert verification.files_checked == 0
    assert "manifest inventory is empty" in verification.errors


def test_nested_project_names_and_non_database_suffixes_are_backed_up(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "build").mkdir()
    (source / "build" / "root-build-output.txt").write_text("excluded", encoding="utf-8")
    (source / "package" / "build").mkdir(parents=True)
    (source / "package" / "build" / "builder.py").write_text("valuable", encoding="utf-8")
    manager = WorkspaceManager(source, tmp_path / "private", tmp_path / "backup")
    manager.bootstrap()
    project = manager.projects_dir / "demo"
    for directory in ("env", "build", "dist", ".tmp"):
        nested = project / directory
        nested.mkdir(parents=True)
        (nested / "valuable.txt").write_text(directory, encoding="utf-8")
    for name in ("notes-journal", "policy-wal", "diagram-shm"):
        (project / name).write_text("ordinary project data", encoding="utf-8")

    result = BackupService(manager).run()
    assert result.ok
    mirror = manager.backup_root / "workspace-current" / "projects" / "demo"
    assert all((mirror / directory / "valuable.txt").is_file() for directory in ("env", "build", "dist", ".tmp"))
    assert all((mirror / name).is_file() for name in ("notes-journal", "policy-wal", "diagram-shm"))
    assert result.skipped_sqlite_sidecars == 0
    assert not (manager.backup_root / "source-current" / "build").exists()
    assert (
        manager.backup_root / "source-current" / "package" / "build" / "builder.py"
    ).is_file()


def test_directory_enumeration_errors_make_the_backup_fail(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "app.py").write_text("content", encoding="utf-8")
    manager = WorkspaceManager(source, tmp_path / "private", tmp_path / "backup")
    manager.bootstrap()
    original_walk = backup_module.os.walk

    def guarded_walk(top, *args, **kwargs):
        if Path(top).resolve() == manager.workspace_root.resolve():
            kwargs["onerror"](
                PermissionError(13, "permission denied", str(manager.projects_dir / "blocked"))
            )
            return iter(())
        return original_walk(top, *args, **kwargs)

    monkeypatch.setattr(backup_module.os, "walk", guarded_walk)
    result = BackupService(manager).run()
    assert not result.ok
    assert any("directory enumeration failed" in error for error in result.errors)
    assert not (manager.backup_root / "latest.json").exists()


def test_live_wal_database_uses_online_snapshot_and_skips_sidecars(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    manager = WorkspaceManager(source, tmp_path / "private", tmp_path / "backup")
    manager.bootstrap()
    database = manager.runs_dir / "runs.sqlite3"
    connection = sqlite3.connect(database)
    try:
        assert connection.execute("PRAGMA journal_mode=WAL").fetchone()[0] == "wal"
        connection.execute("PRAGMA wal_autocheckpoint=0")
        connection.execute("CREATE TABLE events(value TEXT NOT NULL)")
        connection.execute("INSERT INTO events VALUES ('committed in wal')")
        connection.commit()
        assert database.with_name(database.name + "-wal").is_file()

        result = BackupService(manager).run()
        assert result.ok
        assert result.skipped_sqlite_sidecars >= 1
        copied_database = manager.backup_root / "workspace-current" / "training-runs" / database.name
        assert copied_database.is_file()
        assert not copied_database.with_name(copied_database.name + "-wal").exists()
        assert not copied_database.with_name(copied_database.name + "-shm").exists()
        with sqlite3.connect(copied_database) as restored:
            assert restored.execute("SELECT value FROM events").fetchone()[0] == "committed in wal"
        entry = next(item for item in result.inventory if item.path.endswith(database.name))
        assert entry.kind == "sqlite"
    finally:
        connection.close()


def test_structured_lock_reclaims_only_demonstrably_stale_owner(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "app.py").write_text("safe", encoding="utf-8")
    manager = WorkspaceManager(source, tmp_path / "private", tmp_path / "backup")
    manager.bootstrap()
    service = BackupService(manager)
    service._ensure_root()
    lock = manager.backup_root / ".daedalus-backup.lock"
    lock.write_text(
        json.dumps(
            {
                "kind": "daedalus-backup-lock",
                "schema": 1,
                "pid": 2**30,
                "host": socket.gethostname(),
                "process_started_utc": "2000-01-01T00:00:00+00:00",
                "lock_created_utc": "2000-01-01T00:00:00+00:00",
                "token": "stale",
            }
        ),
        encoding="utf-8",
    )

    assert service.run().ok
    assert not lock.exists()
    assert list(manager.backup_root.glob(".daedalus-backup.lock.quarantine-*"))

    active_token = service._acquire_lock()
    try:
        active_payload = json.loads(lock.read_text(encoding="utf-8"))
        assert active_payload["pid"] == os.getpid()
        assert active_payload["host"] == socket.gethostname()
        observed = datetime.fromisoformat(active_payload["process_started_utc"]).timestamp()
        assert abs(observed - psutil.Process().create_time()) <= 2
        with pytest.raises(BackupLockError, match="already running"):
            service.run()
    finally:
        service._release_lock(active_token)


def test_unverifiable_lock_is_preserved(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    manager = WorkspaceManager(source, tmp_path / "private", tmp_path / "backup")
    manager.bootstrap()
    service = BackupService(manager)
    service._ensure_root()
    lock = manager.backup_root / ".daedalus-backup.lock"
    lock.write_text("legacy-or-corrupt-lock", encoding="utf-8")

    with pytest.raises(BackupLockError, match="not safe to reclaim"):
        service.run()
    assert lock.read_text(encoding="utf-8") == "legacy-or-corrupt-lock"


def test_ancient_preboot_malformed_lock_is_quarantined_not_deleted(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "app.py").write_text("safe", encoding="utf-8")
    manager = WorkspaceManager(source, tmp_path / "private", tmp_path / "backup")
    manager.bootstrap()
    service = BackupService(manager)
    service._ensure_root()
    lock = manager.backup_root / ".daedalus-backup.lock"
    debris = b'{"kind":"daedalus-backup-lock"'
    lock.write_bytes(debris)
    ancient = psutil.boot_time() - (2 * 24 * 60 * 60)
    os.utime(lock, (ancient, ancient))

    assert service.run().ok
    assert not lock.exists()
    quarantined = list(manager.backup_root.glob(".daedalus-backup.lock.quarantine-*"))
    assert len(quarantined) == 1
    assert quarantined[0].read_bytes() == debris


def test_backup_refuses_unmarked_nonempty_destination(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    backup = tmp_path / "backup"
    backup.mkdir()
    (backup / "someone-elses-file.txt").write_text("keep", encoding="utf-8")
    manager = WorkspaceManager(source, tmp_path / "private", backup)
    service = BackupService(manager)
    try:
        service.run()
    except RuntimeError as exc:
        assert "valid Daedalus backup marker" in str(exc)
    else:
        raise AssertionError("Unmarked destination should have been rejected")
    status = service.latest_status()
    assert status is not None
    assert status["health"]["state"] == "failed"  # type: ignore[index]
    assert status["health"]["failure_code"] == "backup-exception"  # type: ignore[index]


def test_backup_refuses_a_malformed_marker_in_a_nonempty_destination(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    backup = tmp_path / "backup"
    backup.mkdir()
    (backup / ".daedalus-backup-root.json").write_text("{}", encoding="utf-8")
    (backup / "someone-elses-file.txt").write_text("keep", encoding="utf-8")
    manager = WorkspaceManager(source, tmp_path / "private", backup)

    with pytest.raises(RuntimeError, match="valid Daedalus backup marker"):
        BackupService(manager).run()
    assert (backup / "someone-elses-file.txt").read_text(encoding="utf-8") == "keep"


def test_restore_isolated_from_all_custody_roots(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    manager = WorkspaceManager(source, tmp_path / "private", tmp_path / "backup")
    manager.bootstrap()
    (manager.projects_dir / "recovery.txt").write_text("recover me", encoding="utf-8")
    service = BackupService(manager)
    assert service.run().ok

    restored = service.restore_workspace(tmp_path / "restored-private")
    assert (restored / "projects" / "recovery.txt").read_text(encoding="utf-8") == "recover me"

    for unsafe in (
        manager.source_root / "restore-child",
        manager.workspace_root / "restore-child",
        manager.backup_root / "restore-child",
        tmp_path,
    ):
        with pytest.raises(ValueError, match="Restore destination must be separate"):
            service.restore_workspace(unsafe)


def test_restore_uses_only_the_latest_manifest_inventory(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    manager = WorkspaceManager(source, tmp_path / "private", tmp_path / "backup")
    manager.bootstrap()
    deleted = manager.projects_dir / "deleted-before-latest.txt"
    deleted.write_text("must not be resurrected", encoding="utf-8")
    service = BackupService(manager)
    assert service.run().ok
    deleted.unlink()
    assert service.run().ok
    assert (
        manager.backup_root / "workspace-current" / "projects" / deleted.name
    ).is_file()

    restored = service.restore_workspace(tmp_path / "restored-exact")
    assert not (restored / "projects" / deleted.name).exists()
    assert (restored / ".daedalus-workspace.json").is_file()


def test_restore_refuses_corrupt_objects_before_creating_destination(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    manager = WorkspaceManager(source, tmp_path / "private", tmp_path / "backup")
    manager.bootstrap()
    (manager.projects_dir / "model.txt").write_text("GOOD", encoding="utf-8")
    service = BackupService(manager)
    assert service.run().ok
    payload = json.loads((manager.backup_root / "latest.json").read_text(encoding="utf-8"))
    entry = next(item for item in payload["inventory"] if item["path"].endswith("model.txt"))
    object_path = manager.backup_root / Path(*entry["object_path"].split("/"))
    object_path.write_text("CORRUPTED", encoding="utf-8")
    target = tmp_path / "must-not-exist"

    with pytest.raises(RuntimeError, match="did not verify"):
        service.restore_workspace(target)
    assert not target.exists()


def test_interrupted_latest_update_preserves_last_restorable_manifest(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    manager = WorkspaceManager(source, tmp_path / "private", tmp_path / "backup")
    manager.bootstrap()
    model = manager.projects_dir / "model.txt"
    model.write_text("KNOWN-GOOD", encoding="utf-8")
    service = BackupService(manager)
    assert service.run().ok
    latest = manager.backup_root / "latest.json"
    committed_manifest = latest.read_bytes()
    model.write_text("NEW-UNCOMMITTED-COPY", encoding="utf-8")
    original_atomic_json = backup_module._atomic_json

    def interrupt_latest(path, payload):
        if Path(path).name == "latest.json":
            raise OSError("simulated crash before commit-pointer replacement")
        return original_atomic_json(path, payload)

    monkeypatch.setattr(backup_module, "_atomic_json", interrupt_latest)
    with pytest.raises(OSError, match="simulated crash"):
        service.run()

    assert latest.read_bytes() == committed_manifest
    assert service.verify().ok
    restored = service.restore_workspace(tmp_path / "restored-known-good")
    assert (restored / "projects" / "model.txt").read_text(encoding="utf-8") == "KNOWN-GOOD"
