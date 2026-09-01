from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BACKUP_SCRIPT = ROOT / "tools" / "backup.ps1"
CONFIGURE_SCRIPT = ROOT / "tools" / "configure-backup.ps1"
ENABLE_BACKUP = ROOT / "Enable-Auto-Backup.bat"
INSTALLER = ROOT / "installer" / "install.ps1"
WINDOWS_POWERSHELL = (
    Path(os.environ.get("SystemRoot", "C:/Windows"))
    / "System32"
    / "WindowsPowerShell"
    / "v1.0"
    / "powershell.exe"
)


def _run_backup(*arguments: str, environment: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            str(WINDOWS_POWERSHELL),
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(BACKUP_SCRIPT),
            *arguments,
        ],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )


def _failure_code(probe: dict[str, object], *, required: int = 1024) -> str:
    environment = os.environ.copy()
    environment["DAEDALUS_TEST_PROBE"] = json.dumps(probe)
    command = (
        ". .\\tools\\backup.ps1; "
        "$probe = $env:DAEDALUS_TEST_PROBE | ConvertFrom-Json; "
        f"$result = Get-BackupPreflightFailureCode -Probe $probe -RequiredFreeBytes {required}; "
        "if ($null -eq $result) { 'ready' } else { $result }"
    )
    result = subprocess.run(
        [
            str(WINDOWS_POWERSHELL),
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            command,
        ],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def _healthy_probe() -> dict[str, object]:
    return {
        "DrivePresent": True,
        "MetadataResolved": True,
        "DirtyKnown": True,
        "Dirty": False,
        "StorageResolved": True,
        "HealthStatus": "Healthy",
        "OperationalStatuses": ["OK"],
        "FreeBytesKnown": True,
        "FreeBytes": 2048,
    }


@pytest.mark.skipif(os.name != "nt", reason="Windows backup launcher")
@pytest.mark.parametrize(
    ("updates", "expected"),
    [
        ({"DrivePresent": False}, "backup-volume-missing"),
        ({"MetadataResolved": False}, "backup-volume-probe-failed"),
        ({"DirtyKnown": False}, "backup-volume-probe-failed"),
        ({"Dirty": True}, "backup-volume-dirty"),
        ({"StorageResolved": False}, "backup-volume-probe-failed"),
        ({"HealthStatus": "Warning"}, "backup-volume-unhealthy"),
        ({"OperationalStatuses": ["Full Repair Needed"]}, "backup-volume-not-operational"),
        ({"FreeBytesKnown": False}, "backup-volume-probe-failed"),
        ({"FreeBytes": 1023}, "backup-free-space-low"),
        ({}, "ready"),
    ],
)
def test_volume_preflight_classifies_fail_closed_states(
    updates: dict[str, object], expected: str
) -> None:
    probe = _healthy_probe()
    probe.update(updates)

    assert _failure_code(probe) == expected


def _unused_drive_letter() -> str | None:
    for letter in "ZYXWVUTSRQPONMLKJIHGFED":
        if not Path(f"{letter}:/").exists():
            return letter
    return None


@pytest.mark.skipif(os.name != "nt", reason="Windows backup launcher")
@pytest.mark.parametrize(
    "destination",
    [
        "relative-backup",
        r"\\localhost\F$\Daedalus-Backups\DaedalusAI",
        "F:\\",
        r"F:\Daedalus-Backups\..\outside",
        r"F:\Daedalus-Backups\CON",
        "F:\\Daedalus-Backups\\trailing.",
        r"F:\Daedalus-Backups\DaedalusAI:stream",
    ],
)
def test_preflight_only_rejects_unsafe_destinations_without_writing(
    destination: str, tmp_path: Path
) -> None:
    workspace = tmp_path / "workspace-must-not-be-created"
    environment = os.environ.copy()
    environment["DAEDALUS_BACKUP_ROOT"] = destination
    environment["DAEDALUS_WORKSPACE_ROOT"] = str(workspace)

    result = _run_backup("-PreflightOnly", environment=environment)

    assert result.returncode == 20
    assert "backup-destination-invalid" in result.stderr
    assert "nothing was copied" in result.stderr.lower()
    assert str(ROOT).lower() not in result.stderr.lower()
    assert not workspace.exists()


@pytest.mark.skipif(os.name != "nt", reason="Windows backup launcher")
def test_preflight_only_missing_volume_is_read_only(tmp_path: Path) -> None:
    drive = _unused_drive_letter()
    if drive is None:
        pytest.skip("no unused local drive letter is available")
    workspace = tmp_path / "workspace-must-not-be-created"
    environment = os.environ.copy()
    environment["DAEDALUS_BACKUP_ROOT"] = f"{drive}:\\Daedalus-Backups\\DaedalusAI"
    environment["DAEDALUS_WORKSPACE_ROOT"] = str(workspace)

    result = _run_backup("-PreflightOnly", environment=environment)

    assert result.returncode == 21
    assert "backup-volume-missing" in result.stderr
    assert "nothing was copied" in result.stderr.lower()
    assert not workspace.exists()


@pytest.mark.skipif(os.name != "nt", reason="Windows backup launcher")
def test_scheduled_missing_volume_records_sanitized_failure(tmp_path: Path) -> None:
    drive = _unused_drive_letter()
    if drive is None:
        pytest.skip("no unused local drive letter is available")
    workspace = tmp_path / "workspace"
    environment = os.environ.copy()
    environment["DAEDALUS_BACKUP_ROOT"] = f"{drive}:\\Daedalus-Backups\\DaedalusAI"
    environment["DAEDALUS_WORKSPACE_ROOT"] = str(workspace)

    result = _run_backup("-Scheduled", environment=environment)

    assert result.returncode == 21
    assert result.stdout == ""
    assert result.stderr == ""
    health = json.loads((workspace / ".daedalus" / "backup-health.json").read_text())
    assert health["kind"] == "daedalus-backup-health"
    assert health["state"] == "failed"
    assert health["failure_code"] == "backup-volume-missing"
    assert health["last_exit_code"] == 21
    assert health["scheduled"] is True
    assert str(ROOT) not in json.dumps(health)


def test_every_backup_entry_point_uses_the_guarded_preflight() -> None:
    backup = BACKUP_SCRIPT.read_text(encoding="utf-8")
    configure = CONFIGURE_SCRIPT.read_text(encoding="utf-8")
    enable = ENABLE_BACKUP.read_text(encoding="utf-8")
    installer = INSTALLER.read_text(encoding="utf-8")

    assert backup.index("$Preflight = Invoke-BackupPreflight") < backup.index(
        "& $Python -m daedalus.services.backup"
    )
    assert "$MinimumFreeBytes = [uint64](5GB)" in backup
    assert "DirtyBitSet" in backup
    assert "Get-Volume" in backup
    assert "backup-volume-dirty" in backup
    assert "backup-volume-unhealthy" in backup
    assert "backup-volume-not-operational" in backup
    assert "backup-free-space-low" in backup
    assert "-File $BackupScript -PreflightOnly" in configure
    assert "New-ScheduledTaskAction" in configure
    assert "-Scheduled" in configure
    for contract in (
        "-StartWhenAvailable",
        "-MultipleInstances IgnoreNew",
        "-RestartCount 3",
        "-ExecutionTimeLimit (New-TimeSpan -Minutes 45)",
    ):
        assert contract in configure
    assert "tools\\configure-backup.ps1" in enable
    assert "-File $InitialBackupLauncher -Scheduled" in installer
    assert "& $VenvPython -m daedalus.services.backup" not in installer
