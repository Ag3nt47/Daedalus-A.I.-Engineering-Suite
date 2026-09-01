"""Content-verified, non-destructive backups and isolated recovery."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import socket
import sqlite3
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Iterable
from urllib.parse import quote

import psutil

from daedalus.workspace.manager import WorkspaceManager

_MARKER = ".daedalus-backup-root.json"
_LOCK = ".daedalus-backup.lock"
_HEALTH = "backup-health.json"
_MANIFEST_KIND = "daedalus-backup-manifest"
_MANIFEST_SCHEMA = 3
_OBJECTS_DIRECTORY = "objects"
_MALFORMED_LOCK_MINIMUM_AGE_SECONDS = 24 * 60 * 60
_SOURCE_ROOT_EXCLUDED_NAMES = {
    ".git",
    ".venv",
    "venv",
    "env",
    "build",
    "dist",
    ".tmp",
}
_ALWAYS_EXCLUDED_NAMES = {"__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache"}
_SQLITE_HEADER = b"SQLite format 3\x00"
_SQLITE_SIDECAR_SUFFIXES = ("-wal", "-shm", "-journal")
_VOLUME_PREFLIGHT_MINIMUM_FREE_BYTES = 5 * 1024**3
_VOLUME_PREFLIGHT_FAILURE_CODES = {
    20: "backup-destination-invalid",
    21: "backup-volume-missing",
    22: "backup-volume-dirty",
    23: "backup-volume-unhealthy",
    24: "backup-volume-not-operational",
    25: "backup-free-space-low",
    26: "backup-volume-probe-failed",
}


class BackupLockError(RuntimeError):
    """Raised when another backup lock cannot safely be reclaimed."""


class BackupVolumePreflightError(RuntimeError):
    """Raised when the production backup volume is not safe to write."""

    def __init__(self, failure_code: str) -> None:
        self.failure_code = failure_code
        super().__init__(f"Backup volume preflight failed ({failure_code}). Nothing was copied.")


@dataclass(frozen=True, slots=True)
class InventoryEntry:
    path: str
    size: int
    sha256: str
    kind: str = "file"
    object_path: str | None = None


@dataclass(slots=True)
class BackupResult:
    started_utc: str
    finished_utc: str
    destination: str
    files_scanned: int
    files_copied: int
    bytes_copied: int
    skipped_links: int
    errors: list[str]
    skipped_sqlite_sidecars: int = 0
    inventory: list[InventoryEntry] = field(default_factory=list)
    kind: str = _MANIFEST_KIND
    schema: int = _MANIFEST_SCHEMA

    @property
    def ok(self) -> bool:
        return not self.errors


@dataclass(slots=True)
class VerificationResult:
    verified_utc: str
    manifest: str
    files_checked: int
    missing: list[str]
    mismatched: list[str]
    errors: list[str]

    @property
    def ok(self) -> bool:
        return not self.missing and not self.mismatched and not self.errors


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_sha256(path: Path, *, attempts: int = 3) -> tuple[str, int]:
    """Hash a regular file only when its metadata is stable for the read."""

    for _ in range(attempts):
        before = path.stat()
        digest = _sha256(path)
        after = path.stat()
        if (
            before.st_size == after.st_size
            and before.st_mtime_ns == after.st_mtime_ns
            and before.st_ino == after.st_ino
        ):
            return digest, after.st_size
    raise OSError("source changed repeatedly while it was being checksummed")


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _utc_from_timestamp(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, UTC).isoformat()


class BackupService:
    """Mirror current files without deleting older destination files.

    Destination-only files are retained intentionally, so an accidental local
    deletion is not immediately propagated. Every current source file is
    compared by content and represented in the latest manifest inventory.
    """

    def __init__(self, manager: WorkspaceManager) -> None:
        self.manager = manager
        self.backup_root = manager.backup_root.resolve(strict=False)

    @property
    def health_path(self) -> Path:
        return self.manager.settings_dir / _HEALTH

    @property
    def objects_root(self) -> Path:
        return self.backup_root / _OBJECTS_DIRECTORY / "sha256"

    @staticmethod
    def _marker_is_valid(marker: Path) -> bool:
        try:
            if marker.is_symlink() or not marker.is_file() or marker.stat().st_size > 64 * 1024:
                return False
            payload = json.loads(marker.read_text(encoding="utf-8"))
            if (
                not isinstance(payload, dict)
                or payload.get("kind") != "daedalus-backup-root"
                or payload.get("schema") != 1
            ):
                return False
            created = datetime.fromisoformat(str(payload["created_utc"]))
            return created.tzinfo is not None
        except (KeyError, OSError, UnicodeError, ValueError, json.JSONDecodeError):
            return False

    def validate_destination(self) -> None:
        for protected in (self.manager.source_root, self.manager.workspace_root):
            protected = protected.resolve(strict=False)
            if (
                self.backup_root == protected
                or self.backup_root in protected.parents
                or protected in self.backup_root.parents
            ):
                raise ValueError(
                    f"Backup destination must be separate from source paths: {self.backup_root}"
                )

        if os.name == "nt" and self.backup_root.drive:
            drive_root = Path(self.backup_root.drive + "\\")
            if not drive_root.exists():
                raise FileNotFoundError(
                    f"Backup drive {self.backup_root.drive} is not connected. Nothing was copied."
                )

        if self.backup_root.exists():
            if not self.backup_root.is_dir() or self.backup_root.is_symlink():
                raise RuntimeError(f"Backup destination is not a regular directory: {self.backup_root}")
            marker = self.backup_root / _MARKER
            contents = list(self.backup_root.iterdir())
            if contents and not self._marker_is_valid(marker):
                raise RuntimeError(
                    "The destination is non-empty and lacks a valid Daedalus backup marker: "
                    f"{self.backup_root}"
                )

    def _run_required_volume_preflight(self) -> None:
        """Run the production Windows volume gate without trusting an env bypass."""

        if not self.manager.require_backup_volume_preflight or os.name != "nt":
            return

        preflight_script = self.manager.source_root / "tools" / "backup.ps1"
        windows_root = Path(os.environ.get("SystemRoot", r"C:\Windows"))
        powershell = windows_root / "System32" / "WindowsPowerShell" / "v1.0" / "powershell.exe"
        if (
            not preflight_script.is_file()
            or preflight_script.is_symlink()
            or not powershell.is_file()
            or powershell.is_symlink()
        ):
            raise BackupVolumePreflightError("backup-volume-probe-failed")

        environment = os.environ.copy()
        environment["DAEDALUS_WORKSPACE_ROOT"] = str(self.manager.workspace_root)
        environment["DAEDALUS_BACKUP_ROOT"] = str(self.backup_root)
        try:
            completed = subprocess.run(
                [
                    str(powershell),
                    "-NoLogo",
                    "-NoProfile",
                    "-NonInteractive",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-File",
                    str(preflight_script),
                    "-PreflightOnly",
                ],
                cwd=self.manager.source_root,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
                timeout=45,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise BackupVolumePreflightError("backup-volume-probe-failed") from exc

        if completed.returncode != 0:
            failure_code = _VOLUME_PREFLIGHT_FAILURE_CODES.get(
                completed.returncode, "backup-volume-probe-failed"
            )
            raise BackupVolumePreflightError(failure_code)

        try:
            payload = json.loads(completed.stdout)
            expected_drive = self.backup_root.drive.rstrip(":\\/").upper()
            free_bytes = payload["free_bytes"]
            required_free_bytes = payload["required_free_bytes"]
            valid = bool(
                isinstance(payload, dict)
                and payload.get("kind") == "daedalus-backup-preflight"
                and payload.get("schema") == 1
                and payload.get("state") == "ready"
                and expected_drive
                and str(payload.get("drive", "")).upper() == expected_drive
                and type(free_bytes) is int
                and type(required_free_bytes) is int
                and required_free_bytes >= _VOLUME_PREFLIGHT_MINIMUM_FREE_BYTES
                and free_bytes >= required_free_bytes
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            valid = False
        if not valid:
            raise BackupVolumePreflightError("backup-volume-probe-failed")

    def _ensure_root(self) -> None:
        self.validate_destination()
        self.backup_root.mkdir(parents=True, exist_ok=True)
        marker = self.backup_root / _MARKER
        if not marker.exists():
            _atomic_json(
                marker,
                {
                    "kind": "daedalus-backup-root",
                    "schema": 1,
                    "created_utc": datetime.now(UTC).isoformat(),
                },
            )
        if not self._marker_is_valid(marker):
            raise RuntimeError(f"Backup ownership marker is invalid: {marker}")

    def _read_health(self) -> dict[str, Any] | None:
        for path in (self.health_path, self.backup_root / _HEALTH):
            if not path.is_file():
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return {
                    "kind": "daedalus-backup-health",
                    "schema": 1,
                    "state": "unknown",
                    "failure_code": "health-record-invalid",
                }
            if isinstance(payload, dict):
                return payload
        return None

    def _record_health(
        self,
        state: str,
        *,
        failure_code: str | None = None,
        manifest: str | None = None,
        verification: dict[str, Any] | None = None,
        include_destination: bool = True,
    ) -> None:
        now = datetime.now(UTC).isoformat()
        previous = self._read_health() or {}
        payload: dict[str, Any] = {
            "kind": "daedalus-backup-health",
            "schema": 1,
            "state": state,
            "updated_utc": now,
            "last_attempt_utc": now,
            "last_success_utc": previous.get("last_success_utc"),
            "last_failure_utc": previous.get("last_failure_utc"),
            "failure_code": failure_code,
            "last_manifest": manifest or previous.get("last_manifest"),
        }
        if state == "healthy":
            payload["last_success_utc"] = now
        else:
            payload["last_failure_utc"] = now
        if verification is not None:
            payload["verification"] = verification
        elif isinstance(previous.get("verification"), dict):
            payload["verification"] = previous["verification"]

        destinations = [self.health_path]
        if include_destination and self._marker_is_valid(self.backup_root / _MARKER):
            destinations.append(self.backup_root / _HEALTH)
        for path in destinations:
            try:
                _atomic_json(path, payload)
            except OSError:
                # Health persistence must never conceal the original backup result.
                continue

    @staticmethod
    def _failure_code(error: BaseException) -> str:
        if isinstance(error, BackupLockError):
            return "lock-unavailable"
        if isinstance(error, FileNotFoundError):
            return "destination-unavailable"
        if isinstance(error, PermissionError):
            return "permission-denied"
        if isinstance(error, (json.JSONDecodeError, sqlite3.DatabaseError)):
            return "data-invalid"
        return "backup-exception"

    @staticmethod
    def _lock_is_demonstrably_stale(payload: object) -> bool:
        if not isinstance(payload, dict):
            return False
        if payload.get("kind") != "daedalus-backup-lock" or payload.get("schema") != 1:
            return False
        if str(payload.get("host", "")).casefold() != socket.gethostname().casefold():
            return False
        try:
            pid = int(payload["pid"])
            recorded = datetime.fromisoformat(str(payload["process_started_utc"]))
            recorded_timestamp = recorded.timestamp()
        except (KeyError, TypeError, ValueError, OverflowError):
            return False
        if pid <= 0:
            return False
        try:
            process = psutil.Process(pid)
            observed_timestamp = process.create_time()
        except psutil.NoSuchProcess:
            return True
        except (psutil.AccessDenied, OSError):
            return False
        # A live process with a reused PID proves that the recorded owner is gone.
        return abs(observed_timestamp - recorded_timestamp) > 2.0

    @staticmethod
    def _lock_payload_is_structured(payload: object) -> bool:
        if not isinstance(payload, dict):
            return False
        if payload.get("kind") != "daedalus-backup-lock" or payload.get("schema") != 1:
            return False
        try:
            pid = int(payload["pid"])
            host = str(payload["host"]).strip()
            token = str(payload["token"]).strip()
            process_started = datetime.fromisoformat(str(payload["process_started_utc"]))
            lock_created = datetime.fromisoformat(str(payload["lock_created_utc"]))
        except (KeyError, TypeError, ValueError, OverflowError):
            return False
        return bool(
            pid > 0
            and host
            and token
            and process_started.tzinfo is not None
            and lock_created.tzinfo is not None
        )

    @staticmethod
    def _malformed_lock_is_demonstrably_ancient(lock_path: Path) -> bool:
        """Reclaim only crash debris that predates both a day and this OS boot.

        A recent or same-boot malformed lock could still belong to a process that
        has not finished writing or releasing it, so it is always preserved.
        """

        try:
            if lock_path.is_symlink():
                return False
            modified = lock_path.stat().st_mtime
            now = time.time()
            booted = float(psutil.boot_time())
        except (OSError, TypeError, ValueError):
            return False
        return (
            now - modified >= _MALFORMED_LOCK_MINIMUM_AGE_SECONDS
            and modified < booted - 60
        )

    @staticmethod
    def _quarantine_lock(lock_path: Path, observed: bytes, observed_stat: os.stat_result) -> Path:
        """Atomically preserve proven crash debris under a unique evidence name."""

        if lock_path.is_symlink():
            raise BackupLockError("A symbolic backup lock is not safe to reclaim.")
        try:
            current_stat = lock_path.stat()
            if (
                current_stat.st_size != observed_stat.st_size
                or current_stat.st_mtime_ns != observed_stat.st_mtime_ns
                or current_stat.st_ino != observed_stat.st_ino
                or lock_path.read_bytes() != observed
            ):
                raise BackupLockError("The backup lock changed while it was inspected.")
            stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
            quarantine = lock_path.with_name(
                f"{lock_path.name}.quarantine-{stamp}-{uuid.uuid4().hex}"
            )
            lock_path.replace(quarantine)
            return quarantine
        except FileNotFoundError:
            raise BackupLockError("The backup lock changed while it was inspected.") from None

    def _acquire_lock(self) -> str:
        lock_path = self.backup_root / _LOCK
        token = uuid.uuid4().hex
        process_started = _utc_from_timestamp(psutil.Process(os.getpid()).create_time())
        payload = {
            "kind": "daedalus-backup-lock",
            "schema": 1,
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "process_started_utc": process_started,
            "lock_created_utc": datetime.now(UTC).isoformat(),
            "token": token,
        }
        encoded = json.dumps(payload, indent=2).encode("utf-8")

        for _ in range(2):
            try:
                descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError as exc:
                try:
                    if lock_path.is_symlink():
                        raise BackupLockError("A symbolic backup lock is not safe to reclaim.")
                    observed_stat = lock_path.stat()
                    observed = lock_path.read_bytes()
                    existing = json.loads(observed.decode("utf-8"))
                except BackupLockError:
                    raise
                except (OSError, UnicodeDecodeError, json.JSONDecodeError) as read_error:
                    try:
                        observed_stat = lock_path.stat()
                        observed = lock_path.read_bytes()
                    except OSError:
                        raise BackupLockError(
                            "A backup lock exists but is not safe to reclaim automatically."
                        ) from read_error
                    if self._malformed_lock_is_demonstrably_ancient(lock_path):
                        self._quarantine_lock(lock_path, observed, observed_stat)
                        continue
                    raise BackupLockError(
                        "A backup lock exists but is not safe to reclaim automatically."
                    ) from read_error
                if self._lock_is_demonstrably_stale(existing):
                    self._quarantine_lock(lock_path, observed, observed_stat)
                    continue
                if not self._lock_payload_is_structured(existing):
                    if self._malformed_lock_is_demonstrably_ancient(lock_path):
                        self._quarantine_lock(lock_path, observed, observed_stat)
                        continue
                    raise BackupLockError(
                        "A backup lock exists but is not safe to reclaim automatically."
                    ) from exc
                else:
                    raise BackupLockError("A Daedalus backup is already running.") from exc
            try:
                with os.fdopen(descriptor, "wb") as handle:
                    handle.write(encoded)
                    handle.flush()
                    os.fsync(handle.fileno())
            except Exception:
                lock_path.unlink(missing_ok=True)
                raise
            return token
        raise BackupLockError("The stale backup lock could not be reclaimed safely.")

    def _release_lock(self, token: str) -> None:
        lock_path = self.backup_root / _LOCK
        try:
            payload = json.loads(lock_path.read_text(encoding="utf-8"))
            if payload.get("token") == token:
                lock_path.unlink(missing_ok=True)
        except (OSError, json.JSONDecodeError, AttributeError):
            # Never remove a lock that cannot be proven to belong to this run.
            return

    @staticmethod
    def _is_sqlite_database(path: Path) -> bool:
        try:
            with path.open("rb") as handle:
                return handle.read(len(_SQLITE_HEADER)) == _SQLITE_HEADER
        except OSError:
            return False

    @classmethod
    def _sqlite_sidecar_base(cls, path: Path) -> Path | None:
        """Return the verified SQLite database owning a real transient sidecar."""

        folded = path.name.casefold()
        for suffix in _SQLITE_SIDECAR_SUFFIXES:
            if not folded.endswith(suffix) or len(path.name) == len(suffix):
                continue
            database = path.with_name(path.name[: -len(suffix)])
            if (
                database.is_file()
                and not database.is_symlink()
                and cls._is_sqlite_database(database)
            ):
                return database
        return None

    @staticmethod
    def _temporary_target(target: Path) -> Path:
        return target.with_name(
            f".{target.name}.{os.getpid()}.{uuid.uuid4().hex}.daedalus-copying"
        )

    @staticmethod
    def _is_within(candidate: Path, root: Path) -> bool:
        resolved_candidate = candidate.resolve(strict=False)
        resolved_root = root.resolve(strict=False)
        return resolved_candidate == resolved_root or resolved_root in resolved_candidate.parents

    @staticmethod
    def _digest_is_valid(value: str) -> bool:
        return len(value) == 64 and all(character in "0123456789abcdef" for character in value)

    @staticmethod
    def _object_relative(digest: str) -> PurePosixPath:
        return PurePosixPath(_OBJECTS_DIRECTORY, "sha256", digest[:2], digest)

    def _object_path(self, digest: str) -> Path:
        relative = self._object_relative(digest)
        target = self.backup_root / Path(*relative.parts)
        if not self._is_within(target, self.backup_root):
            raise OSError("content-addressed object path escapes the backup root")
        return target

    def _staging_directory(self) -> Path:
        staging = self.backup_root / _OBJECTS_DIRECTORY / ".staging"
        if not self._is_within(staging, self.backup_root):
            raise OSError("backup staging path escapes the backup root")
        if staging.exists() and (staging.is_symlink() or not staging.is_dir()):
            raise OSError("backup staging path is not a regular directory")
        staging.mkdir(parents=True, exist_ok=True)
        return staging

    @staticmethod
    def _object_matches(path: Path, digest: str, size: int) -> bool:
        if path.is_symlink() or not path.is_file():
            return False
        try:
            observed_digest, observed_size = _stable_sha256(path)
        except OSError:
            return False
        return observed_digest == digest and observed_size == size

    def _publish_staged_object(
        self, staged: Path, digest: str, size: int
    ) -> tuple[Path, bool]:
        target = self._object_path(digest)
        target.parent.mkdir(parents=True, exist_ok=True)
        if self._object_matches(target, digest, size):
            staged.unlink(missing_ok=True)
            return target, False
        if target.exists() and (target.is_symlink() or not target.is_file()):
            raise OSError("content-addressed backup target is not a regular file")
        staged_digest, staged_size = _stable_sha256(staged)
        if staged_digest != digest or staged_size != size:
            raise OSError("staged backup object failed its checksum contract")
        staged.replace(target)
        return target, True

    def _capture_regular(self, source: Path) -> tuple[Path, str, int, bool]:
        source_digest, source_size = _stable_sha256(source)
        existing = self._object_path(source_digest)
        if self._object_matches(existing, source_digest, source_size):
            return existing, source_digest, source_size, False

        staging_directory = self._staging_directory()
        for _ in range(3):
            temporary = staging_directory / f"{uuid.uuid4().hex}.file"
            try:
                shutil.copy2(source, temporary)
                copied_digest, copied_size = _stable_sha256(temporary)
                current_digest, current_size = _stable_sha256(source)
                if copied_digest == current_digest and copied_size == current_size:
                    target, created = self._publish_staged_object(
                        temporary, copied_digest, copied_size
                    )
                    return target, copied_digest, copied_size, created
            finally:
                temporary.unlink(missing_ok=True)
        raise OSError("source changed repeatedly while it was being copied")

    def _capture_sqlite(self, source: Path) -> tuple[Path, str, int, bool]:
        staging_directory = self._staging_directory()
        temporary = staging_directory / f"{uuid.uuid4().hex}.sqlite"
        source_uri = "file:" + quote(source.resolve().as_posix(), safe="/:?") + "?mode=ro"
        source_connection: sqlite3.Connection | None = None
        target_connection: sqlite3.Connection | None = None
        try:
            source_connection = sqlite3.connect(source_uri, uri=True, timeout=30)
            target_connection = sqlite3.connect(temporary, timeout=30)
            source_connection.backup(target_connection, pages=256, sleep=0.01)
            check = target_connection.execute("PRAGMA quick_check").fetchone()
            if not check or check[0] != "ok":
                raise sqlite3.DatabaseError("SQLite snapshot did not pass quick_check")
            target_connection.close()
            target_connection = None
            source_connection.close()
            source_connection = None
            shutil.copystat(source, temporary)
            digest, size = _stable_sha256(temporary)
            target, created = self._publish_staged_object(temporary, digest, size)
            return target, digest, size, created
        finally:
            if target_connection is not None:
                target_connection.close()
            if source_connection is not None:
                source_connection.close()
            temporary.unlink(missing_ok=True)

    def _materialize_object(self, source: Path, target: Path, digest: str, size: int) -> bool:
        if not self._object_matches(source, digest, size):
            raise OSError("content-addressed backup object failed verification")
        if target.exists():
            if target.is_symlink() or not target.is_file():
                raise OSError("backup target is not a regular file")
            target_digest, target_size = _stable_sha256(target)
            if target_digest == digest and target_size == size:
                return False

        temporary = self._temporary_target(target)
        try:
            temporary.unlink(missing_ok=True)
            shutil.copy2(source, temporary)
            copied_digest, copied_size = _stable_sha256(temporary)
            if copied_digest != digest or copied_size != size:
                raise OSError("materialized backup copy failed verification")
            temporary.replace(target)
            return True
        finally:
            temporary.unlink(missing_ok=True)

    def _copy_tree(
        self,
        source: Path,
        destination: Path,
        counters: dict[str, int],
        errors: list[str],
        inventory: list[InventoryEntry],
        *,
        root_exclusions: frozenset[str] = frozenset(),
    ) -> None:
        original_source = source
        original_destination = destination
        if original_source.is_symlink():
            errors.append(f"{original_source}: backup source cannot be a symbolic link")
            return
        if not self._is_within(original_destination, self.backup_root):
            errors.append(f"{original_destination}: backup mirror escaped the backup root")
            return
        source = original_source.resolve(strict=False)
        destination = original_destination.resolve(strict=False)
        if not source.is_dir():
            errors.append(f"{source}: backup source is not a regular directory")
            return
        if original_destination.exists() and (
            original_destination.is_symlink() or not original_destination.is_dir()
        ):
            errors.append(f"{destination}: backup mirror is not a regular directory")
            return
        destination.mkdir(parents=True, exist_ok=True)

        def walk_error(error: OSError) -> None:
            location = getattr(error, "filename", None) or str(source)
            errors.append(f"{location}: directory enumeration failed ({type(error).__name__})")

        for root, directories, files in os.walk(
            source, topdown=True, onerror=walk_error, followlinks=False
        ):
            root_path = Path(root)
            if not self._is_within(root_path, source):
                errors.append(f"{root_path}: directory escaped the backup source")
                directories[:] = []
                continue
            relative = root_path.relative_to(source)
            retained_directories: list[str] = []
            for name in directories:
                directory = root_path / name
                if name.casefold() in _ALWAYS_EXCLUDED_NAMES:
                    continue
                if relative == Path(".") and name.casefold() in root_exclusions:
                    continue
                if directory.is_symlink() or not self._is_within(directory, source):
                    counters["links"] += 1
                    continue
                retained_directories.append(name)
            directories[:] = retained_directories
            target_root = destination / relative
            if not self._is_within(target_root, destination):
                errors.append(f"{target_root}: backup mirror escaped its custody root")
                directories[:] = []
                continue
            if target_root.exists() and (
                target_root.is_symlink() or not target_root.is_dir()
            ):
                errors.append(f"{target_root}: backup mirror path is not a regular directory")
                directories[:] = []
                continue
            target_root.mkdir(parents=True, exist_ok=True)
            for name in files:
                source_file = root_path / name
                if source_file.resolve(strict=False) == self.health_path.resolve(strict=False):
                    continue
                if source_file.is_symlink() or not self._is_within(source_file, source):
                    counters["links"] += 1
                    continue
                if self._sqlite_sidecar_base(source_file) is not None:
                    counters["sidecars"] += 1
                    continue
                counters["scanned"] += 1
                target_file = target_root / name
                try:
                    target_file.parent.mkdir(parents=True, exist_ok=True)
                    if not self._is_within(target_file, destination):
                        raise OSError("backup target escaped its custody root")
                    is_sqlite = self._is_sqlite_database(source_file)
                    if is_sqlite:
                        object_path, digest, size, object_created = self._capture_sqlite(
                            source_file
                        )
                    else:
                        object_path, digest, size, object_created = self._capture_regular(
                            source_file
                        )
                    mirror_changed = self._materialize_object(
                        object_path, target_file, digest, size
                    )
                    if object_created or mirror_changed:
                        counters["copied"] += 1
                        counters["bytes"] += size
                    inventory.append(
                        InventoryEntry(
                            target_file.relative_to(self.backup_root).as_posix(),
                            size,
                            digest,
                            "sqlite" if is_sqlite else "file",
                            self._object_relative(digest).as_posix(),
                        )
                    )
                except (OSError, ValueError, sqlite3.DatabaseError) as exc:
                    errors.append(f"{source_file}: {exc}")

    def run(self) -> BackupResult:
        started = datetime.now(UTC)
        lock_token: str | None = None
        try:
            self._run_required_volume_preflight()
            self.manager.bootstrap()
            self._ensure_root()
            lock_token = self._acquire_lock()
            counters = {"scanned": 0, "copied": 0, "bytes": 0, "links": 0, "sidecars": 0}
            errors: list[str] = []
            inventory: list[InventoryEntry] = []
            self._copy_tree(
                self.manager.source_root,
                self.backup_root / "source-current",
                counters,
                errors,
                inventory,
                root_exclusions=frozenset(_SOURCE_ROOT_EXCLUDED_NAMES),
            )
            self._copy_tree(
                self.manager.workspace_root,
                self.backup_root / "workspace-current",
                counters,
                errors,
                inventory,
            )
            finished = datetime.now(UTC)
            inventory.sort(key=lambda entry: entry.path.casefold())
            result = BackupResult(
                started.isoformat(),
                finished.isoformat(),
                str(self.backup_root),
                counters["scanned"],
                counters["copied"],
                counters["bytes"],
                counters["links"],
                errors,
                counters["sidecars"],
                inventory,
            )
            manifests = self.backup_root / "manifests"
            manifests.mkdir(exist_ok=True)
            stamp = finished.strftime("%Y%m%dT%H%M%S%fZ")
            manifest = manifests / f"backup-{stamp}.json"
            payload = asdict(result)
            _atomic_json(manifest, payload)
            # A failed or interrupted run must never replace the last complete
            # recovery pointer. Its timestamped manifest remains reviewable.
            if result.ok:
                _atomic_json(self.backup_root / "latest.json", payload)
            self._record_health(
                "healthy" if result.ok else "failed",
                failure_code=None if result.ok else "copy-errors",
                manifest=manifest.name,
            )
            return result
        except BackupVolumePreflightError as exc:
            # An unsafe destination must receive no write, including no mirrored
            # health record. Preserve only sanitized evidence in the local
            # workspace health path after the read-only gate has failed.
            self._record_health(
                "failed",
                failure_code=exc.failure_code,
                include_destination=False,
            )
            raise
        except Exception as exc:
            self._record_health("failed", failure_code=self._failure_code(exc))
            raise
        finally:
            if lock_token is not None:
                self._release_lock(lock_token)

    def latest_status(self) -> dict[str, object] | None:
        health = self._read_health()
        manifest = self.backup_root / "latest.json"
        if not manifest.is_file():
            if health is None:
                return None
            return {"status": "no backup yet", "health": health}
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {
                "status": "latest manifest is unreadable",
                "health": health
                or {
                    "state": "unknown",
                    "failure_code": "latest-manifest-invalid",
                },
            }
        if not isinstance(payload, dict):
            return {
                "status": "latest manifest is unreadable",
                "health": health
                or {
                    "state": "unknown",
                    "failure_code": "latest-manifest-invalid",
                },
            }
        payload["health"] = health
        return payload

    def verify(self, manifest_path: Path | None = None) -> VerificationResult:
        manifest_candidate = manifest_path or (self.backup_root / "latest.json")
        manifest_is_symlink = manifest_candidate.is_symlink()
        manifest = manifest_candidate.resolve(strict=False)
        checked = 0
        missing: list[str] = []
        mismatched: list[str] = []
        errors: list[str] = []
        backup_root = self.backup_root.resolve(strict=False)
        if manifest != backup_root and backup_root not in manifest.parents:
            errors.append("manifest path escapes the marked backup root")
        if manifest_is_symlink:
            errors.append("manifest cannot be a symbolic link")
        if not self._marker_is_valid(self.backup_root / _MARKER):
            errors.append("backup ownership marker is missing or invalid")
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            payload = {}
            errors.append(f"manifest could not be read: {type(exc).__name__}")

        schema: int | None = None
        if not isinstance(payload, dict):
            errors.append("manifest root must be a JSON object")
            payload = {}
        if payload.get("kind") != _MANIFEST_KIND:
            errors.append("manifest kind is missing or unsupported")
        raw_schema = payload.get("schema")
        if type(raw_schema) is not int or raw_schema not in {2, _MANIFEST_SCHEMA}:
            errors.append("manifest schema is missing or unsupported")
        else:
            schema = raw_schema

        inventory = payload.get("inventory")
        if not isinstance(inventory, list):
            errors.append("manifest does not contain a file inventory")
            inventory = []
        elif not inventory:
            errors.append("manifest inventory is empty")
        manifest_errors = payload.get("errors")
        if not isinstance(manifest_errors, list):
            errors.append("manifest errors field is invalid")
        elif manifest_errors:
            errors.append("the backup run recorded copy errors")
        try:
            raw_files_scanned = payload["files_scanned"]
            if type(raw_files_scanned) is not int:
                raise TypeError("files_scanned must be an integer")
            if raw_files_scanned != len(inventory):
                errors.append("manifest inventory count does not match files_scanned")
        except (KeyError, TypeError, ValueError, OverflowError):
            errors.append("manifest files_scanned field is invalid")

        seen: set[str] = set()
        workspace_entries = 0
        for index, raw_entry in enumerate(inventory):
            if not isinstance(raw_entry, dict):
                errors.append(f"inventory entry {index} is invalid")
                continue
            relative_text = str(raw_entry.get("path", ""))
            pure = PurePosixPath(relative_text)
            if (
                not relative_text
                or "\\" in relative_text
                or pure.is_absolute()
                or ".." in pure.parts
                or len(pure.parts) < 2
                or pure.parts[0] not in {"source-current", "workspace-current"}
                or pure.as_posix() != relative_text
            ):
                errors.append(f"inventory entry {index} has an unsafe path")
                continue
            normalized = pure.as_posix()
            identity = normalized.casefold()
            if identity in seen:
                errors.append(f"inventory contains duplicate path: {normalized}")
                continue
            seen.add(identity)
            workspace_entries += pure.parts[0] == "workspace-current"
            try:
                raw_size = raw_entry["size"]
                if type(raw_size) is not int:
                    raise TypeError("size must be an integer")
                expected_size = raw_size
                expected_digest = str(raw_entry["sha256"]).casefold()
                entry_kind = str(raw_entry["kind"])
            except (KeyError, TypeError, ValueError, OverflowError) as exc:
                errors.append(f"{normalized}: metadata is invalid ({type(exc).__name__})")
                continue
            if (
                expected_size < 0
                or not self._digest_is_valid(expected_digest)
                or entry_kind not in {"file", "sqlite"}
            ):
                errors.append(f"{normalized}: size, digest, or kind is invalid")
                continue

            if schema == _MANIFEST_SCHEMA:
                object_text = str(raw_entry.get("object_path", ""))
                expected_object = self._object_relative(expected_digest).as_posix()
                if object_text != expected_object or "\\" in object_text:
                    errors.append(f"{normalized}: content-object path is invalid")
                    continue
                target = (backup_root / Path(*PurePosixPath(object_text).parts)).resolve(
                    strict=False
                )
            else:
                target = (backup_root / Path(*pure.parts)).resolve(strict=False)
            if backup_root not in target.parents:
                errors.append(f"inventory entry {index} escapes the backup root")
                continue
            if not target.is_file() or target.is_symlink():
                missing.append(normalized)
                continue
            try:
                digest, size = _stable_sha256(target)
            except OSError as exc:
                errors.append(f"{normalized}: could not verify ({type(exc).__name__})")
                continue
            checked += 1
            if size != expected_size or digest.casefold() != expected_digest:
                mismatched.append(normalized)
        if inventory and workspace_entries == 0:
            errors.append("manifest contains no private-workspace files")

        result = VerificationResult(
            datetime.now(UTC).isoformat(),
            str(manifest),
            checked,
            missing,
            mismatched,
            errors,
        )
        verification_health = {
            "verified_utc": result.verified_utc,
            "ok": result.ok,
            "files_checked": checked,
            "missing": len(missing),
            "mismatched": len(mismatched),
            "errors": len(errors),
        }
        self._record_health(
            "healthy" if result.ok else "failed",
            failure_code=None if result.ok else "verification-failed",
            verification=verification_health,
            # Verification is deliberately read-only with respect to the
            # recovery volume. This remains safe even when Windows has marked
            # that volume dirty or unhealthy; only local status evidence is
            # refreshed.
            include_destination=False,
        )
        return result

    def restore_workspace(
        self, destination: Path | None = None, *, manifest_path: Path | None = None
    ) -> Path:
        """Restore one verified manifest into a new, exact workspace directory."""

        manifest_candidate = manifest_path or (self.backup_root / "latest.json")
        if manifest_candidate.is_symlink():
            raise RuntimeError("Backup manifest is a symbolic link; restore was not started.")
        manifest = manifest_candidate.resolve(strict=False)
        try:
            committed_manifest = manifest.read_bytes()
        except OSError as exc:
            raise RuntimeError("Backup manifest could not be read; restore was not started.") from exc
        verification = self.verify(manifest)
        if not verification.ok:
            raise RuntimeError("Backup manifest did not verify; restore was not started.")
        try:
            if manifest.read_bytes() != committed_manifest:
                raise RuntimeError("Backup manifest changed during verification; restore was not started.")
            payload = json.loads(committed_manifest.decode("utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("Backup manifest changed during verification; restore was not started.") from exc
        schema = int(payload["schema"])
        workspace_inventory = [
            item
            for item in payload["inventory"]
            if str(item["path"]).startswith("workspace-current/")
        ]
        if not workspace_inventory:
            raise RuntimeError("Verified manifest has no private workspace to restore.")
        stamp = time.strftime("%Y%m%d-%H%M%S")
        requested_target = (
            destination
            or self.manager.workspace_root.with_name(
                f"{self.manager.workspace_root.name} Restored {stamp}"
            )
        )
        if requested_target.is_symlink():
            raise FileExistsError(
                f"Restore destination is a symbolic link: {requested_target}"
            )
        target = requested_target.resolve(strict=False)
        for label, protected in (
            ("public source", self.manager.source_root),
            ("active workspace", self.manager.workspace_root),
            ("backup root", self.backup_root),
        ):
            protected = protected.resolve(strict=False)
            if (
                target == protected
                or target in protected.parents
                or protected in target.parents
            ):
                raise ValueError(
                    f"Restore destination must be separate from the {label}: {target}"
                )
        if target.exists():
            raise FileExistsError(f"Restore destination already exists: {target}")
        target.mkdir(parents=True, exist_ok=False)
        backup_root = self.backup_root.resolve(strict=False)
        for raw_entry in workspace_inventory:
            logical = PurePosixPath(str(raw_entry["path"]))
            relative = Path(*logical.parts[1:])
            output = (target / relative).resolve(strict=False)
            if target not in output.parents:
                raise RuntimeError("Verified restore entry escaped the new workspace.")
            digest = str(raw_entry["sha256"]).casefold()
            size = int(raw_entry["size"])
            if schema == _MANIFEST_SCHEMA:
                object_relative = PurePosixPath(str(raw_entry["object_path"]))
                source = (backup_root / Path(*object_relative.parts)).resolve(strict=False)
            else:
                source = (backup_root / Path(*logical.parts)).resolve(strict=False)
            if backup_root not in source.parents or source.is_symlink():
                raise RuntimeError("Verified restore source escaped the backup root.")
            output.parent.mkdir(parents=True, exist_ok=True)
            if output.exists():
                raise RuntimeError("Verified restore inventory contains a path collision.")
            self._materialize_object(source, output, digest, size)
        return target


def backup_once(manager: WorkspaceManager | None = None) -> BackupResult:
    return BackupService(manager or WorkspaceManager.from_environment()).run()


def cli(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Daedalus safe incremental backup")
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--status", action="store_true", help="show backup and scheduler health")
    actions.add_argument("--verify", action="store_true", help="verify the latest backup inventory")
    actions.add_argument(
        "--restore-to", type=Path, help="restore the workspace into a new explicit directory"
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    manager = WorkspaceManager.from_environment()
    service = BackupService(manager)
    if args.status:
        status = service.latest_status()
        print(json.dumps(status or {"status": "no backup yet"}, indent=2))
        health = status.get("health") if isinstance(status, dict) else None
        return 2 if isinstance(health, dict) and health.get("state") == "failed" else 0
    if args.verify:
        result = service.verify()
        print(json.dumps(asdict(result), indent=2))
        return 0 if result.ok else 2
    if args.restore_to:
        print(service.restore_workspace(args.restore_to))
        return 0
    try:
        result = service.run()
    except Exception as exc:
        print(f"Backup failed safely ({type(exc).__name__}).", file=sys.stderr)
        return 2
    print(json.dumps(asdict(result), indent=2))
    return 0 if result.ok else 2


if __name__ == "__main__":
    raise SystemExit(cli())
