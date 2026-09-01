"""Read-only recovery inventory and safe, non-overwriting restore proposals."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

from daedalus.developer.models import ToolIntent, ToolKey

if TYPE_CHECKING:
    from daedalus.developer.store import DeveloperSessionStore


DEFAULT_MAX_BACKUP_AGE = timedelta(hours=24)
MAX_BACKUP_CLOCK_SKEW = timedelta(minutes=5)


@dataclass(frozen=True, slots=True)
class RecoveryInventory:
    project_name: str
    project_present: bool
    session_present: bool
    session_revision: int | None
    session_revision_count: int
    run_count: int
    completed_run_count: int
    run_inventory_complete: bool
    checkpoint_count: int
    valid_checkpoint_count: int
    checkpoint_inventory_complete: bool
    backup_manifest_present: bool
    backup_verified: bool
    backup_file_count: int
    backup_finished_utc: str | None
    findings: tuple[str, ...]

    @property
    def ready(self) -> bool:
        return (
            self.project_present
            and self.session_present
            and self.run_inventory_complete
            and self.checkpoint_inventory_complete
            and self.backup_verified
        )


@dataclass(frozen=True, slots=True)
class RecoveryProposal:
    source_label: str
    destination: str
    verified_files: int
    mode: str = "new-directory-only"
    requires_confirmation: bool = True


def validate_restore_destination(destination: Path, protected_roots: tuple[Path, ...]) -> Path:
    """Return a safe nonexistent restore path or fail without creating anything."""

    original = Path(destination)
    if not original.is_absolute():
        raise ValueError("restore destination must be an absolute path")
    if original.exists() or original.is_symlink():
        raise FileExistsError("restore destination already exists; choose a new directory")
    target = original.resolve(strict=False)
    existing_parent = target.parent
    while not existing_parent.exists() and existing_parent != existing_parent.parent:
        existing_parent = existing_parent.parent
    if not existing_parent.is_dir() or existing_parent.is_symlink():
        raise PermissionError("restore destination does not have a safe existing parent")
    for root in protected_roots:
        protected = Path(root).resolve(strict=False)
        if target == protected or target in protected.parents or protected in target.parents:
            raise PermissionError("restore destination overlaps a protected active or backup path")
    return target


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _has_link_component(path: Path, root: Path) -> bool:
    """Reject symlink/junction indirection anywhere below a custody root."""

    relative = path.relative_to(root)
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            return True
        is_junction = getattr(current, "is_junction", None)
        if callable(is_junction) and is_junction():
            return True
    return False


class RecoveryPlanner:
    """Inventory recoverable evidence and hand a safe proposal to Vault & Backup.

    This class intentionally does not copy or delete files.  The existing backup
    service owns the explicit, user-confirmed restore operation; the proposal
    guarantees that the supplied target is new and separate first.
    """

    def __init__(
        self,
        project_root: Path,
        workspace_root: Path,
        backup_root: Path,
        *,
        source_root: Path | None = None,
        max_backup_age: timedelta = DEFAULT_MAX_BACKUP_AGE,
    ) -> None:
        self.project_root = Path(project_root).resolve(strict=False)
        self.workspace_root = Path(workspace_root).resolve(strict=False)
        self.backup_root = Path(backup_root).resolve(strict=False)
        self.source_root = Path(source_root).resolve(strict=False) if source_root else None
        for original in (project_root, workspace_root, backup_root, source_root):
            if original is not None and Path(original).is_symlink():
                raise PermissionError("recovery roots cannot be symbolic links")
        try:
            self.project_relative = self.project_root.relative_to(self.workspace_root)
        except ValueError as exc:
            raise PermissionError("recovery project escapes the active workspace") from exc
        if not self.project_relative.parts:
            raise PermissionError("recovery requires a selected project, not the workspace root")
        if max_backup_age <= timedelta(0):
            raise ValueError("maximum backup age must be positive")
        self.max_backup_age = max_backup_age

    def _project_name(self) -> str:
        manifest = self.project_root / "project.json"
        try:
            raw = json.loads(manifest.read_text(encoding="utf-8"))
            if isinstance(raw, dict) and str(raw.get("name", "")).strip():
                return str(raw["name"]).strip()
        except (OSError, UnicodeError, ValueError, json.JSONDecodeError):
            pass
        return self.project_root.name

    def _runs(self, project_name: str, findings: list[str]) -> tuple[int, int, bool]:
        count = completed = 0
        complete = True
        database = self.workspace_root / "training-runs" / "runs.sqlite3"
        if database.is_file() and not database.is_symlink():
            try:
                uri = database.as_uri() + "?mode=ro"
                with closing(sqlite3.connect(uri, uri=True, timeout=2)) as connection:
                    rows = connection.execute(
                        "SELECT status FROM runs WHERE project=?", (project_name,)
                    ).fetchall()
                count += len(rows)
                completed += sum(str(row[0]) == "completed" for row in rows)
            except (sqlite3.DatabaseError, OSError):
                complete = False
                findings.append("The workspace run registry could not be inventoried read-only.")
        local_runs = self.project_root / "runs"
        for path in sorted(local_runs.glob("*.json"))[:500]:
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, ValueError, json.JSONDecodeError):
                complete = False
                continue
            if isinstance(raw, dict):
                count += 1
                completed += raw.get("status") == "completed"
        return count, completed, complete

    def _checkpoints(
        self, project_name: str, findings: list[str]
    ) -> tuple[int, int, bool]:
        directories = (
            self.workspace_root / "checkpoints" / project_name,
            self.project_root / "checkpoints",
        )
        count = valid = 0
        complete = True
        seen: set[Path] = set()
        for directory in directories:
            if not directory.is_dir():
                continue
            if directory.is_symlink():
                complete = False
                findings.append("A checkpoint directory is a symbolic link and was skipped.")
                continue
            for metadata in sorted(directory.glob("*.json"))[:500]:
                resolved = metadata.resolve(strict=False)
                if resolved in seen:
                    continue
                seen.add(resolved)
                count += 1
                try:
                    if metadata.is_symlink():
                        raise ValueError("checkpoint metadata is a symbolic link")
                    raw = json.loads(metadata.read_text(encoding="utf-8"))
                    if not isinstance(raw, dict):
                        raise ValueError("checkpoint metadata is not an object")
                    arrays = (metadata.parent / str(raw["array_file"])).resolve(strict=False)
                    arrays.relative_to(metadata.parent.resolve(strict=False))
                    if (
                        raw.get("format") != "daedalus-npz"
                        or raw.get("schema") not in {1, 2}
                        or arrays.is_symlink()
                        or not arrays.is_file()
                        or _sha256(arrays) != raw.get("sha256")
                    ):
                        raise ValueError("checkpoint did not verify")
                    valid += 1
                except (KeyError, OSError, UnicodeError, ValueError, json.JSONDecodeError):
                    complete = False
                    findings.append(f"Checkpoint metadata did not verify: {metadata.name}")
        return count, valid, complete

    def _backup(self, findings: list[str]) -> tuple[bool, bool, int, str | None]:
        marker = self.backup_root / ".daedalus-backup-root.json"
        latest = self.backup_root / "latest.json"
        manifest_present = marker.is_file() and latest.is_file()
        try:
            if marker.is_symlink() or latest.is_symlink():
                raise ValueError("backup metadata cannot be a symbolic link")
            if marker.stat().st_size > 64 * 1024 or latest.stat().st_size > 16 * 1024 * 1024:
                raise ValueError("backup metadata exceeds the inspection limit")
            marker_raw = json.loads(marker.read_text(encoding="utf-8"))
            marker_ok = (
                isinstance(marker_raw, dict)
                and marker_raw.get("kind") == "daedalus-backup-root"
                and marker_raw.get("schema") == 1
            )
            raw = json.loads(latest.read_text(encoding="utf-8"))
            if not marker_ok or not isinstance(raw, dict):
                raise ValueError("backup metadata is invalid")
        except (OSError, UnicodeError, ValueError, json.JSONDecodeError):
            findings.append("No readable marked backup manifest is available.")
            return manifest_present, False, 0, None
        finished_text = str(raw.get("finished_utc") or "") or None
        raw_schema = raw.get("schema")
        if (
            raw.get("kind") != "daedalus-backup-manifest"
            or type(raw_schema) is not int
            or raw_schema not in {2, 3}
        ):
            findings.append(
                "The latest backup is not a supported schema-2 or schema-3 Daedalus manifest."
            )
            return True, False, 0, finished_text
        schema = raw_schema
        try:
            finished = datetime.fromisoformat(finished_text or "")
            if finished.tzinfo is None:
                raise ValueError("backup timestamp has no timezone")
            finished = finished.astimezone(UTC)
            now = datetime.now(UTC)
            if finished > now + MAX_BACKUP_CLOCK_SKEW:
                raise ValueError("backup timestamp is implausibly in the future")
            if now - finished > self.max_backup_age:
                raise ValueError("backup is stale")
        except ValueError:
            findings.append(
                "The latest backup timestamp is invalid, in the future, or outside the configured freshness window."
            )
            return True, False, 0, finished_text
        inventory = raw.get("inventory")
        errors = raw.get("errors")
        if not isinstance(errors, list) or errors or not isinstance(inventory, list):
            findings.append("The latest backup has errors or lacks a verifiable file inventory.")
            return True, False, 0, finished_text
        if len(inventory) > 100_000:
            findings.append("The backup inventory exceeds the bounded verification limit.")
            return True, False, 0, finished_text
        if not inventory:
            findings.append("The backup inventory has no files to restore.")
            return True, False, 0, finished_text
        if schema == 3 and (
            type(raw.get("files_scanned")) is not int
            or raw["files_scanned"] != len(inventory)
        ):
            findings.append("The schema-3 inventory count does not match files_scanned.")
            return True, False, 0, finished_text
        verified = 0
        seen: set[str] = set()
        verified_sources: dict[str, Path] = {}
        root = self.backup_root.resolve(strict=False)
        for item in inventory:
            try:
                if not isinstance(item, dict):
                    raise ValueError("inventory item is invalid")
                relative = str(item["path"])
                pure = PurePosixPath(relative)
                if (
                    not relative
                    or "\\" in relative
                    or pure.is_absolute()
                    or ".." in pure.parts
                    or len(pure.parts) < 2
                    or pure.parts[0] not in {"workspace-current", "source-current"}
                    or pure.as_posix() != relative
                ):
                    raise ValueError("inventory path is unsafe")
                identity = relative.casefold()
                if identity in seen:
                    raise ValueError("inventory path is duplicated")
                expected_size = item["size"]
                expected_digest = item["sha256"]
                if (
                    type(expected_size) is not int
                    or expected_size < 0
                    or not isinstance(expected_digest, str)
                    or len(expected_digest) != 64
                    or any(character not in "0123456789abcdef" for character in expected_digest)
                ):
                    raise ValueError("inventory size or digest is invalid")
                entry_kind = item.get("kind", "file")
                if entry_kind not in {"file", "sqlite"}:
                    raise ValueError("inventory kind is invalid")
                if schema == 3:
                    object_text = item.get("object_path")
                    expected_object = PurePosixPath(
                        "objects", "sha256", expected_digest[:2], expected_digest
                    ).as_posix()
                    if (
                        not isinstance(object_text, str)
                        or object_text != expected_object
                        or "\\" in object_text
                    ):
                        raise ValueError("content-object path is invalid")
                    source_candidate = root / Path(*PurePosixPath(object_text).parts)
                else:
                    source_candidate = root / Path(*pure.parts)
                if _has_link_component(source_candidate, root):
                    raise ValueError("inventory source contains a link or junction")
                source = source_candidate.resolve(strict=False)
                source.relative_to(root)
                if (
                    source.is_symlink()
                    or not source.is_file()
                    or source.stat().st_size != expected_size
                    or _sha256(source) != expected_digest
                ):
                    raise ValueError("inventory content mismatch")
                seen.add(identity)
                verified_sources[identity] = source
                verified += 1
            except (KeyError, OSError, TypeError, ValueError):
                findings.append("At least one backup inventory entry failed verification.")
                return True, False, verified, finished_text
        workspace_entries = [path for path in seen if path.startswith("workspace-current/")]
        if not workspace_entries:
            findings.append("The backup inventory has no workspace files to restore.")
            return True, False, verified, finished_text

        workspace_marker_relative = "workspace-current/.daedalus-workspace.json"
        project_manifest_relative = PurePosixPath(
            "workspace-current", *self.project_relative.parts, "project.json"
        ).as_posix()
        workspace_marker_key = workspace_marker_relative.casefold()
        project_manifest_key = project_manifest_relative.casefold()
        if workspace_marker_key not in seen:
            findings.append("The backup does not inventory the active workspace identity marker.")
            return True, False, verified, finished_text
        if project_manifest_key not in seen:
            findings.append("The selected project is absent from the latest workspace backup.")
            return True, False, verified, finished_text

        active_marker = self.workspace_root / ".daedalus-workspace.json"
        backed_marker = verified_sources[workspace_marker_key]
        backed_project = verified_sources[project_manifest_key]
        try:
            if active_marker.is_symlink() or active_marker.stat().st_size > 64 * 1024:
                raise ValueError("active workspace marker is invalid")
            if backed_marker.stat().st_size > 64 * 1024 or backed_project.stat().st_size > 2 * 1024 * 1024:
                raise ValueError("backed identity metadata exceeds the inspection limit")
            active_raw = json.loads(active_marker.read_text(encoding="utf-8"))
            backed_raw = json.loads(backed_marker.read_text(encoding="utf-8"))
            project_raw = json.loads(backed_project.read_text(encoding="utf-8"))
            active_id = active_raw.get("id") if isinstance(active_raw, dict) else None
            backed_id = backed_raw.get("id") if isinstance(backed_raw, dict) else None
            identities_valid = (
                isinstance(active_raw, dict)
                and active_raw.get("kind") == "daedalus-user-workspace"
                and active_raw.get("schema") == 1
                and isinstance(active_id, str)
                and bool(active_id.strip())
                and isinstance(backed_raw, dict)
                and backed_raw.get("kind") == "daedalus-user-workspace"
                and backed_raw.get("schema") == 1
                and backed_id == active_id
            )
            project_valid = (
                isinstance(project_raw, dict)
                and project_raw.get("schema") == 1
                and str(project_raw.get("name", "")).strip() == self._project_name()
            )
            if not identities_valid:
                raise ValueError("workspace identities do not match")
            if not project_valid:
                raise ValueError("backed project manifest does not match the selected project")
        except (OSError, UnicodeError, ValueError, json.JSONDecodeError):
            findings.append(
                "The backup workspace identity or selected project manifest does not match the active project."
            )
            return True, False, verified, finished_text
        return True, True, verified, finished_text

    def inventory(
        self,
        *,
        session_store: DeveloperSessionStore | None = None,
        session_id: str | None = None,
    ) -> RecoveryInventory:
        findings: list[str] = []
        session_present = False
        session_revision: int | None = None
        revision_count = 0
        if session_store is not None and session_id is not None:
            try:
                session = session_store.load(session_id)
                session_present = True
                session_revision = session.revision
                revision_count = len(session_store.history(session_id))
            except (KeyError, RuntimeError, ValueError):
                findings.append("The developer session could not be loaded; use last-valid recovery.")
        project_name = self._project_name()
        run_count, completed_runs, run_complete = self._runs(project_name, findings)
        checkpoint_count, valid_checkpoints, checkpoint_complete = self._checkpoints(
            project_name, findings
        )
        manifest, verified, backup_files, finished = self._backup(findings)
        return RecoveryInventory(
            project_name,
            self.project_root.is_dir(),
            session_present,
            session_revision,
            revision_count,
            run_count,
            completed_runs,
            run_complete,
            checkpoint_count,
            valid_checkpoints,
            checkpoint_complete,
            manifest,
            verified,
            backup_files,
            finished,
            tuple(findings),
        )

    def propose_restore(
        self, destination: Path, inventory: RecoveryInventory
    ) -> RecoveryProposal:
        if not inventory.backup_verified:
            raise ValueError("restore cannot be proposed until the backup inventory verifies")
        protected = [self.workspace_root, self.project_root, self.backup_root]
        if self.source_root is not None:
            protected.append(self.source_root)
        target = validate_restore_destination(destination, tuple(protected))
        return RecoveryProposal(
            "verified workspace-current snapshot",
            str(target),
            inventory.backup_file_count,
        )

    @staticmethod
    def tool_intent(proposal: RecoveryProposal) -> ToolIntent:
        return ToolIntent(
            ToolKey.VAULT,
            "Review safe restore",
            "Vault & Backup must re-verify and copy into the proposed new destination after explicit confirmation.",
            {
                "destination": proposal.destination,
                "restore_mode": proposal.mode,
                "requires_confirmation": proposal.requires_confirmation,
                "verified_files": proposal.verified_files,
            },
        )


__all__ = [
    "DEFAULT_MAX_BACKUP_AGE",
    "RecoveryInventory",
    "RecoveryPlanner",
    "RecoveryProposal",
    "validate_restore_destination",
]
