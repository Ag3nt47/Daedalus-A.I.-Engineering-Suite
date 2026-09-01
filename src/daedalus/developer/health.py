"""Read-only project evidence and health assessment.

The inspector reads bounded metadata and text files.  It never repairs, writes,
executes code, imports a checkpoint, starts a process, or contacts a service.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from contextlib import closing
from dataclasses import dataclass, fields, replace
from enum import StrEnum
from pathlib import Path
from typing import Any

from daedalus.developer.models import DeveloperSession, ProjectEvidence
from daedalus.developer.recovery import (
    RecoveryInventory,
    RecoveryPlanner,
    validate_restore_destination,
)

MAX_INSPECTED_TEXT_BYTES = 2 * 1024 * 1024
MAX_INSPECTED_FILES = 2_000
MAX_INSPECTED_ENTRIES = 20_000
_TEXT_SUFFIXES = {".py", ".md", ".txt", ".json", ".toml", ".yaml", ".yml", ".ini"}
_SKIP_DIRECTORIES = {".git", ".venv", "venv", "__pycache__", "node_modules"}
_SECRET_PATTERNS = (
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----", re.IGNORECASE),
    re.compile(r"\bAKIA[A-Z0-9]{16}\b"),
    re.compile(r"\b(?:sk-[A-Za-z0-9_-]{12,}|ghp_[A-Za-z0-9]{12,})\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{12,}\b"),
)
_DRAFT_ARTIFACT_MARKERS = ("not yet recorded",)


class FindingSeverity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    BLOCKING = "blocking"


@dataclass(frozen=True, slots=True)
class HealthFinding:
    code: str
    severity: FindingSeverity
    summary: str
    location: str = ""


@dataclass(frozen=True, slots=True)
class HealthReport:
    evidence: ProjectEvidence
    findings: tuple[HealthFinding, ...]
    files_checked: int
    bytes_checked: int

    @property
    def ok(self) -> bool:
        return not any(item.severity == FindingSeverity.BLOCKING for item in self.findings)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_object(path: Path, maximum: int = MAX_INSPECTED_TEXT_BYTES) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("metadata path must be a regular non-symlink file")
    if path.stat().st_size > maximum:
        raise ValueError("metadata file exceeds the inspection limit")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("metadata must contain a JSON object")
    return value


def _merge_evidence(observed: ProjectEvidence, supplied: ProjectEvidence | None) -> ProjectEvidence:
    if supplied is None:
        return observed
    values = {}
    for item in fields(ProjectEvidence):
        observed_value = getattr(observed, item.name)
        supplied_value = getattr(supplied, item.name)
        values[item.name] = observed_value if observed_value is not None else supplied_value
    return ProjectEvidence(**values)


class ProjectHealthInspector:
    """Collect bounded, non-executable evidence for a selected private project."""

    def __init__(
        self,
        project_root: Path,
        workspace_root: Path,
        *,
        source_root: Path | None = None,
        backup_root: Path | None = None,
    ) -> None:
        original_workspace = Path(workspace_root)
        original_project = Path(project_root)
        if original_workspace.is_symlink() or original_project.is_symlink():
            raise PermissionError("health-inspection roots cannot be symbolic links")
        self.workspace_root = original_workspace.resolve(strict=False)
        self.project_root = original_project.resolve(strict=False)
        try:
            self.project_root.relative_to(self.workspace_root)
        except ValueError as exc:
            raise PermissionError("selected project escapes the private workspace") from exc
        self.source_root = Path(source_root).resolve(strict=False) if source_root else None
        self.backup_root = Path(backup_root).resolve(strict=False) if backup_root else None

    def _relative(self, path: Path) -> str:
        for label, root in (
            ("project", self.project_root),
            ("workspace", self.workspace_root),
            ("source", self.source_root),
            ("backup", self.backup_root),
        ):
            if root is None:
                continue
            try:
                relative = path.resolve(strict=False).relative_to(root)
                return f"{label}/{relative.as_posix()}" if relative.parts else label
            except ValueError:
                continue
        return "outside configured roots"

    def _workspace_status(self, findings: list[HealthFinding]) -> tuple[bool, bool]:
        marker = self.workspace_root / ".daedalus-workspace.json"
        workspace_ok = False
        try:
            raw = _load_object(marker)
            workspace_ok = raw.get("kind") == "daedalus-user-workspace" and raw.get("schema") == 1
        except (OSError, UnicodeError, ValueError, json.JSONDecodeError):
            pass
        if not workspace_ok:
            findings.append(
                HealthFinding(
                    "workspace.marker",
                    FindingSeverity.BLOCKING,
                    "Private workspace ownership marker is missing or invalid.",
                    self._relative(marker),
                )
            )

        manifest = self.project_root / "project.json"
        project_ok = False
        try:
            raw = _load_object(manifest)
            project_ok = raw.get("schema") == 1 and bool(str(raw.get("name", "")).strip())
        except (OSError, UnicodeError, ValueError, json.JSONDecodeError):
            pass
        if not project_ok:
            findings.append(
                HealthFinding(
                    "project.manifest",
                    FindingSeverity.BLOCKING,
                    "Project manifest is missing or invalid.",
                    self._relative(manifest),
                )
            )
        return workspace_ok, project_ok

    def _dataset_status(self, findings: list[HealthFinding]) -> tuple[bool, bool | None]:
        directory = self.workspace_root / "datasets"
        metadata_files = sorted(directory.glob("*.dataset.json")) if directory.is_dir() else []
        if not metadata_files:
            return False, None
        all_valid = True
        for metadata_path in metadata_files[:200]:
            try:
                raw = _load_object(metadata_path)
                filename = raw.get("file")
                digest = raw.get("sha256")
                if not isinstance(filename, str) or not isinstance(digest, str):
                    raise ValueError("dataset metadata is incomplete")
                dataset = (metadata_path.parent / filename).resolve(strict=False)
                dataset.relative_to(directory.resolve(strict=False))
                if dataset.is_symlink() or not dataset.is_file() or _sha256(dataset) != digest:
                    raise ValueError("dataset checksum is invalid")
            except (OSError, UnicodeError, ValueError, json.JSONDecodeError):
                all_valid = False
                findings.append(
                    HealthFinding(
                        "dataset.integrity",
                        FindingSeverity.BLOCKING,
                        "A registered dataset failed its path, metadata, or checksum check.",
                        self._relative(metadata_path),
                    )
                )
        if len(metadata_files) > 200:
            all_valid = False
            findings.append(
                HealthFinding(
                    "dataset.limit",
                    FindingSeverity.WARNING,
                    "More than 200 dataset records exist; the bounded health pass did not inspect all of them.",
                    self._relative(directory),
                )
            )
        return True, all_valid

    def _run_status(
        self, session: DeveloperSession | None, findings: list[HealthFinding]
    ) -> tuple[bool | None, bool | None, bool]:
        run_files = sorted((self.project_root / "runs").glob("*.json"))
        completed = False
        heldout = False
        found = False
        complete = True
        for run_path in run_files[:200]:
            try:
                raw = _load_object(run_path)
            except (OSError, UnicodeError, ValueError, json.JSONDecodeError):
                findings.append(
                    HealthFinding(
                        "run.metadata",
                        FindingSeverity.WARNING,
                        "A run record is unreadable or invalid JSON.",
                        self._relative(run_path),
                    )
                )
                complete = False
                continue
            found = True
            if raw.get("status") == "completed":
                completed = True
                metrics = raw.get("heldout_metrics")
                heldout = heldout or (isinstance(metrics, dict) and bool(metrics))
        database = self.workspace_root / "training-runs" / "runs.sqlite3"
        if session is not None and database.is_file() and not database.is_symlink():
            try:
                with closing(
                    sqlite3.connect(database.as_uri() + "?mode=ro", uri=True, timeout=2)
                ) as connection:
                    rows = connection.execute(
                        "SELECT status, metrics_json FROM runs WHERE project=?",
                        (session.brief.project_name,),
                    ).fetchall()
                found = found or bool(rows)
                for status, metrics_json in rows:
                    if str(status) != "completed":
                        continue
                    completed = True
                    metrics = json.loads(metrics_json)
                    if isinstance(metrics, dict):
                        heldout = heldout or any(
                            str(name).casefold().startswith(("val_", "test_", "holdout"))
                            for name in metrics
                        )
            except (
                sqlite3.DatabaseError,
                OSError,
                TypeError,
                ValueError,
                json.JSONDecodeError,
            ):
                complete = False
                findings.append(
                    HealthFinding(
                        "run.registry",
                        FindingSeverity.WARNING,
                        "The workspace run registry could not be inspected read-only.",
                        self._relative(database),
                    )
                )
        if len(run_files) > 200:
            complete = False
        if found:
            return completed, heldout, complete
        return None, None, complete

    def _checkpoint_status(
        self, session: DeveloperSession | None, findings: list[HealthFinding]
    ) -> tuple[bool | None, bool]:
        directories = [self.project_root / "checkpoints"]
        if session is not None:
            directories.insert(
                0, self.workspace_root / "checkpoints" / session.brief.project_name
            )
        metadata_files: list[Path] = []
        for directory in directories:
            metadata_files.extend(sorted(directory.glob("*.json")))
        metadata_files = list(dict.fromkeys(path.resolve(strict=False) for path in metadata_files))
        if not metadata_files:
            return None, True
        all_valid = True
        for metadata_path in metadata_files[:100]:
            try:
                raw = _load_object(metadata_path)
                if raw.get("format") != "daedalus-npz" or raw.get("schema") not in {1, 2}:
                    raise ValueError("unsupported checkpoint metadata")
                array_name = raw.get("array_file")
                expected = raw.get("sha256")
                if not isinstance(array_name, str) or not isinstance(expected, str):
                    raise ValueError("checkpoint metadata is incomplete")
                arrays = (metadata_path.parent / array_name).resolve(strict=False)
                arrays.relative_to(metadata_path.parent.resolve(strict=False))
                if arrays.is_symlink() or not arrays.is_file() or _sha256(arrays) != expected:
                    raise ValueError("checkpoint checksum is invalid")
            except (OSError, UnicodeError, ValueError, json.JSONDecodeError):
                all_valid = False
                findings.append(
                    HealthFinding(
                        "checkpoint.integrity",
                        FindingSeverity.BLOCKING,
                        "A checkpoint failed its format, path, or checksum check.",
                        self._relative(metadata_path),
                    )
                )
        if len(metadata_files) > 100:
            all_valid = False
        return all_valid, all_valid

    def _declared_evidence(self, findings: list[HealthFinding]) -> ProjectEvidence:
        path = self.project_root / "DAEDALUS_EVIDENCE.json"
        if not path.exists():
            return ProjectEvidence()
        allowed = {
            "schema_version",
            "architecture_validated",
            "dependency_scan_passed",
            "backup_current",
            "release_guard_passed",
        }
        try:
            raw = _load_object(path)
            if set(raw) - allowed or raw.get("schema_version") != 1:
                raise ValueError("unsupported evidence fields")
            values = {key: raw.get(key) for key in allowed - {"schema_version"}}
            if any(value is not None and not isinstance(value, bool) for value in values.values()):
                raise ValueError("evidence values must be booleans")
            return ProjectEvidence(**values)
        except (OSError, UnicodeError, ValueError, json.JSONDecodeError):
            findings.append(
                HealthFinding(
                    "evidence.declaration",
                    FindingSeverity.WARNING,
                    "DAEDALUS_EVIDENCE.json is invalid and was not trusted.",
                    self._relative(path),
                )
            )
            return ProjectEvidence()

    def _recovery_inventory(
        self, findings: list[HealthFinding]
    ) -> RecoveryInventory | None:
        """Use the recovery verifier as the sole source of backup-current evidence."""

        if self.backup_root is None:
            return None
        inventory = RecoveryPlanner(
            self.project_root,
            self.workspace_root,
            self.backup_root,
            source_root=self.source_root,
        ).inventory()
        if not inventory.backup_verified:
            findings.append(
                HealthFinding(
                    "backup.verification",
                    FindingSeverity.WARNING,
                    "The latest backup failed schema, freshness, workspace identity, selected-project, or content verification.",
                    self._relative(self.backup_root / "latest.json"),
                )
            )
        return inventory

    def _source_status(self, findings: list[HealthFinding]) -> None:
        if self.source_root is None:
            return
        manager_source = self.source_root / "src" / "daedalus" / "workspace" / "manager.py"
        if not manager_source.is_file():
            findings.append(
                HealthFinding(
                    "source.workspace-package",
                    FindingSeverity.BLOCKING,
                    "The required workspace manager source file is missing.",
                    self._relative(manager_source),
                )
            )
        ignore = self.source_root / ".gitignore"
        try:
            lines = ignore.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeError):
            return
        dangerous = {
            line.strip()
            for line in lines
            if line.strip() in {"workspace/", "datasets/", "models/", "logs/"}
        }
        if "workspace/" in dangerous and manager_source.is_file():
            findings.append(
                HealthFinding(
                    "source.ignore-shadow",
                    FindingSeverity.BLOCKING,
                    "An unanchored workspace/ ignore rule can exclude the required source package; use /workspace/ for repository-root private data.",
                    self._relative(ignore),
                )
            )

    def _artifact_complete(
        self,
        filename: str,
        label: str,
        findings: list[HealthFinding],
    ) -> bool:
        """Accept a planning artifact only when it is readable and not a draft."""

        path = self.project_root / filename
        if not path.exists() and not path.is_symlink():
            return False
        try:
            if path.is_symlink() or not path.is_file():
                raise ValueError("artifact is not a regular file")
            if path.stat().st_size > MAX_INSPECTED_TEXT_BYTES:
                raise ValueError("artifact exceeds the bounded inspection limit")
            content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError, ValueError):
            findings.append(
                HealthFinding(
                    "artifact.invalid",
                    FindingSeverity.WARNING,
                    f"The {label} exists but could not be validated as a regular UTF-8 planning artifact.",
                    self._relative(path),
                )
            )
            return False
        normalized = content.casefold()
        if not content.strip() or any(marker in normalized for marker in _DRAFT_ARTIFACT_MARKERS):
            findings.append(
                HealthFinding(
                    "artifact.draft",
                    FindingSeverity.WARNING,
                    f"The {label} is still a draft with unresolved placeholders.",
                    self._relative(path),
                )
            )
            return False
        return True

    def _secret_status(self, findings: list[HealthFinding]) -> tuple[bool, int, int]:
        files_checked = 0
        bytes_checked = 0
        entries_checked = 0
        complete = True
        stack = [self.project_root]
        while (
            stack
            and files_checked < MAX_INSPECTED_FILES
            and entries_checked < MAX_INSPECTED_ENTRIES
        ):
            directory = stack.pop()
            try:
                entries = sorted(directory.iterdir(), key=lambda item: item.name.casefold())
            except OSError:
                complete = False
                continue
            for entry in entries:
                entries_checked += 1
                if entries_checked > MAX_INSPECTED_ENTRIES:
                    complete = False
                    break
                if entry.is_symlink():
                    complete = False
                    findings.append(
                        HealthFinding(
                            "project.symlink",
                            FindingSeverity.WARNING,
                            "A symbolic link was skipped by the bounded project scan.",
                            self._relative(entry),
                        )
                    )
                    continue
                if entry.is_dir():
                    if entry.name not in _SKIP_DIRECTORIES:
                        stack.append(entry)
                    continue
                if entry.suffix.casefold() not in _TEXT_SUFFIXES:
                    continue
                try:
                    size = entry.stat().st_size
                    if size > MAX_INSPECTED_TEXT_BYTES:
                        complete = False
                        findings.append(
                            HealthFinding(
                                "project.scan-limit",
                                FindingSeverity.WARNING,
                                "A text-like file exceeded the bounded secret-scan limit.",
                                self._relative(entry),
                            )
                        )
                        continue
                    content = entry.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    complete = False
                    continue
                files_checked += 1
                bytes_checked += size
                if any(pattern.search(content) for pattern in _SECRET_PATTERNS):
                    complete = False
                    findings.append(
                        HealthFinding(
                            "project.possible-secret",
                            FindingSeverity.BLOCKING,
                            "Possible credential material was detected; the value is intentionally redacted.",
                            self._relative(entry),
                        )
                    )
                if files_checked >= MAX_INSPECTED_FILES:
                    break
        if stack or entries_checked > MAX_INSPECTED_ENTRIES:
            complete = False
            findings.append(
                HealthFinding(
                    "project.file-limit",
                    FindingSeverity.WARNING,
                    "The project contains more files than the bounded health scan can inspect.",
                    "project",
                )
            )
        return complete, files_checked, bytes_checked

    def inspect(
        self,
        session: DeveloperSession | None = None,
        *,
        supplied_evidence: ProjectEvidence | None = None,
    ) -> HealthReport:
        findings: list[HealthFinding] = []
        if session is not None and Path(session.project_root).resolve(strict=False) != self.project_root:
            raise PermissionError("developer session belongs to a different project")
        workspace_ok, project_ok = self._workspace_status(findings)
        dataset_present, dataset_integrity = self._dataset_status(findings)
        run_completed, heldout, run_inventory_complete = self._run_status(session, findings)
        checkpoint, checkpoint_inventory_complete = self._checkpoint_status(session, findings)
        declared = self._declared_evidence(findings)
        recovery_inventory = self._recovery_inventory(findings)
        backup = (
            recovery_inventory.backup_verified
            if recovery_inventory is not None
            else None
        )
        self._source_status(findings)
        secret_passed, files_checked, bytes_checked = self._secret_status(findings)

        split_documented = None
        baseline_recorded = None
        if session is not None:
            split_documented = bool(str(session.answers.get("split_strategy", "")).strip())
        baseline_file = self.project_root / "runs" / "baseline.json"
        if baseline_file.exists():
            try:
                baseline_raw = _load_object(baseline_file)
                baseline_recorded = bool(baseline_raw.get("metric")) and isinstance(
                    baseline_raw.get("value"), (int, float)
                )
            except (OSError, UnicodeError, ValueError, json.JSONDecodeError):
                baseline_recorded = False
                findings.append(
                    HealthFinding(
                        "baseline.record",
                        FindingSeverity.WARNING,
                        "The baseline result record is invalid.",
                        self._relative(baseline_file),
                    )
                )

        restore_target_safe = None
        if session is not None and session.answers.get("restore_destination"):
            protected = [self.workspace_root, self.project_root]
            if self.backup_root is not None:
                protected.append(self.backup_root)
            if self.source_root is not None:
                protected.append(self.source_root)
            try:
                validate_restore_destination(
                    Path(str(session.answers["restore_destination"])), tuple(protected)
                )
                restore_target_safe = True
            except (OSError, PermissionError, ValueError):
                restore_target_safe = False

        experiment_plan_present = self._artifact_complete(
            "EXPERIMENT_PLAN.md", "experiment plan", findings
        )
        model_card_present = self._artifact_complete("MODEL_CARD.md", "model card", findings)
        deployment_plan_present = self._artifact_complete(
            "DEPLOYMENT_RUNBOOK.md", "deployment runbook", findings
        )
        threat_model_present = self._artifact_complete(
            "THREAT_MODEL.md", "threat model", findings
        )

        observed = ProjectEvidence(
            workspace_ready=workspace_ok,
            project_manifest_valid=project_ok,
            session_inventory_complete=session is not None,
            run_inventory_complete=(
                (self.workspace_root / "training-runs").is_dir()
                and run_inventory_complete
                and (
                    recovery_inventory is None
                    or recovery_inventory.run_inventory_complete
                )
            ),
            checkpoint_inventory_complete=(
                (self.workspace_root / "checkpoints").is_dir()
                and checkpoint_inventory_complete
                and (
                    recovery_inventory is None
                    or recovery_inventory.checkpoint_inventory_complete
                )
            ),
            restore_target_safe=restore_target_safe,
            dataset_present=dataset_present,
            dataset_integrity=dataset_integrity,
            split_documented=split_documented,
            baseline_recorded=baseline_recorded,
            architecture_validated=declared.architecture_validated,
            experiment_plan_present=experiment_plan_present,
            run_completed=run_completed,
            heldout_metrics_present=heldout,
            checkpoint_valid=checkpoint,
            model_card_present=model_card_present,
            deployment_plan_present=deployment_plan_present,
            threat_model_present=threat_model_present,
            secret_scan_passed=secret_passed,
            dependency_scan_passed=declared.dependency_scan_passed,
            backup_current=backup if backup is not None else declared.backup_current,
            release_guard_passed=declared.release_guard_passed,
        )
        merged = _merge_evidence(observed, supplied_evidence)
        if backup is not None:
            merged = replace(merged, backup_current=backup)
        return HealthReport(merged, tuple(findings), files_checked, bytes_checked)


__all__ = [
    "FindingSeverity",
    "HealthFinding",
    "HealthReport",
    "ProjectHealthInspector",
]
