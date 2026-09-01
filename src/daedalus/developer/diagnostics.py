"""Bounded, read-only diagnostics for a selected private AI project.

The scanner deliberately parses text and metadata without importing project
modules, loading executable formats, starting processes, or changing files.
Log matches are reported by category and location only so a diagnostic report
does not repeat credentials or other private values from a log line.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any, Iterable

from daedalus.developer.health import FindingSeverity, ProjectHealthInspector

_SKIP_DIRECTORIES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "venv",
}
_PROJECT_LOG_SUFFIXES = {".err", ".log", ".out"}
_LOG_DIRECTORY_NAMES = {"log", "logs", "run", "runs"}
_LOG_MARKERS: tuple[tuple[str, re.Pattern[str], "DiagnosticSeverity"], ...] = ()


class DiagnosticSeverity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


_LOG_MARKERS = (
    (
        "traceback",
        re.compile(r"\btraceback\b", re.IGNORECASE),
        DiagnosticSeverity.ERROR,
    ),
    (
        "critical or fatal error",
        re.compile(r"\b(?:critical|fatal)\b", re.IGNORECASE),
        DiagnosticSeverity.ERROR,
    ),
    (
        "exception",
        re.compile(r"\bexception\b", re.IGNORECASE),
        DiagnosticSeverity.ERROR,
    ),
    (
        "error",
        re.compile(r"\berrors?\b", re.IGNORECASE),
        DiagnosticSeverity.ERROR,
    ),
    (
        "failed operation",
        re.compile(r"\b(?:failed|failure)\b", re.IGNORECASE),
        DiagnosticSeverity.WARNING,
    ),
    (
        "non-finite value",
        re.compile(r"\b(?:nan|infinity)\b", re.IGNORECASE),
        DiagnosticSeverity.WARNING,
    ),
)
_EMPTY_ERROR_VALUE = re.compile(
    r"[\"']?error[\"']?\s*[:=]\s*(?:null|none|false|[\"']{2}|\[\]|\{\})\s*[,}]?\s*$",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class DiagnosticFinding:
    code: str
    severity: DiagnosticSeverity
    summary: str
    location: str = ""
    suggestion: str = ""


@dataclass(frozen=True, slots=True)
class ProjectDiagnosticReport:
    project_name: str
    project_root: str
    findings: tuple[DiagnosticFinding, ...]
    files_scanned: int
    log_files_scanned: int
    lines_scanned: int
    bytes_scanned: int
    truncated: bool
    generated_utc: str

    @property
    def error_count(self) -> int:
        return sum(item.severity == DiagnosticSeverity.ERROR for item in self.findings)

    @property
    def warning_count(self) -> int:
        return sum(item.severity == DiagnosticSeverity.WARNING for item in self.findings)

    @property
    def info_count(self) -> int:
        return sum(item.severity == DiagnosticSeverity.INFO for item in self.findings)

    @property
    def ok(self) -> bool:
        return self.error_count == 0

    def format_text(self) -> str:
        status = (
            "No errors found in the bounded scan."
            if not self.findings
            else (
                f"Found {self.error_count} error(s), {self.warning_count} warning(s), "
                f"and {self.info_count} informational item(s)."
            )
        )
        lines = [
            f"Project diagnostics — {self.project_name}",
            status,
            (
                f"Read-only scope: {self.files_scanned} file(s), "
                f"{self.log_files_scanned} log file(s), {self.lines_scanned} log line(s), "
                f"{self.bytes_scanned:,} byte(s)."
            ),
        ]
        if self.truncated:
            lines.append(
                "The safety limits were reached, so this report is partial. Narrow large logs or "
                "remove generated folders, then scan again."
            )
        lines.append(
            "Project code was parsed, never imported or executed. Log values are not copied into "
            "this report."
        )
        rank = {
            DiagnosticSeverity.ERROR: 0,
            DiagnosticSeverity.WARNING: 1,
            DiagnosticSeverity.INFO: 2,
        }
        for finding in sorted(
            self.findings,
            key=lambda item: (rank[item.severity], item.location.casefold(), item.code),
        ):
            lines.extend(("", f"[{finding.severity.value.upper()}] {finding.summary}"))
            if finding.location:
                lines.append(f"Location: {finding.location}")
            if finding.suggestion:
                lines.append(f"Suggested next step: {finding.suggestion}")
        return "\n".join(lines)


@dataclass(slots=True)
class _ScanState:
    maximum_findings: int
    findings: list[DiagnosticFinding] = field(default_factory=list)
    seen: set[tuple[str, str, str]] = field(default_factory=set)
    files_scanned: int = 0
    log_files_scanned: int = 0
    lines_scanned: int = 0
    bytes_scanned: int = 0
    budget_bytes: int = 0
    truncated: bool = False

    def add(self, finding: DiagnosticFinding) -> None:
        identity = (finding.code, finding.location, finding.summary)
        if identity in self.seen:
            return
        if len(self.findings) >= self.maximum_findings:
            self.truncated = True
            return
        self.seen.add(identity)
        self.findings.append(finding)


class ProjectDiagnosticsScanner:
    """Inspect one project, related run records, logs, data, and checkpoints."""

    def __init__(
        self,
        manager: Any,
        *,
        maximum_files: int = 750,
        maximum_file_bytes: int = 1024 * 1024,
        maximum_total_bytes: int = 16 * 1024 * 1024,
        maximum_checksum_bytes: int = 64 * 1024 * 1024,
        maximum_log_lines: int = 50_000,
        maximum_findings: int = 250,
    ) -> None:
        if min(
            maximum_files,
            maximum_file_bytes,
            maximum_total_bytes,
            maximum_checksum_bytes,
            maximum_log_lines,
            maximum_findings,
        ) <= 0:
            raise ValueError("diagnostic scan limits must be positive")
        self.manager = manager
        self.maximum_files = int(maximum_files)
        self.maximum_file_bytes = int(maximum_file_bytes)
        self.maximum_total_bytes = int(maximum_total_bytes)
        self.maximum_checksum_bytes = int(maximum_checksum_bytes)
        self.maximum_log_lines = int(maximum_log_lines)
        self.maximum_findings = int(maximum_findings)

    def scan(self, project_root: str | Path) -> ProjectDiagnosticReport:
        project = self._validated_project(project_root)
        state = _ScanState(self.maximum_findings)
        self._inspect_existing_health(project, state)
        project_files = tuple(self._project_files(project, state))
        self._inspect_expected_files(project, project_files, state)
        self._inspect_project_files(project, project_files, state)
        rows = self._inspect_run_registry(project, state)
        self._inspect_referenced_datasets(project, rows, state)
        self._inspect_checkpoints(project, rows, state)
        self._inspect_workspace_logs(project, state)
        return ProjectDiagnosticReport(
            project.name,
            str(project),
            tuple(state.findings),
            state.files_scanned,
            state.log_files_scanned,
            state.lines_scanned,
            state.bytes_scanned,
            state.truncated,
            datetime.now(UTC).isoformat(),
        )

    def change_token(self, project_root: str | Path) -> str:
        """Return a bounded metadata fingerprint for inexpensive live watching.

        Only names, sizes, and modification timestamps are hashed. File contents
        remain unread until a change triggers the explicit diagnostic scan.
        """

        project = self._validated_project(project_root)
        roots = (
            project,
            Path(self.manager.checkpoints_dir) / project.name,
            Path(self.manager.datasets_dir),
            Path(self.manager.logs_dir),
            Path(self.manager.runs_dir),
        )
        digest = hashlib.sha256()
        entries = 0
        stack = list(
            reversed([root for root in roots if root.exists() and not root.is_symlink()])
        )
        while stack and entries < self.maximum_files:
            directory = stack.pop()
            if directory.is_file():
                candidates = (directory,)
            else:
                try:
                    candidates = tuple(
                        sorted(directory.iterdir(), key=lambda item: item.name.casefold())
                    )
                except OSError:
                    continue
            for entry in candidates:
                if entries >= self.maximum_files:
                    break
                if entry.is_symlink():
                    continue
                try:
                    if entry.is_dir():
                        if entry.name.casefold() not in _SKIP_DIRECTORIES:
                            stack.append(entry)
                        continue
                    if not entry.is_file():
                        continue
                    stat = entry.stat()
                except OSError:
                    continue
                entries += 1
                try:
                    identity = entry.resolve(strict=False).relative_to(
                        Path(self.manager.workspace_root).resolve(strict=False)
                    ).as_posix()
                except ValueError:
                    identity = entry.name
                digest.update(identity.encode("utf-8", errors="replace"))
                digest.update(b"\0")
                digest.update(str(stat.st_size).encode("ascii"))
                digest.update(b"\0")
                digest.update(str(stat.st_mtime_ns).encode("ascii"))
                digest.update(b"\0")
        digest.update(f"entries={entries};truncated={bool(stack)}".encode("ascii"))
        return digest.hexdigest()

    def _validated_project(self, value: str | Path) -> Path:
        original = Path(value)
        projects_root = Path(self.manager.projects_dir)
        if original.is_symlink() or projects_root.is_symlink():
            raise PermissionError("diagnostic scan roots cannot be symbolic links")
        project = original.resolve(strict=False)
        root = projects_root.resolve(strict=False)
        if project.parent != root:
            raise PermissionError("diagnostic scans require a direct private project directory")
        if not project.is_dir():
            raise FileNotFoundError(project)
        return project

    def _location(self, path: Path, *, line: int | None = None) -> str:
        resolved = path.resolve(strict=False)
        for label, root_value in (
            ("project", self.manager.projects_dir),
            ("workspace", self.manager.workspace_root),
        ):
            root = Path(root_value).resolve(strict=False)
            try:
                relative = resolved.relative_to(root)
            except ValueError:
                continue
            value = f"{label}/{relative.as_posix()}"
            return f"{value}:{line}" if line is not None else value
        return "outside configured project scope"

    def _inspect_existing_health(self, project: Path, state: _ScanState) -> None:
        try:
            report = ProjectHealthInspector(project, self.manager.workspace_root).inspect()
        except (OSError, PermissionError, RuntimeError, ValueError) as exc:
            state.add(
                DiagnosticFinding(
                    "health.scan",
                    DiagnosticSeverity.WARNING,
                    f"The standard project health pass could not finish ({type(exc).__name__}).",
                    "project",
                    "Review the project boundary and retry the scan.",
                )
            )
            return
        state.files_scanned += report.files_checked
        state.bytes_scanned += report.bytes_checked
        for item in report.findings:
            severity = {
                FindingSeverity.BLOCKING: DiagnosticSeverity.ERROR,
                FindingSeverity.WARNING: DiagnosticSeverity.WARNING,
                FindingSeverity.INFO: DiagnosticSeverity.INFO,
            }[item.severity]
            suggestion = {
                "workspace.marker": "Open Settings and restore a valid private workspace.",
                "project.manifest": "Restore or recreate project.json with the correct project identity.",
                "dataset.integrity": "Re-import the source data instead of editing registered files.",
                "checkpoint.integrity": "Use an earlier verified checkpoint or train a fresh one.",
                "project.secret": "Remove the suspected credential and rotate it before continuing.",
            }.get(item.code, "Review this evidence before continuing to train or release.")
            state.add(
                DiagnosticFinding(
                    f"health.{item.code}",
                    severity,
                    item.summary,
                    item.location,
                    suggestion,
                )
            )
            if item.code in {"project.scan-limit", "dataset.limit"}:
                state.truncated = True

    def _project_files(self, project: Path, state: _ScanState) -> Iterable[Path]:
        stack = [project]
        visited = 0
        while stack:
            directory = stack.pop()
            try:
                entries = sorted(directory.iterdir(), key=lambda item: item.name.casefold())
            except OSError:
                state.add(
                    DiagnosticFinding(
                        "filesystem.unreadable",
                        DiagnosticSeverity.WARNING,
                        "A project directory could not be read.",
                        self._location(directory),
                        "Check local file permissions and retry.",
                    )
                )
                continue
            for entry in entries:
                if visited >= self.maximum_files:
                    state.truncated = True
                    return
                visited += 1
                if entry.is_symlink():
                    state.add(
                        DiagnosticFinding(
                            "filesystem.symlink",
                            DiagnosticSeverity.WARNING,
                            "A symbolic link was skipped by the safe scanner.",
                            self._location(entry),
                            "Inspect the link target separately if it belongs to this project.",
                        )
                    )
                    continue
                try:
                    if entry.is_dir():
                        if entry.name.casefold() not in _SKIP_DIRECTORIES:
                            stack.append(entry)
                    elif entry.is_file():
                        yield entry
                except OSError:
                    state.add(
                        DiagnosticFinding(
                            "filesystem.entry",
                            DiagnosticSeverity.WARNING,
                            "A project entry changed or became unreadable during the scan.",
                            self._location(entry),
                            "Retry after other project tools finish writing files.",
                        )
                    )

    def _read_text(self, path: Path, state: _ScanState) -> str | None:
        try:
            size = path.stat().st_size
        except OSError:
            return None
        if size > self.maximum_file_bytes:
            state.truncated = True
            state.add(
                DiagnosticFinding(
                    "file.limit",
                    DiagnosticSeverity.INFO,
                    "A text or log file was too large for the bounded scan.",
                    self._location(path),
                    "Inspect a smaller recent excerpt or rotate the log, then scan again.",
                )
            )
            return None
        if state.budget_bytes + size > self.maximum_total_bytes:
            state.truncated = True
            return None
        try:
            value = path.read_text(encoding="utf-8")
        except UnicodeError:
            state.add(
                DiagnosticFinding(
                    "file.encoding",
                    DiagnosticSeverity.WARNING,
                    "A text-like project file is not valid UTF-8.",
                    self._location(path),
                    "Convert the file to UTF-8 or inspect it with its original application.",
                )
            )
            return None
        except OSError:
            return None
        state.files_scanned += 1
        state.bytes_scanned += size
        state.budget_bytes += size
        return value

    def _inspect_expected_files(
        self, project: Path, files: tuple[Path, ...], state: _ScanState
    ) -> None:
        names = {path.name.casefold() for path in files if path.parent == project}
        if "main.py" not in names:
            state.add(
                DiagnosticFinding(
                    "project.entrypoint",
                    DiagnosticSeverity.WARNING,
                    "The project has no main.py entry point.",
                    self._location(project / "main.py"),
                    "Create or restore the project entry point in Code Workshop.",
                )
            )
        if "readme.md" not in names:
            state.add(
                DiagnosticFinding(
                    "project.readme",
                    DiagnosticSeverity.INFO,
                    "The project has no README.md describing how to reproduce it.",
                    self._location(project / "README.md"),
                    "Document the goal, data identity, run command, and expected result.",
                )
            )
        logs = project / "logs"
        if not logs.is_dir() or logs.is_symlink():
            state.add(
                DiagnosticFinding(
                    "project.logs",
                    DiagnosticSeverity.WARNING,
                    "The project has no safe local logs folder.",
                    self._location(logs),
                    "Reopen the project in Daedalus to create its standard logs folder.",
                )
            )

    def _inspect_project_files(
        self, project: Path, files: tuple[Path, ...], state: _ScanState
    ) -> None:
        for path in files:
            suffix = path.suffix.casefold()
            if suffix == ".py":
                source = self._read_text(path, state)
                if source is None:
                    continue
                try:
                    ast.parse(source, filename=str(path))
                except SyntaxError as exc:
                    state.add(
                        DiagnosticFinding(
                            "python.syntax",
                            DiagnosticSeverity.ERROR,
                            f"Python could not parse this file ({exc.msg or 'syntax error'}).",
                            self._location(path, line=exc.lineno),
                            "Open this location in Code Workshop and correct the syntax before running it.",
                        )
                    )
            if self._is_project_log(project, path):
                content = self._read_text(path, state)
                if content is not None:
                    self._inspect_log_text(path, content, state)
            if path.parent.name.casefold() in {"run", "runs"} and suffix == ".json":
                self._inspect_local_run(path, state)

    @staticmethod
    def _is_project_log(project: Path, path: Path) -> bool:
        if path.suffix.casefold() in _PROJECT_LOG_SUFFIXES:
            return True
        try:
            relative = path.relative_to(project)
        except ValueError:
            return False
        return (
            path.suffix.casefold() == ".txt"
            and any(part.casefold() in _LOG_DIRECTORY_NAMES for part in relative.parts[:-1])
        )

    def _inspect_log_text(
        self,
        path: Path,
        content: str,
        state: _ScanState,
        *,
        require_project: str | None = None,
    ) -> None:
        state.log_files_scanned += 1
        project_token = require_project.casefold() if require_project else ""
        for line_number, line in enumerate(content.splitlines(), start=1):
            if state.lines_scanned >= self.maximum_log_lines:
                state.truncated = True
                return
            state.lines_scanned += 1
            folded = line.casefold()
            if project_token and project_token not in folded:
                continue
            if _EMPTY_ERROR_VALUE.search(line.strip()):
                continue
            for label, pattern, severity in _LOG_MARKERS:
                if not pattern.search(line):
                    continue
                state.add(
                    DiagnosticFinding(
                        "log.error",
                        severity,
                        f"A log line reports {label}; its private values were not copied.",
                        self._location(path, line=line_number),
                        "Open the surrounding log lines locally and trace the first reported failure.",
                    )
                )
                break

    def _inspect_local_run(self, path: Path, state: _ScanState) -> None:
        value = self._read_json_object(path, state, "run record")
        if value is None:
            return
        status = str(value.get("status", "")).casefold()
        if status == "failed":
            state.add(
                DiagnosticFinding(
                    "run.failed",
                    DiagnosticSeverity.ERROR,
                    "A project run record is marked failed; stored error text was not copied.",
                    self._location(path),
                    "Inspect the run configuration and its preceding log entries, then retry a bounded run.",
                )
            )
        elif status == "cancelled":
            state.add(
                DiagnosticFinding(
                    "run.cancelled",
                    DiagnosticSeverity.WARNING,
                    "A project run record is marked cancelled.",
                    self._location(path),
                    "Confirm whether cancellation was intentional before using its artifacts.",
                )
            )
        if self._contains_nonfinite(value.get("metrics")):
            state.add(
                DiagnosticFinding(
                    "run.nonfinite",
                    DiagnosticSeverity.ERROR,
                    "A run record contains a non-finite metric.",
                    self._location(path),
                    "Check input scaling, learning rate, and the first epoch where the metric diverged.",
                )
            )

    def _inspect_run_registry(
        self, project: Path, state: _ScanState
    ) -> tuple[dict[str, Any], ...]:
        database = Path(self.manager.runs_dir) / "runs.sqlite3"
        if not database.exists():
            return ()
        if database.is_symlink() or not database.is_file():
            state.add(
                DiagnosticFinding(
                    "run.registry-path",
                    DiagnosticSeverity.ERROR,
                    "The training run registry is not a regular local file.",
                    self._location(database),
                    "Restore the registry from a verified backup before training again.",
                )
            )
            return ()
        try:
            uri = database.resolve(strict=True).as_uri() + "?mode=ro"
            with sqlite3.connect(uri, uri=True, timeout=2) as connection:
                connection.row_factory = sqlite3.Row
                rows = connection.execute(
                    """SELECT id, status, dataset, updated_utc, metrics_json, checkpoint, error
                    FROM runs WHERE project=? ORDER BY updated_utc DESC LIMIT 101""",
                    (project.name,),
                ).fetchall()
        except (OSError, sqlite3.DatabaseError) as exc:
            state.add(
                DiagnosticFinding(
                    "run.registry",
                    DiagnosticSeverity.ERROR,
                    f"The training run registry could not be read ({type(exc).__name__}).",
                    self._location(database),
                    "Close competing tools, verify the workspace drive, and retry before training.",
                )
            )
            return ()
        if len(rows) > 100:
            state.truncated = True
            rows = rows[:100]
        results: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            results.append(item)
            run_label = f"{self._location(database)} (run {str(item['id'])[:8]})"
            status = str(item.get("status", "")).casefold()
            if status == "failed":
                state.add(
                    DiagnosticFinding(
                        "run.failed",
                        DiagnosticSeverity.ERROR,
                        "A registered training run failed; stored error text was not copied.",
                        run_label,
                        "Review this run's configuration and the earliest related error, then retry.",
                    )
                )
            elif status == "cancelled":
                state.add(
                    DiagnosticFinding(
                        "run.cancelled",
                        DiagnosticSeverity.WARNING,
                        "A registered training run was cancelled.",
                        run_label,
                        "Confirm cancellation was intentional and do not treat it as completed evidence.",
                    )
                )
            elif status in {"queued", "running"} and self._is_stale(item.get("updated_utc")):
                state.add(
                    DiagnosticFinding(
                        "run.stale",
                        DiagnosticSeverity.WARNING,
                        f"A training run is still marked {status} after more than one hour.",
                        run_label,
                        "Check whether the application closed during training and preserve the run journal.",
                    )
                )
            try:
                metrics = json.loads(str(item.get("metrics_json") or "{}"))
            except (TypeError, ValueError, json.JSONDecodeError):
                state.add(
                    DiagnosticFinding(
                        "run.metrics-json",
                        DiagnosticSeverity.ERROR,
                        "A training run contains unreadable metrics metadata.",
                        run_label,
                        "Keep the record for recovery and run a fresh experiment.",
                    )
                )
            else:
                if self._contains_nonfinite(metrics):
                    state.add(
                        DiagnosticFinding(
                            "run.nonfinite",
                            DiagnosticSeverity.ERROR,
                            "A training run contains a non-finite metric.",
                            run_label,
                            "Check data scaling, learning rate, and the first divergent epoch.",
                        )
                    )
        return tuple(results)

    @staticmethod
    def _is_stale(value: Any) -> bool:
        try:
            timestamp = datetime.fromisoformat(str(value))
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=UTC)
            return datetime.now(UTC) - timestamp.astimezone(UTC) > timedelta(hours=1)
        except (TypeError, ValueError):
            return True

    @classmethod
    def _contains_nonfinite(cls, value: Any, *, depth: int = 0) -> bool:
        if depth > 12:
            return False
        if isinstance(value, float):
            return not math.isfinite(value)
        if isinstance(value, dict):
            return any(cls._contains_nonfinite(item, depth=depth + 1) for item in value.values())
        if isinstance(value, (list, tuple)):
            return any(cls._contains_nonfinite(item, depth=depth + 1) for item in value)
        return False

    def _inspect_referenced_datasets(
        self,
        project: Path,
        rows: tuple[dict[str, Any], ...],
        state: _ScanState,
    ) -> None:
        names = sorted(
            {
                str(item.get("dataset") or "").strip()
                for item in rows
                if str(item.get("dataset") or "").strip()
                and not str(item.get("dataset") or "").casefold().startswith("built-in ")
            },
            key=str.casefold,
        )
        for name in names[:100]:
            if Path(name).name != name or name in {".", ".."}:
                state.add(
                    DiagnosticFinding(
                        "dataset.path",
                        DiagnosticSeverity.ERROR,
                        "A run references an unsafe dataset name.",
                        self._location(Path(self.manager.runs_dir) / "runs.sqlite3"),
                        "Do not load this record; re-import data under a simple local name.",
                    )
                )
                continue
            metadata_path = Path(self.manager.datasets_dir) / f"{name}.dataset.json"
            metadata = self._read_json_object(metadata_path, state, "dataset metadata")
            if metadata is None:
                state.add(
                    DiagnosticFinding(
                        "dataset.missing",
                        DiagnosticSeverity.ERROR,
                        f"Training history references dataset {name!r}, but its metadata is missing or invalid.",
                        self._location(metadata_path),
                        "Restore the exact registered dataset or re-import it and start a new run.",
                    )
                )
                continue
            try:
                if metadata.get("schema") != 1 or metadata.get("name") != name:
                    raise ValueError("identity")
                file_name = str(metadata["file"])
                data_path = (metadata_path.parent / file_name).resolve(strict=False)
                if Path(file_name).is_absolute() or Path(file_name).name != file_name:
                    raise ValueError("path")
                if data_path.parent != Path(self.manager.datasets_dir).resolve(strict=False):
                    raise ValueError("boundary")
                expected = str(metadata["sha256"])
                actual = self._bounded_sha256(data_path, state)
                if actual is not None and actual != expected:
                    raise ValueError("checksum")
            except (KeyError, OSError, TypeError, ValueError):
                state.add(
                    DiagnosticFinding(
                        "dataset.integrity",
                        DiagnosticSeverity.ERROR,
                        f"Referenced dataset {name!r} failed its identity, path, or checksum check.",
                        self._location(metadata_path),
                        "Re-import the original data; do not edit registered dataset files in place.",
                    )
                )
        if len(names) > 100:
            state.truncated = True

    def _inspect_checkpoints(
        self,
        project: Path,
        rows: tuple[dict[str, Any], ...],
        state: _ScanState,
    ) -> None:
        candidates: set[Path] = set()
        for item in rows:
            raw = str(item.get("checkpoint") or "").strip()
            if raw:
                candidates.add(Path(raw).resolve(strict=False))
        for directory in (
            Path(self.manager.checkpoints_dir) / project.name,
            project / "checkpoints",
        ):
            if directory.is_dir() and not directory.is_symlink():
                candidates.update(path.resolve(strict=False) for path in directory.glob("*.json"))
        for metadata_path in sorted(candidates, key=lambda path: str(path).casefold())[:100]:
            if not self._inside_checkpoint_roots(project, metadata_path):
                state.add(
                    DiagnosticFinding(
                        "checkpoint.path",
                        DiagnosticSeverity.ERROR,
                        "A run references a checkpoint outside the selected project boundary.",
                        "workspace/training-runs/runs.sqlite3",
                        "Do not load the checkpoint; select one stored under this project's checkpoint folder.",
                    )
                )
                continue
            metadata = self._read_json_object(metadata_path, state, "checkpoint metadata")
            if metadata is None:
                state.add(
                    DiagnosticFinding(
                        "checkpoint.missing",
                        DiagnosticSeverity.ERROR,
                        "A referenced checkpoint is missing or has invalid metadata.",
                        self._location(metadata_path),
                        "Restore a verified checkpoint or train a fresh model.",
                    )
                )
                continue
            try:
                if metadata.get("format") != "daedalus-npz" or metadata.get("schema") not in {
                    1,
                    2,
                }:
                    raise ValueError("format")
                array_name = str(metadata["array_file"])
                if Path(array_name).is_absolute() or Path(array_name).name != array_name:
                    raise ValueError("path")
                arrays_path = (metadata_path.parent / array_name).resolve(strict=False)
                if arrays_path.parent != metadata_path.parent.resolve(strict=False):
                    raise ValueError("boundary")
                expected = str(metadata["sha256"])
                actual = self._bounded_sha256(arrays_path, state)
                if actual is not None and actual != expected:
                    raise ValueError("checksum")
            except (KeyError, OSError, TypeError, ValueError):
                state.add(
                    DiagnosticFinding(
                        "checkpoint.integrity",
                        DiagnosticSeverity.ERROR,
                        "A checkpoint failed its format, path, or checksum check.",
                        self._location(metadata_path),
                        "Use an earlier verified checkpoint or train a fresh one.",
                    )
                )
        if len(candidates) > 100:
            state.truncated = True

    def _inside_checkpoint_roots(self, project: Path, path: Path) -> bool:
        for root in (
            (Path(self.manager.checkpoints_dir) / project.name).resolve(strict=False),
            (project / "checkpoints").resolve(strict=False),
        ):
            try:
                path.resolve(strict=False).relative_to(root)
                return True
            except ValueError:
                continue
        return False

    def _inspect_workspace_logs(self, project: Path, state: _ScanState) -> None:
        directory = Path(self.manager.logs_dir)
        if not directory.is_dir() or directory.is_symlink():
            return
        checked = 0
        stack = [directory]
        while stack and checked < 100:
            parent = stack.pop()
            try:
                entries = sorted(parent.iterdir(), key=lambda item: item.name.casefold())
            except OSError:
                continue
            for entry in entries:
                if checked >= 100:
                    state.truncated = True
                    return
                if entry.is_symlink():
                    continue
                if entry.is_dir():
                    stack.append(entry)
                    continue
                if not entry.is_file() or entry.suffix.casefold() not in (
                    _PROJECT_LOG_SUFFIXES | {".txt"}
                ):
                    continue
                checked += 1
                content = self._read_text(entry, state)
                if content is None:
                    continue
                require_project = None if project.name.casefold() in entry.name.casefold() else project.name
                self._inspect_log_text(
                    entry,
                    content,
                    state,
                    require_project=require_project,
                )

    def _read_json_object(
        self, path: Path, state: _ScanState, label: str
    ) -> dict[str, Any] | None:
        if path.is_symlink() or not path.is_file():
            return None
        content = self._read_text(path, state)
        if content is None:
            return None
        try:
            value = json.loads(content)
        except (TypeError, ValueError, json.JSONDecodeError):
            state.add(
                DiagnosticFinding(
                    "metadata.json",
                    DiagnosticSeverity.ERROR,
                    f"A {label} file is not valid JSON.",
                    self._location(path),
                    "Restore the metadata from a verified source or create a new artifact.",
                )
            )
            return None
        if not isinstance(value, dict):
            state.add(
                DiagnosticFinding(
                    "metadata.object",
                    DiagnosticSeverity.ERROR,
                    f"A {label} file does not contain the expected JSON object.",
                    self._location(path),
                    "Restore the metadata from a verified source or create a new artifact.",
                )
            )
            return None
        return value

    def _bounded_sha256(self, path: Path, state: _ScanState) -> str | None:
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(path)
        size = path.stat().st_size
        if size > self.maximum_checksum_bytes:
            state.truncated = True
            state.add(
                DiagnosticFinding(
                    "checksum.limit",
                    DiagnosticSeverity.INFO,
                    "A large data or checkpoint file was not checksummed by the bounded scan.",
                    self._location(path),
                    "Use the dedicated data or checkpoint verifier for a full integrity check.",
                )
            )
            return None
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        state.files_scanned += 1
        state.bytes_scanned += size
        return digest.hexdigest()


__all__ = [
    "DiagnosticFinding",
    "DiagnosticSeverity",
    "ProjectDiagnosticReport",
    "ProjectDiagnosticsScanner",
]
