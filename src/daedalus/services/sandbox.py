"""Constrained project runner for trusted learning code.

This is deliberately not described as a security boundary. It adds path,
syntax, import, environment, and timeout controls for ordinary learner mistakes;
hostile code belongs in a disposable virtual machine.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from daedalus.workspace.manager import WorkspaceManager

_BLOCKED_IMPORTS = {
    "ctypes",
    "http",
    "importlib",
    "multiprocessing",
    "os",
    "requests",
    "shutil",
    "socket",
    "subprocess",
    "urllib",
    "winreg",
}
_BLOCKED_CALLS = {"breakpoint", "compile", "eval", "exec", "__import__"}
_LOG_LOCK = threading.Lock()


@dataclass(slots=True)
class RunResult:
    path: str
    return_code: int
    stdout: str
    stderr: str
    elapsed_seconds: float
    timed_out: bool = False
    policy_warnings: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return self.return_code == 0 and not self.timed_out


class SandboxPolicyError(ValueError):
    pass


def inspect_source(source: str) -> tuple[ast.AST, list[str]]:
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise SandboxPolicyError(f"Syntax error on line {exc.lineno}: {exc.msg}") from exc
    warnings: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = {alias.name.split(".")[0] for alias in node.names}
            blocked = names & _BLOCKED_IMPORTS
            if blocked:
                warnings.append(f"blocked import: {', '.join(sorted(blocked))}")
        elif isinstance(node, ast.ImportFrom) and node.module:
            root = node.module.split(".")[0]
            if root in _BLOCKED_IMPORTS:
                warnings.append(f"blocked import: {root}")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in _BLOCKED_CALLS:
                warnings.append(f"blocked dynamic-code call: {node.func.id}")
    return tree, warnings


class SandboxRunner:
    def __init__(self, manager: WorkspaceManager, *, timeout_seconds: float = 15.0) -> None:
        self.manager = manager
        self.timeout_seconds = max(1.0, min(float(timeout_seconds), 120.0))

    def validate_file(self, path: str | os.PathLike[str]) -> tuple[Path, tuple[str, ...]]:
        candidate = self.manager.resolve_user_path(path, must_exist=True)
        projects_root = self.manager.projects_dir.resolve(strict=False)
        if candidate != projects_root and projects_root not in candidate.parents:
            raise PermissionError("Only files inside the private projects directory may run.")
        if candidate.suffix.casefold() != ".py" or not candidate.is_file():
            raise SandboxPolicyError("The workshop runner accepts Python source files only.")
        source = candidate.read_text(encoding="utf-8")
        _, warnings = inspect_source(source)
        return candidate, tuple(warnings)

    def run_file(
        self,
        path: str | os.PathLike[str],
        *,
        allow_restricted_imports: bool = False,
    ) -> RunResult:
        candidate, warnings = self.validate_file(path)
        if warnings and not allow_restricted_imports:
            self._record_run(candidate, level="WARNING", event="sandbox_policy_blocked")
            raise SandboxPolicyError(
                "Workshop policy stopped this run: " + "; ".join(warnings)
                + ". Use a VM for untrusted or system-accessing code."
            )

        environment = {
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUTF8": "1",
            "DAEDALUS_WORKSPACE_ROOT": str(self.manager.workspace_root),
        }
        for name in ("PATH", "SystemRoot", "WINDIR", "TEMP", "TMP", "HOME"):
            if value := os.environ.get(name):
                environment[name] = value
        creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        started = time.perf_counter()
        try:
            completed = subprocess.run(
                [sys.executable, "-I", str(candidate)],
                cwd=candidate.parent,
                env=environment,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.timeout_seconds,
                creationflags=creationflags,
                check=False,
            )
            result = RunResult(
                str(candidate),
                completed.returncode,
                completed.stdout[-200_000:],
                completed.stderr[-200_000:],
                time.perf_counter() - started,
                policy_warnings=warnings,
            )
            self._record_result(candidate, result)
            return result
        except subprocess.TimeoutExpired as exc:
            result = RunResult(
                str(candidate),
                124,
                (exc.stdout or "") if isinstance(exc.stdout, str) else "",
                "Execution exceeded the workshop time limit and was stopped.",
                time.perf_counter() - started,
                timed_out=True,
                policy_warnings=warnings,
            )
            self._record_result(candidate, result)
            return result

    def _record_result(self, candidate: Path, result: RunResult) -> None:
        if result.timed_out:
            event = "sandbox_timed_out"
        elif result.ok:
            event = "sandbox_completed"
        else:
            event = "sandbox_failed"
        self._record_run(
            candidate,
            level="INFO" if result.ok else "ERROR",
            event=event,
            return_code=result.return_code,
            elapsed_seconds=result.elapsed_seconds,
        )

    def _record_run(
        self,
        candidate: Path,
        *,
        level: str,
        event: str,
        return_code: int | None = None,
        elapsed_seconds: float | None = None,
    ) -> None:
        """Append a small private event without copying stdout or stderr values."""

        try:
            projects_root = self.manager.projects_dir.resolve(strict=True)
            relative = candidate.resolve(strict=True).relative_to(projects_root)
            if len(relative.parts) < 2:
                return
            project = projects_root / relative.parts[0]
            logs = self.manager.ensure_project_logs(project)
            log_path = logs / f"daedalus-{datetime.now(UTC).date().isoformat()}.log"
            fields = [
                datetime.now(UTC).isoformat(),
                f"level={level}",
                f"event={event}",
                f"file={relative.as_posix()}",
            ]
            if return_code is not None:
                fields.append(f"return_code={int(return_code)}")
            if elapsed_seconds is not None:
                fields.append(f"elapsed_seconds={max(0.0, float(elapsed_seconds)):.6f}")
            with _LOG_LOCK, log_path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(" ".join(fields) + "\n")
        except (OSError, PermissionError, TypeError, ValueError):
            # Logging must never hide or alter the actual sandbox outcome.
            return
