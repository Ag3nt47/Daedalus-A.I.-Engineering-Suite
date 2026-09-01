"""Privacy, quality, dependency, and GitHub advisory gates for safe publishing."""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Iterator, Mapping
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Iterable, Sequence
from urllib.parse import urlparse

BLOCK = "block"
WARNING = "warning"
INFO = "info"
PASS = "pass"

_ROOT_PRIVATE_PREFIXES = {
    ".daedalus",
    ".env",
    ".venv",
    "checkpoints",
    "datasets",
    "logs",
    "models",
    "projects",
    "runtime-data",
    "training-runs",
    "user-workspaces",
    "weights",
    "workspace",
    "workspaces",
}
_PRIVATE_SUFFIXES = {
    ".ckpt",
    ".gguf",
    ".h5",
    ".hdf5",
    ".joblib",
    ".key",
    ".npy",
    ".npz",
    ".onnx",
    ".p12",
    ".parquet",
    ".pem",
    ".pfx",
    ".pickle",
    ".pkl",
    ".pt",
    ".pth",
    ".safetensors",
    ".tflite",
    ".token",
}
_REQUIRED_IGNORE_RULES = {
    "/.env",
    "/.venv/",
    "/workspace/",
    "/workspaces/",
    "/user-workspaces/",
    "/projects/",
    "/datasets/",
    "/weights/",
    "/checkpoints/",
    "/training-runs/",
    "/logs/",
    "/models/",
    "*.npz",
}
_PRIVATE_DIRECTORY_NAMES = {
    "workspace",
    "workspaces",
    "user-workspaces",
    "projects",
    "datasets",
    "weights",
    "checkpoints",
    "training-runs",
    "logs",
    "models",
}
_PRIVATE_MARKER_NAMES = {
    ".daedalus-backup-root.json",
    ".daedalus-workspace.json",
}
_REQUIRED_PUBLIC_WORKSPACE_FILES = {
    PurePosixPath("src/daedalus/workspace/__init__.py"),
    PurePosixPath("src/daedalus/workspace/checkpoints.py"),
    PurePosixPath("src/daedalus/workspace/datasets.py"),
    PurePosixPath("src/daedalus/workspace/manager.py"),
    PurePosixPath("src/daedalus/workspace/run_registry.py"),
}
_SECRET_PATTERNS = {
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "GitHub token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"),
    "GitHub fine-grained token": re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
    "AWS access key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "Slack token": re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{20,}\b"),
    "credential assignment": re.compile(
        r"(?im)^\s*(?:api[_-]?key|secret|access[_-]?token|password)\s*[:=]\s*"
        r"[\"']?(?!example|change-me|your[_ -]?)[A-Za-z0-9_./+\-=]{16,}"
    ),
}
_TEXT_SUFFIXES = {
    "",
    ".bat",
    ".c",
    ".cfg",
    ".cc",
    ".cmd",
    ".conf",
    ".config",
    ".cpp",
    ".css",
    ".cs",
    ".csv",
    ".cxx",
    ".env",
    ".gql",
    ".go",
    ".gradle",
    ".graphql",
    ".h",
    ".hcl",
    ".html",
    ".hpp",
    ".ini",
    ".ipynb",
    ".java",
    ".js",
    ".jsx",
    ".json",
    ".kt",
    ".kts",
    ".lock",
    ".log",
    ".md",
    ".php",
    ".pl",
    ".properties",
    ".ps1",
    ".py",
    ".qss",
    ".rb",
    ".rego",
    ".rst",
    ".rs",
    ".scala",
    ".sh",
    ".sql",
    ".svelte",
    ".svg",
    ".swift",
    ".tf",
    ".tfvars",
    ".toml",
    ".ts",
    ".tsv",
    ".tsx",
    ".txt",
    ".vue",
    ".xml",
    ".yaml",
    ".yml",
}

_GIT_ZERO_OIDS = {"0" * 40, "0" * 64}
_STREAM_CHUNK_BYTES = 1024 * 1024
_SECRET_SCAN_OVERLAP_BYTES = 4096
_INITIAL_PUBLISH_ENV = "DAEDALUS_INITIAL_PUBLISH"
_TEST_SUITE_TIMEOUT_SECONDS = 20 * 60


def _dangerously_unanchored_private_rule(rule: str) -> bool:
    """Return whether a broad directory rule can hide package source at any depth."""

    value = rule.strip()
    if not value or value.startswith(("!", "/")):
        return False
    parts = value.rstrip("/").split("/")
    return bool(parts) and parts[-1].casefold() in _PRIVATE_DIRECTORY_NAMES and all(
        part in {"*", "**"} for part in parts[:-1]
    )


@dataclass(slots=True)
class Finding:
    level: str
    check: str
    message: str
    path: str | None = None
    remediation: str | None = None


@dataclass(slots=True)
class GuardReport:
    started_utc: str
    finished_utc: str
    repository: str
    scope: str
    findings: list[Finding] = field(default_factory=list)

    @property
    def blocking(self) -> list[Finding]:
        return [finding for finding in self.findings if finding.level == BLOCK]

    @property
    def ok(self) -> bool:
        return not self.blocking

    def counts(self) -> dict[str, int]:
        return {
            level: sum(finding.level == level for finding in self.findings)
            for level in (BLOCK, WARNING, INFO, PASS)
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "started_utc": self.started_utc,
            "finished_utc": self.finished_utc,
            "repository": self.repository,
            "scope": self.scope,
            "ok": self.ok,
            "counts": self.counts(),
            "findings": [asdict(finding) for finding in self.findings],
        }

    def format_text(self) -> str:
        counts = self.counts()
        lines = [
            "DAEDALUS RELEASE GUARD",
            f"Repository: {self.repository}",
            f"Scope: {self.scope}",
            f"Result: {'PASS' if self.ok else 'BLOCKED'}",
            f"Blocking: {counts[BLOCK]}  Warnings: {counts[WARNING]}  "
            f"Info: {counts[INFO]}  Passed checks: {counts[PASS]}",
            "",
        ]
        for finding in self.findings:
            location = f" [{finding.path}]" if finding.path else ""
            lines.append(f"{finding.level.upper():7} {finding.check}{location}: {finding.message}")
            if finding.remediation:
                lines.append(f"         Fix: {finding.remediation}")
        return "\n".join(lines)


class ReleaseGuard:
    def __init__(self, repository: str | os.PathLike[str]) -> None:
        self.repository = Path(repository).resolve(strict=False)
        if not self.repository.is_dir():
            raise FileNotFoundError(self.repository)

    def _run(
        self,
        command: Sequence[str],
        *,
        timeout: float = 120,
        check: bool = False,
        env: Mapping[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            list(command),
            cwd=self.repository,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=check,
            env=env,
        )

    def _git(
        self,
        *arguments: str,
        timeout: float = 60,
        env: Mapping[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        return self._run(["git", *arguments], timeout=timeout, env=env)

    @property
    def is_git_repository(self) -> bool:
        return (self.repository / ".git").exists()

    def candidate_paths(self, *, staged: bool = False) -> list[Path]:
        if self.is_git_repository:
            arguments = (
                ["diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z"]
                if staged
                else ["ls-files", "-co", "--exclude-standard", "-z"]
            )
            result = self._git(*arguments)
            if result.returncode != 0:
                raise RuntimeError(result.stderr.strip() or "Git could not enumerate files.")
            return [Path(value) for value in result.stdout.split("\0") if value]

        excluded = {".git", ".venv", "venv", "env", "__pycache__", "build", "dist"}
        results: list[Path] = []
        for root, directories, files in os.walk(self.repository, followlinks=False):
            directories[:] = [name for name in directories if name not in excluded]
            root_path = Path(root)
            for name in files:
                results.append((root_path / name).relative_to(self.repository))
        return results

    def _staged_blob(self, relative: Path) -> bytes | None:
        completed = subprocess.run(
            ["git", "show", f":{relative.as_posix()}"],
            cwd=self.repository,
            capture_output=True,
            timeout=30,
            check=False,
        )
        return completed.stdout if completed.returncode == 0 else None

    @staticmethod
    def _push_update(line: str) -> tuple[str, str, str, str]:
        """Parse one line from Git's pre-push hook protocol, failing closed."""

        fields = line.rstrip("\r\n").split()
        if len(fields) != 4:
            raise ValueError("Malformed pre-push update; expected four fields.")
        local_ref, local_oid, remote_ref, remote_oid = fields
        oid_pattern = re.compile(r"[0-9a-fA-F]{40}|[0-9a-fA-F]{64}")
        if not oid_pattern.fullmatch(local_oid) or not oid_pattern.fullmatch(remote_oid):
            raise ValueError("Malformed pre-push update; object IDs are invalid.")
        return local_ref, local_oid.casefold(), remote_ref, remote_oid.casefold()

    def _outgoing_commits(self, local_oid: str, remote_oid: str) -> list[str]:
        """Return every commit made reachable by a single ref update."""

        arguments = ["rev-list", "--reverse", local_oid]
        if remote_oid not in _GIT_ZERO_OIDS:
            arguments.append(f"^{remote_oid}")
        result = self._git(*arguments, timeout=120)
        if result.returncode:
            raise RuntimeError(result.stderr.strip() or "Git could not enumerate outgoing commits.")
        commits = [value.strip() for value in result.stdout.splitlines() if value.strip()]
        if any(not re.fullmatch(r"[0-9a-fA-F]{40}|[0-9a-fA-F]{64}", value) for value in commits):
            raise RuntimeError("Git returned an invalid outgoing commit ID.")
        return commits

    def _branch_update_is_fast_forward(
        self,
        local_oid: str,
        remote_ref: str,
        remote_oid: str,
        findings: list[Finding],
    ) -> bool:
        """Fail closed when an existing remote branch would be rewritten."""

        if remote_ref.startswith("refs/heads/") and remote_oid not in _GIT_ZERO_OIDS:
            result = self._git("merge-base", "--is-ancestor", remote_oid, local_oid)
            if result.returncode == 0:
                return True
            if result.returncode == 1:
                message = "The attempted push would rewrite existing remote branch history."
            else:
                detail = result.stderr.strip() or "Git could not prove branch ancestry."
                message = f"Remote branch ancestry could not be verified safely: {detail}"
            findings.append(
                Finding(
                    BLOCK,
                    "non-fast-forward",
                    message,
                    remote_ref,
                    "Fetch the remote branch, reconcile it without force, and run Safe Push again.",
                )
            )
            return False
        return True

    def _commit_tree_blobs(self, commit: str) -> list[tuple[Path, str]]:
        """List path/blob pairs from one immutable commit snapshot."""

        result = self._git("ls-tree", "-r", "-z", "--full-tree", commit, timeout=120)
        if result.returncode:
            raise RuntimeError(result.stderr.strip() or f"Git could not inspect commit {commit}.")
        blobs: list[tuple[Path, str]] = []
        for record in result.stdout.split("\0"):
            if not record:
                continue
            metadata, separator, path = record.partition("\t")
            fields = metadata.split()
            if not separator or len(fields) != 3:
                raise RuntimeError(f"Git returned a malformed tree entry for commit {commit}.")
            _mode, object_type, oid = fields
            if object_type == "blob":
                blobs.append((Path(path), oid))
        return blobs

    def _git_blob_size(self, oid: str) -> int:
        result = self._git("cat-file", "-s", oid)
        if result.returncode:
            raise RuntimeError(result.stderr.strip() or f"Git blob {oid} could not be sized.")
        try:
            size = int(result.stdout.strip())
        except ValueError as exc:
            raise RuntimeError(f"Git returned an invalid size for blob {oid}.") from exc
        if size < 0:
            raise RuntimeError(f"Git returned an invalid size for blob {oid}.")
        return size

    def _git_blob_chunks(self, oid: str) -> Iterator[bytes]:
        """Stream an immutable blob without materializing a large file in memory."""

        process = subprocess.Popen(  # noqa: S603 - fixed git executable and validated object ID
            ["git", "cat-file", "blob", oid],
            cwd=self.repository,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if process.stdout is None or process.stderr is None:  # pragma: no cover - Popen contract
            process.kill()
            raise RuntimeError(f"Git blob {oid} could not be opened.")
        try:
            while chunk := process.stdout.read(_STREAM_CHUNK_BYTES):
                yield chunk
            stderr = process.stderr.read().decode("utf-8", errors="replace").strip()
            returncode = process.wait(timeout=30)
        except BaseException:
            process.kill()
            process.wait()
            raise
        if returncode:
            raise RuntimeError(stderr or f"Git blob {oid} could not be read.")

    @staticmethod
    def _file_chunks(path: Path) -> Iterator[bytes]:
        with path.open("rb") as handle:
            while chunk := handle.read(_STREAM_CHUNK_BYTES):
                yield chunk

    @staticmethod
    def _byte_chunks(value: bytes) -> Iterator[bytes]:
        for offset in range(0, len(value), _STREAM_CHUNK_BYTES):
            yield value[offset : offset + _STREAM_CHUNK_BYTES]

    def _scan_text_chunks(
        self,
        chunks: Iterable[bytes],
        findings: list[Finding],
        pure: PurePosixPath,
        *,
        collect: bool,
    ) -> bytes | None:
        """Scan arbitrarily large text streams, preserving matches across chunk edges."""

        overlap = b""
        collected = bytearray() if collect else None
        detected: set[str] = set()
        for chunk in chunks:
            if collected is not None:
                collected.extend(chunk)
            window = overlap + chunk
            text = window.decode("utf-8", errors="replace")
            for label, pattern in _SECRET_PATTERNS.items():
                if label not in detected and pattern.search(text):
                    detected.add(label)
                    findings.append(
                        Finding(
                            BLOCK,
                            "secret-scan",
                            f"Credential-like {label} detected; value is redacted.",
                            pure.as_posix(),
                            "Remove it, rotate/revoke the credential, then scan again.",
                        )
                    )
            overlap = window[-_SECRET_SCAN_OVERLAP_BYTES:]
        return bytes(collected) if collected is not None else None

    @staticmethod
    def _scan_path_policy(pure: PurePosixPath, findings: list[Finding]) -> bool:
        """Apply path-based publication rules; return whether content can be skipped."""

        first = pure.parts[0].casefold() if pure.parts else ""
        name = pure.name.casefold()
        suffix = Path(name).suffix.casefold()
        if name.startswith(".env") and name != ".env.example":
            findings.append(
                Finding(
                    BLOCK,
                    "private-path",
                    "An environment file is a publish candidate.",
                    pure.as_posix(),
                    "Remove the tracked file, rotate any exposed credentials, and publish only .env.example.",
                )
            )
            return True
        if name in _PRIVATE_MARKER_NAMES:
            findings.append(
                Finding(
                    BLOCK,
                    "private-marker",
                    "A Daedalus workspace or backup ownership marker cannot be published.",
                    pure.as_posix(),
                    "Move the complete private workspace or backup outside the source repository.",
                )
            )
            return True
        if first in _ROOT_PRIVATE_PREFIXES:
            findings.append(
                Finding(
                    BLOCK,
                    "private-path",
                    "A private runtime or user-workspace path is a publish candidate.",
                    pure.as_posix(),
                    "Move user work to the external workspace and remove tracked copies.",
                )
            )
            return True
        if suffix in _PRIVATE_SUFFIXES:
            findings.append(
                Finding(
                    BLOCK,
                    "private-artifact",
                    f"Private/model artifact type {suffix} cannot be published by Safe Push.",
                    pure.as_posix(),
                )
            )
            return True
        return False

    def scan(
        self,
        *,
        staged: bool = False,
        include_tests: bool = False,
        include_dependencies: bool = False,
        include_github: bool = False,
    ) -> GuardReport:
        started = datetime.now(UTC)
        findings: list[Finding] = []
        self._scan_repository_policy(findings)
        paths = self.candidate_paths(staged=staged)
        if not staged:
            self._scan_required_public_workspace(paths, findings)
        self._scan_paths(paths, findings, staged=staged)
        if include_tests:
            self._scan_ruff(findings)
            self._scan_tests(findings)
        if include_dependencies:
            self._scan_dependencies(findings)
        if include_github:
            self._scan_github_dependabot(findings)
        if not any(finding.level == BLOCK for finding in findings):
            findings.append(Finding(PASS, "gate", "No blocking release finding remains."))
        return GuardReport(
            started.isoformat(),
            datetime.now(UTC).isoformat(),
            str(self.repository),
            "staged index" if staged else "publish candidates",
            findings,
        )

    def scan_outgoing(
        self,
        updates: Iterable[str],
        *,
        remote_name: str = "origin",
        include_tests: bool = False,
        include_dependencies: bool = False,
        include_github: bool = False,
        initial_publish: bool = False,
    ) -> GuardReport:
        """Scan immutable objects in every commit an attempted push would publish."""

        started = datetime.now(UTC)
        findings: list[Finding] = []
        self._scan_repository_policy(findings)
        if initial_publish:
            initial_error = self._initial_publish_error(remote_name)
            if initial_error:
                findings.append(Finding(BLOCK, "initial-publish", initial_error))
        update_count = 0
        commits: list[str] = []
        try:
            for raw_update in updates:
                if not raw_update.strip():
                    continue
                update_count += 1
                _local_ref, local_oid, remote_ref, remote_oid = self._push_update(raw_update)
                if local_oid in _GIT_ZERO_OIDS:  # A deletion publishes no new object.
                    continue
                if initial_publish and remote_oid not in _GIT_ZERO_OIDS:
                    findings.append(
                        Finding(
                            BLOCK,
                            "initial-publish",
                            "Git reported an existing remote ref during initial publication.",
                            remote_ref,
                        )
                    )
                    continue
                if not self._branch_update_is_fast_forward(
                    local_oid, remote_ref, remote_oid, findings
                ):
                    continue
                commits.extend(self._outgoing_commits(local_oid, remote_oid))
        except (OSError, RuntimeError, ValueError) as exc:
            findings.append(
                Finding(
                    BLOCK,
                    "outgoing-history",
                    f"Outgoing Git objects could not be enumerated safely: {exc}",
                    remediation="Fetch the remote, verify the push update, and run Safe Push again.",
                )
            )

        if update_count == 0:
            findings.append(
                Finding(
                    BLOCK,
                    "outgoing-history",
                    "Git supplied no pre-push ref updates, so publication scope is unknown.",
                )
            )
        elif not any(finding.check == "outgoing-history" for finding in findings):
            self._scan_outgoing_commits(commits, findings)
        if include_tests:
            self._scan_ruff(findings)
            self._scan_tests(findings)
        if include_dependencies:
            self._scan_dependencies(findings)
        if include_github and not initial_publish:
            self._scan_github_dependabot(findings)
        if not any(finding.level == BLOCK for finding in findings):
            findings.append(Finding(PASS, "gate", "No blocking release finding remains."))
        return GuardReport(
            started.isoformat(),
            datetime.now(UTC).isoformat(),
            str(self.repository),
            f"outgoing commits for {remote_name}",
            findings,
        )

    def _scan_outgoing_commits(self, commits: Iterable[str], findings: list[Finding]) -> None:
        unique_commits = list(dict.fromkeys(commits))
        entries: dict[tuple[str, str], tuple[Path, str]] = {}
        try:
            for commit in unique_commits:
                for relative, oid in self._commit_tree_blobs(commit):
                    entries.setdefault((relative.as_posix(), oid), (relative, oid))
        except (OSError, RuntimeError) as exc:
            findings.append(
                Finding(BLOCK, "outgoing-history", f"Outgoing commit tree could not be read: {exc}")
            )
            return

        for relative, oid in entries.values():
            pure = PurePosixPath(relative.as_posix())
            if self._scan_path_policy(pure, findings):
                continue
            suffix = Path(pure.name).suffix.casefold()
            try:
                size = self._git_blob_size(oid)
                self._scan_blob_stream(
                    self._git_blob_chunks(oid), size, suffix, pure, findings
                )
            except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
                findings.append(Finding(BLOCK, "read", str(exc), pure.as_posix()))
        findings.append(
            Finding(
                INFO,
                "outgoing-history",
                f"Inspected {len(entries)} immutable blobs across {len(unique_commits)} outgoing commits.",
            )
        )

    def _scan_repository_policy(self, findings: list[Finding]) -> None:
        ignore_path = self.repository / ".gitignore"
        if not ignore_path.is_file():
            findings.append(
                Finding(BLOCK, "privacy-policy", ".gitignore is missing.", ".gitignore")
            )
        else:
            rules = {
                line.strip()
                for line in ignore_path.read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.lstrip().startswith("#")
            }
            missing = sorted(_REQUIRED_IGNORE_RULES - rules)
            dangerous = sorted(rule for rule in rules if _dangerously_unanchored_private_rule(rule))
            if missing:
                findings.append(
                    Finding(
                        BLOCK,
                        "privacy-policy",
                        "Required private-workspace ignore rules are missing: "
                        + ", ".join(missing),
                        ".gitignore",
                    )
                )
            if dangerous:
                findings.append(
                    Finding(
                        BLOCK,
                        "privacy-policy",
                        "Private workspace ignore rules must be anchored to the repository root; "
                        "these broad rules can hide application source: " + ", ".join(dangerous),
                        ".gitignore",
                        "Prefix each private directory rule with '/', for example /workspace/.",
                    )
                )
            if not missing and not dangerous:
                findings.append(
                    Finding(PASS, "privacy-policy", "Private workspace ignore rules are present.")
                )
        for required in ("LICENSE", "SECURITY.md", "pyproject.toml"):
            if not (self.repository / required).is_file():
                findings.append(Finding(BLOCK, "repository-contract", f"{required} is missing."))
        if not self.is_git_repository:
            findings.append(
                Finding(
                    WARNING,
                    "git",
                    "This folder is not initialized as a Git repository yet.",
                    remediation="Run Safe-Push.bat once or: git init -b main",
                )
            )

    def _scan_required_public_workspace(
        self, paths: Iterable[Path], findings: list[Finding]
    ) -> None:
        candidates = {PurePosixPath(path.as_posix()) for path in paths}
        missing_files = sorted(
            path.as_posix()
            for path in _REQUIRED_PUBLIC_WORKSPACE_FILES
            if not (self.repository / Path(path.as_posix())).is_file()
        )
        excluded_files = sorted(
            path.as_posix()
            for path in _REQUIRED_PUBLIC_WORKSPACE_FILES
            if (self.repository / Path(path.as_posix())).is_file() and path not in candidates
        )
        if missing_files:
            findings.append(
                Finding(
                    BLOCK,
                    "repository-contract",
                    "Required workspace service source is missing: " + ", ".join(missing_files),
                )
            )
        if excluded_files:
            findings.append(
                Finding(
                    BLOCK,
                    "publish-scope",
                    "Required workspace service source is excluded from publication: "
                    + ", ".join(excluded_files),
                    ".gitignore",
                    "Use root-anchored private rules such as /workspace/, never workspace/.",
                )
            )
        if not missing_files and not excluded_files:
            findings.append(
                Finding(
                    PASS,
                    "publish-scope",
                    "The complete src/daedalus/workspace package is a publish candidate.",
                )
            )

    def _scan_paths(
        self, paths: Iterable[Path], findings: list[Finding], *, staged: bool
    ) -> None:
        scanned = 0
        for relative in paths:
            pure = PurePosixPath(relative.as_posix())
            suffix = Path(pure.name).suffix.casefold()
            if self._scan_path_policy(pure, findings):
                continue
            absolute = (self.repository / relative).resolve(strict=False)
            if not staged and (absolute.is_symlink() or self.repository not in absolute.parents):
                findings.append(
                    Finding(BLOCK, "path-boundary", "Symlink or path escape is not publishable.", pure.as_posix())
                )
                continue
            blob = self._staged_blob(relative) if staged else None
            try:
                if blob is not None:
                    size = len(blob)
                    chunks = self._byte_chunks(blob)
                else:
                    if not absolute.is_file():
                        continue
                    size = absolute.stat().st_size
                    chunks = self._file_chunks(absolute)
                scanned += 1
                self._scan_blob_stream(chunks, size, suffix, pure, findings)
            except OSError as exc:
                findings.append(Finding(BLOCK, "read", str(exc), pure.as_posix()))
        findings.append(Finding(INFO, "scope", f"Inspected {scanned} publish-candidate files."))

    def _scan_blob_stream(
        self,
        chunks: Iterable[bytes],
        size: int,
        suffix: str,
        pure: PurePosixPath,
        findings: list[Finding],
    ) -> None:
        if size > 25 * 1024 * 1024:
            findings.append(
                Finding(
                    BLOCK,
                    "large-file",
                    f"File is {size / (1024 * 1024):.1f} MiB; Safe Push limit is 25 MiB.",
                    pure.as_posix(),
                    "Use release assets or Git LFS after reviewing privacy and licensing.",
                )
            )
        if suffix not in _TEXT_SUFFIXES:
            return
        source = self._scan_text_chunks(
            chunks,
            findings,
            pure,
            collect=suffix == ".py" and size <= 3 * 1024 * 1024,
        )
        if suffix == ".py" and source is not None:
            text = source.decode("utf-8", errors="replace")
            try:
                ast.parse(text, filename=pure.as_posix())
            except SyntaxError as exc:
                findings.append(
                    Finding(
                        BLOCK,
                        "python-syntax",
                        f"Line {exc.lineno}: {exc.msg}",
                        pure.as_posix(),
                    )
                )

    def _scan_ruff(self, findings: list[Finding]) -> None:
        result = self._run([sys.executable, "-m", "ruff", "check", "."], timeout=300)
        if result.returncode:
            tail = (result.stdout + "\n" + result.stderr)[-3000:].strip()
            findings.append(
                Finding(
                    BLOCK,
                    "lint",
                    "Ruff checks failed.\n" + tail,
                    remediation="Fix lint findings before publishing.",
                )
            )
        else:
            findings.append(Finding(PASS, "lint", "Ruff checks passed."))

    def _scan_tests(self, findings: list[Finding]) -> None:
        environment = os.environ.copy()
        environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
        environment["QT_QPA_PLATFORM"] = "offscreen"
        command = [sys.executable, "-m", "pytest", "-q"]
        try:
            result = self._run(
                command,
                timeout=_TEST_SUITE_TIMEOUT_SECONDS,
                env=environment,
            )
        except subprocess.TimeoutExpired:
            findings.append(
                Finding(
                    BLOCK,
                    "tests",
                    "Test suite exceeded the 20-minute timeout.",
                    remediation="Investigate slow or hung tests and rerun the complete suite before publishing.",
                )
            )
            return
        if result.returncode:
            tail = (result.stdout + "\n" + result.stderr)[-3000:].strip()
            findings.append(
                Finding(BLOCK, "tests", "Test suite failed.\n" + tail, remediation="Fix tests before publishing.")
            )
        else:
            summary = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else "passed"
            findings.append(Finding(PASS, "tests", summary))

    def dependency_audit(self) -> tuple[list[dict[str, object]], str | None]:
        result = self._run(
            [
                sys.executable,
                "-m",
                "pip_audit",
                "-r",
                "requirements.txt",
                "--format=json",
                "--progress-spinner=off",
            ],
            timeout=600,
        )
        if "No module named pip_audit" in result.stderr:
            return [], "pip-audit is not installed in the active Daedalus environment."
        payload = result.stdout.strip()
        try:
            parsed = json.loads(payload) if payload else {}
        except json.JSONDecodeError:
            return [], (result.stderr or result.stdout or "pip-audit failed").strip()[-2000:]
        dependencies = parsed.get("dependencies", []) if isinstance(parsed, dict) else parsed
        vulnerabilities = []
        for dependency in dependencies or []:
            for vulnerability in dependency.get("vulns", []):
                vulnerabilities.append(
                    {
                        "name": dependency.get("name"),
                        "version": dependency.get("version"),
                        "id": vulnerability.get("id"),
                        "fix_versions": vulnerability.get("fix_versions") or [],
                        "aliases": vulnerability.get("aliases") or [],
                    }
                )
        if result.returncode not in (0, 1) and not vulnerabilities:
            return [], (result.stderr or "pip-audit failed").strip()[-2000:]
        return vulnerabilities, None

    def _scan_dependencies(self, findings: list[Finding]) -> None:
        vulnerabilities, error = self.dependency_audit()
        if error:
            findings.append(Finding(BLOCK, "dependency-audit", error))
            return
        if not vulnerabilities:
            findings.append(Finding(PASS, "dependency-audit", "No known PyPI advisory was found."))
            return
        for vulnerability in vulnerabilities:
            fixes = vulnerability["fix_versions"]
            findings.append(
                Finding(
                    BLOCK,
                    "dependency-audit",
                    f"{vulnerability['name']} {vulnerability['version']} is affected by "
                    f"{vulnerability['id']}; fixes: {', '.join(fixes) if fixes else 'none published'}.",
                    "requirements.txt",
                    "Run Safe-Push.bat and approve a bounded fix, or replace/remove the dependency.",
                )
            )

    def _origin_url(self) -> str | None:
        if not self.is_git_repository:
            return None
        result = self._git("remote", "get-url", "origin")
        if result.returncode:
            return None
        return result.stdout.strip() or None

    def _push_origin_error(self) -> str | None:
        origin_url = self._origin_url()
        if not origin_url:
            return "No GitHub origin is configured, so the requested push cannot be completed."
        if not self._github_repository_name(origin_url):
            return "Origin must be credential-free HTTPS on github.com."
        return None

    def _initial_publish_error(self, remote_name: str = "origin") -> str | None:
        result = self._git("ls-remote", "--refs", remote_name, timeout=120)
        if result.returncode:
            return "The remote could not be inspected to prove that this is its first publication."
        if result.stdout.strip():
            return "Initial publication is allowed only when the GitHub repository has no refs."
        return None

    def _upstream_divergence(self) -> tuple[int, int] | None:
        """Return upstream-only/local-only commit counts, or None without an upstream."""

        upstream = self._git("rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}")
        if upstream.returncode:
            return None
        result = self._git("rev-list", "--left-right", "--count", "@{u}...HEAD")
        if result.returncode:
            raise RuntimeError(result.stderr.strip() or "Git could not compare HEAD with its upstream.")
        fields = result.stdout.split()
        if len(fields) != 2:
            raise RuntimeError("Git returned malformed upstream divergence counts.")
        try:
            behind, ahead = (int(value) for value in fields)
        except ValueError as exc:
            raise RuntimeError("Git returned malformed upstream divergence counts.") from exc
        if behind < 0 or ahead < 0:
            raise RuntimeError("Git returned malformed upstream divergence counts.")
        return behind, ahead

    @staticmethod
    def _github_repository_name(origin_url: str) -> str | None:
        parsed = urlparse(origin_url)
        if parsed.scheme != "https" or parsed.hostname != "github.com" or parsed.username:
            return None
        parts = parsed.path.strip("/").removesuffix(".git").split("/")
        if len(parts) != 2 or not all(parts) or parsed.query or parsed.fragment:
            return None
        return "/".join(parts)

    def _scan_github_dependabot(self, findings: list[Finding]) -> None:
        origin_url = self._origin_url()
        if not origin_url:
            findings.append(
                Finding(INFO, "dependabot", "No credential-free GitHub origin is configured yet.")
            )
            return
        repository_name = self._github_repository_name(origin_url)
        if not repository_name:
            findings.append(
                Finding(
                    BLOCK,
                    "dependabot",
                    "The configured origin cannot be verified as credential-free HTTPS on github.com.",
                    remediation="Set origin to https://github.com/OWNER/REPOSITORY.git and rescan.",
                )
            )
            return
        gh = shutil.which("gh")
        if not gh:
            findings.append(
                Finding(
                    BLOCK,
                    "dependabot",
                    "GitHub CLI is unavailable, so repository Dependabot alerts were not read.",
                    remediation="Install GitHub CLI and run: gh auth login",
                )
            )
            return
        result = self._run(
            [
                gh,
                "api",
                f"repos/{repository_name}/dependabot/alerts?state=open&per_page=100",
                "--paginate",
                "--slurp",
            ],
            timeout=120,
        )
        if result.returncode:
            findings.append(
                Finding(
                    BLOCK,
                    "dependabot",
                    "GitHub Dependabot alerts could not be read completely; publication is blocked.",
                    remediation="Authenticate with `gh auth login`, verify alert access, and rescan.",
                )
            )
            return
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError:
            findings.append(
                Finding(BLOCK, "dependabot", "GitHub returned unreadable Dependabot alert data.")
            )
            return
        if not isinstance(payload, list):
            findings.append(
                Finding(BLOCK, "dependabot", "GitHub returned an unexpected Dependabot response.")
            )
            return
        if all(isinstance(page, list) for page in payload):
            alerts = [alert for page in payload for alert in page]
        elif all(isinstance(alert, dict) for alert in payload):
            # Accept a direct list for compatibility with older GitHub CLI output and test doubles.
            alerts = payload
        else:
            findings.append(
                Finding(BLOCK, "dependabot", "GitHub returned incomplete Dependabot pagination data.")
            )
            return
        blocking = 0
        for alert in alerts:
            if not isinstance(alert, dict):
                findings.append(
                    Finding(BLOCK, "dependabot", "GitHub returned a malformed Dependabot alert.")
                )
                return
            advisory = alert.get("security_advisory") or {}
            dependency_record = alert.get("dependency") or {}
            if not isinstance(advisory, dict) or not isinstance(dependency_record, dict):
                findings.append(
                    Finding(BLOCK, "dependabot", "GitHub returned a malformed Dependabot alert.")
                )
                return
            dependency = dependency_record.get("package") or {}
            if not isinstance(dependency, dict):
                findings.append(
                    Finding(BLOCK, "dependabot", "GitHub returned a malformed dependency record.")
                )
                return
            severity = str(advisory.get("severity", "unknown")).casefold()
            level = BLOCK if severity in {"critical", "high"} else WARNING
            blocking += level == BLOCK
            findings.append(
                Finding(
                    level,
                    "dependabot",
                    f"Open {severity} alert {advisory.get('ghsa_id', 'unknown')} for "
                    f"{dependency.get('name', 'dependency')}.",
                    remediation="Update on a branch, reinstall, run tests, and rescan; never force a silent fix.",
                )
            )
        if not alerts:
            findings.append(Finding(PASS, "dependabot", "GitHub reports no open Dependabot alerts."))
        elif not blocking:
            findings.append(Finding(INFO, "dependabot", "Open alerts are below the blocking threshold."))

    @staticmethod
    def _version_key(value: str) -> tuple[object, ...]:
        return tuple(int(part) if part.isdigit() else part for part in re.split(r"[.-]", value))

    def remediate_dependencies(self, *, include_github: bool = True) -> GuardReport:
        vulnerabilities, error = self.dependency_audit()
        if error:
            raise RuntimeError(error)
        fixable = {}
        for vulnerability in vulnerabilities:
            fixes = [str(value) for value in vulnerability["fix_versions"]]
            if fixes:
                name = str(vulnerability["name"])
                fixable[name] = max(fixes, key=self._version_key)
        if not fixable:
            return self.scan(
                include_tests=True,
                include_dependencies=True,
                include_github=include_github,
            )

        constraints = self.repository / "constraints-security.txt"
        existing = constraints.read_text(encoding="utf-8") if constraints.exists() else ""
        comments = [line for line in existing.splitlines() if line.lstrip().startswith("#")]
        lines = comments + [""] + [
            f"{name}>={version}  # added by Daedalus Release Guard"
            for name, version in sorted(fixable.items(), key=lambda item: item[0].casefold())
        ]
        constraints.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
        install = self._run(
            [sys.executable, "-m", "pip", "install", "--upgrade", "-r", "requirements-dev.txt"],
            timeout=1200,
        )
        if install.returncode:
            raise RuntimeError("Dependency upgrade failed:\n" + (install.stderr or install.stdout)[-4000:])
        return self.scan(
            include_tests=True,
            include_dependencies=True,
            include_github=include_github,
        )

    def write_report(self, report: GuardReport) -> Path:
        directory = self.repository / "reports" / "local"
        directory.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        path = directory / f"release-guard-{stamp}.json"
        path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
        return path

    def initialize_git(self) -> None:
        if self.is_git_repository:
            return
        result = self._git("init", "-b", "main")
        if result.returncode:
            raise RuntimeError(result.stderr.strip())

    def safe_push(
        self,
        message: str,
        *,
        auto_fix: bool = False,
        push: bool = True,
        initial_publish: bool = False,
    ) -> GuardReport:
        if not message.strip():
            raise ValueError("A non-empty commit message is required.")
        if initial_publish and not push:
            raise ValueError("Initial publication requires an actual push.")
        self.initialize_git()
        if push:
            origin_error = self._push_origin_error()
            initial_error = self._initial_publish_error() if initial_publish else None
            if origin_error or initial_error:
                report = self.scan()
                report.findings.append(
                    Finding(
                        BLOCK,
                        "initial-publish" if initial_error else "push",
                        initial_error or origin_error or "Push preflight failed.",
                        remediation=(
                            "Use initial publication only with a new, empty GitHub repository."
                            if initial_error
                            else "Configure a credential-free HTTPS GitHub origin, then run Safe Push again."
                        ),
                    )
                )
                self.write_report(report)
                return report

        include_github = not initial_publish
        report = self.scan(
            include_tests=True,
            include_dependencies=True,
            include_github=include_github,
        )
        if report.blocking and auto_fix and all(
            finding.check == "dependency-audit" for finding in report.blocking
        ):
            report = self.remediate_dependencies(include_github=include_github)
        self.write_report(report)
        if not report.ok:
            return report

        add = self._git("add", "-A")
        if add.returncode:
            raise RuntimeError(add.stderr.strip())
        staged_report = self.scan(
            staged=True,
            include_tests=True,
            include_dependencies=True,
            include_github=include_github,
        )
        self.write_report(staged_report)
        if not staged_report.ok:
            return staged_report

        if push:
            origin_error = self._push_origin_error()
            initial_error = self._initial_publish_error() if initial_publish else None
            if origin_error or initial_error:
                staged_report.findings.append(
                    Finding(
                        BLOCK,
                        "initial-publish" if initial_error else "push",
                        initial_error or origin_error or "Push preflight failed.",
                        remediation="Restore the reviewed GitHub origin and rerun Safe Push.",
                    )
                )
                return staged_report

        changed = self._git("diff", "--cached", "--quiet").returncode != 0
        if changed:
            commit = self._git("commit", "-m", message.strip(), timeout=180)
            if commit.returncode:
                raise RuntimeError(commit.stderr.strip() or commit.stdout.strip())
        if not push:
            return staged_report

        origin_error = self._push_origin_error()
        if origin_error:
            staged_report.findings.append(
                Finding(
                    BLOCK,
                    "push",
                    origin_error,
                    remediation="Restore the reviewed credential-free GitHub origin before pushing.",
                )
            )
            return staged_report
        if initial_publish:
            initial_error = self._initial_publish_error()
            if initial_error:
                staged_report.findings.append(
                    Finding(
                        BLOCK,
                        "initial-publish",
                        initial_error,
                        remediation="Fetch and review the now-populated remote before pushing.",
                    )
                )
                return staged_report

        try:
            divergence = self._upstream_divergence()
        except RuntimeError as exc:
            staged_report.findings.append(
                Finding(BLOCK, "push", str(exc), remediation="Fetch origin and run Safe Push again.")
            )
            return staged_report
        if divergence is not None:
            behind, ahead = divergence
            if behind and not ahead:
                staged_report.findings.append(
                    Finding(
                        BLOCK,
                        "push",
                        "The local branch has no outgoing commits and is behind its upstream.",
                        remediation="Fetch and reconcile the upstream branch before pushing.",
                    )
                )
                return staged_report
            if behind and ahead:
                staged_report.findings.append(
                    Finding(
                        BLOCK,
                        "push",
                        "The local branch has diverged from its upstream.",
                        remediation="Fetch and reconcile the upstream branch without force-pushing.",
                    )
                )
                return staged_report
            if not ahead:
                staged_report.findings.append(
                    Finding(PASS, "push", "No changes or outgoing commits; origin is up to date.")
                )
                return staged_report

        arguments = ["push"] if divergence is not None else ["push", "-u", "origin", "HEAD"]
        push_environment = os.environ.copy()
        if initial_publish:
            push_environment[_INITIAL_PUBLISH_ENV] = "1"
        else:
            push_environment.pop(_INITIAL_PUBLISH_ENV, None)
        pushed = self._git(*arguments, timeout=300, env=push_environment)
        if pushed.returncode:
            staged_report.findings.append(
                Finding(BLOCK, "push", pushed.stderr.strip() or "Git push failed.")
            )
        else:
            staged_report.findings.append(Finding(PASS, "push", "Commit pushed to GitHub."))
        return staged_report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Daedalus guarded GitHub publishing")
    parser.add_argument("--repo", default=".", help="source repository root")
    commands = parser.add_subparsers(dest="command", required=True)
    scan = commands.add_parser("scan", help="run privacy, tests, and advisory checks")
    scan.add_argument("--quick", action="store_true", help="skip tests and network advisory audit")
    commands.add_parser("fix", help="apply advisory-backed dependency constraints and retest")
    push = commands.add_parser("push", help="guard, commit, and push")
    push.add_argument("--message", required=True)
    push.add_argument(
        "--fix-dependencies",
        action="store_true",
        help="explicitly apply advisory-backed dependency constraints before retrying the gate",
    )
    push.add_argument(
        "--initial-publish",
        action="store_true",
        help="skip only the unavailable GitHub alert query for a proven-empty remote",
    )
    push.add_argument("--local-only", action="store_true")
    pre_push = commands.add_parser(
        "pre-push", help="scan immutable objects named by Git's pre-push protocol"
    )
    pre_push.add_argument("--remote-name", default="origin")
    commands.add_parser("init", help="initialize a local main-branch repository")
    return parser


def cli(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    guard = ReleaseGuard(args.repo)
    if args.command == "init":
        guard.initialize_git()
        print("Git repository is ready.")
        return 0
    if args.command == "fix":
        report = guard.remediate_dependencies()
    elif args.command == "pre-push":
        report = guard.scan_outgoing(
            sys.stdin,
            remote_name=args.remote_name,
            include_tests=True,
            include_dependencies=True,
            include_github=True,
            initial_publish=os.getenv(_INITIAL_PUBLISH_ENV) == "1",
        )
    elif args.command == "push":
        report = guard.safe_push(
            args.message,
            auto_fix=args.fix_dependencies,
            push=not args.local_only,
            initial_publish=args.initial_publish,
        )
    else:
        report = guard.scan(
            include_tests=not args.quick,
            include_dependencies=not args.quick,
            include_github=not args.quick,
        )
    print(report.format_text())
    print(f"Report: {guard.write_report(report)}")
    return 0 if report.ok else 2


if __name__ == "__main__":
    raise SystemExit(cli())
