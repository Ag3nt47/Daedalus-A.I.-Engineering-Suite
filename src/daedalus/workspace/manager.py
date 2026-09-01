"""External workspace management with strict path boundaries.

The application source repository is public. User projects, datasets, model
weights, and training journals are private by default and therefore live in a
separate workspace root.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

WORKSPACE_SCHEMA = 1
_SAFE_NAME = re.compile(r"[^A-Za-z0-9._ -]+")


def _expanded_path(raw: str | os.PathLike[str]) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(str(raw)))).resolve(strict=False)


def _atomic_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def safe_project_name(value: str) -> str:
    """Return a filesystem-safe project name or raise for an empty/dot name."""

    cleaned = _SAFE_NAME.sub("-", value.strip()).strip(" .-")
    if not cleaned or cleaned in {".", ".."}:
        raise ValueError("Project name must contain at least one letter or number.")
    return cleaned[:80]


@dataclass(slots=True)
class WorkspaceManager:
    source_root: Path
    workspace_root: Path
    backup_root: Path
    require_backup_volume_preflight: bool = False

    @classmethod
    def from_environment(cls) -> "WorkspaceManager":
        source_root = Path(__file__).resolve().parents[3]
        # Keep generated work visibly separate from the public Desktop source.
        # Using the profile root also avoids Windows Controlled Folder Access
        # denying Python writes inside Documents on otherwise healthy systems.
        default_workspace = Path.home() / "Daedalus Workspaces"
        workspace_root = _expanded_path(
            os.getenv("DAEDALUS_WORKSPACE_ROOT", str(default_workspace))
        )
        backup_root = _expanded_path(
            os.getenv("DAEDALUS_BACKUP_ROOT", r"F:\Daedalus-Backups\DaedalusAI")
        )
        return cls(
            source_root.resolve(strict=False),
            workspace_root,
            backup_root,
            require_backup_volume_preflight=True,
        )

    @property
    def projects_dir(self) -> Path:
        return self.workspace_root / "projects"

    @property
    def datasets_dir(self) -> Path:
        return self.workspace_root / "datasets"

    @property
    def checkpoints_dir(self) -> Path:
        return self.workspace_root / "checkpoints"

    @property
    def logs_dir(self) -> Path:
        return self.workspace_root / "logs"

    @property
    def runs_dir(self) -> Path:
        return self.workspace_root / "training-runs"

    @property
    def settings_dir(self) -> Path:
        return self.workspace_root / ".daedalus"

    @property
    def marker_path(self) -> Path:
        return self.workspace_root / ".daedalus-workspace.json"

    def bootstrap(self) -> None:
        """Create the workspace tree and a stable ownership marker."""

        self._assert_separate_from_source()
        for directory in (
            self.workspace_root,
            self.projects_dir,
            self.datasets_dir,
            self.checkpoints_dir,
            self.logs_dir,
            self.runs_dir,
            self.settings_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

        if self.marker_path.exists():
            marker = json.loads(self.marker_path.read_text(encoding="utf-8"))
            if marker.get("kind") != "daedalus-user-workspace":
                raise RuntimeError(f"Workspace marker is invalid: {self.marker_path}")
        else:
            _atomic_json(
                self.marker_path,
                {
                    "kind": "daedalus-user-workspace",
                    "schema": WORKSPACE_SCHEMA,
                    "id": str(uuid.uuid4()),
                    "created_utc": datetime.now(UTC).isoformat(),
                    "private_by_default": True,
                },
            )

        welcome = self.workspace_root / "README.txt"
        if not welcome.exists():
            welcome.write_text(
                "Daedalus private workspace\n"
                "==========================\n\n"
                "Projects, datasets, checkpoints, and run logs here are separate from the "
                "public source repository. Safe Push must never publish this directory.\n",
                encoding="utf-8",
            )

    def _assert_separate_from_source(self) -> None:
        source = self.source_root.resolve(strict=False)
        workspace = self.workspace_root.resolve(strict=False)
        if source == workspace or source in workspace.parents or workspace in source.parents:
            raise ValueError(
                "The user workspace must be outside the Daedalus source repository. "
                f"Source={source}; workspace={workspace}"
            )

    def update_workspace_root(self, path: str | os.PathLike[str]) -> None:
        candidate = _expanded_path(path)
        original = self.workspace_root
        self.workspace_root = candidate
        try:
            self.bootstrap()
        except Exception:
            self.workspace_root = original
            raise
        os.environ["DAEDALUS_WORKSPACE_ROOT"] = str(candidate)

    def resolve_user_path(
        self, path: str | os.PathLike[str], *, must_exist: bool = False
    ) -> Path:
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = self.workspace_root / candidate
        candidate = candidate.resolve(strict=False)
        root = self.workspace_root.resolve(strict=False)
        if candidate != root and root not in candidate.parents:
            raise PermissionError(f"Path escapes the private workspace: {candidate}")
        if must_exist and not candidate.exists():
            raise FileNotFoundError(candidate)
        return candidate

    def list_projects(self) -> list[Path]:
        self.bootstrap()
        return sorted(
            (
                path
                for path in self.projects_dir.iterdir()
                if path.is_dir() and not path.name.startswith(".")
            ),
            key=lambda path: path.name.casefold(),
        )

    def create_project(self, name: str, template: str = "minimal") -> Path:
        """Validate, stage, and atomically publish a private starter project."""

        project_name = safe_project_name(name)
        templates = {
            "minimal": _MINIMAL_PROJECT,
            "xor": _XOR_PROJECT,
            "regression": _REGRESSION_PROJECT,
        }
        if template not in templates:
            raise ValueError(f"Unknown project template: {template}")

        # Validation above intentionally precedes bootstrap so an invalid request
        # cannot create or modify a workspace tree.
        self.bootstrap()
        project = self.resolve_user_path(self.projects_dir / project_name)
        if project.exists():
            raise FileExistsError(f"A project named '{project_name}' already exists.")
        staging = self.resolve_user_path(
            self.projects_dir / f".{project_name}.creating-{uuid.uuid4().hex}"
        )
        try:
            created_utc = datetime.now(UTC).isoformat()
            project_id = str(uuid.uuid4())
            staging.mkdir(parents=False)
            for child in ("data", "checkpoints", "logs", "runs"):
                (staging / child).mkdir()
            (staging / "main.py").write_text(templates[template], encoding="utf-8")
            (staging / "README.md").write_text(
                f"# {project_name}\n\n"
                f"Created by Daedalus from the `{template}` template. This project is private "
                "and lives outside the public suite repository.\n",
                encoding="utf-8",
            )
            (staging / "logs" / "README.txt").write_text(
                "Daedalus project logs\n"
                "=====================\n\n"
                "Place project .log, .out, .err, or logs/*.txt files here. The project "
                "diagnostics tool can scan them live or on demand without executing code. "
                "Diagnostic reports identify the matching category and line number but do not "
                "copy private log values.\n",
                encoding="utf-8",
            )
            _atomic_json(
                staging / "project.json",
                {
                    "schema": 1,
                    "kind": "daedalus-ai-project",
                    "id": project_id,
                    "name": project_name,
                    "template": template,
                    "created_utc": created_utc,
                    "entrypoint": "main.py",
                    "standards": {
                        "project_manifest": "pyproject.toml",
                        "environment_lock": "environment.lock.json",
                        "environment_snapshot": "ENVIRONMENT_SNAPSHOT.json",
                        "run_manifest_pattern": "runs/<run-id>.manifest.json",
                    },
                },
            )
            # All standards files are written inside the unpublished staging
            # directory. A failure therefore removes the entire candidate and
            # can never expose a partially initialized project.
            from daedalus.services.project_standards import initialize_staged_project

            initialize_staged_project(
                staging,
                project_name=project_name,
                template=template,
            )
            if project.exists():
                raise FileExistsError(f"A project named '{project_name}' already exists.")
            for attempt in range(5):
                try:
                    staging.rename(project)
                    break
                except PermissionError:
                    if attempt == 4:
                        raise
                    # Windows indexing/antivirus can briefly retain a handle
                    # after the final staged file closes. The target remains
                    # nonexistent, so a bounded retry preserves atomic publish.
                    time.sleep(0.05 * (attempt + 1))
        except Exception:
            if staging.exists():
                shutil.rmtree(staging)
            raise
        return project

    def ensure_project_standards(
        self, project: str | os.PathLike[str]
    ) -> tuple[Path, ...]:
        """Create only missing professional-baseline files for a private project."""

        from daedalus.services.project_standards import ProjectStandardsService

        return ProjectStandardsService(self).initialize_missing(project)

    def ensure_project_logs(self, project: str | os.PathLike[str]) -> Path:
        """Return a project's local log folder, creating the standard folder if absent.

        This is a narrowly scoped compatibility step for projects created before
        automatic log folders were introduced. Existing files are never replaced.
        """

        original = Path(project)
        if original.is_symlink() or self.projects_dir.is_symlink():
            raise PermissionError("Project log paths cannot be symbolic links.")
        resolved = self.resolve_user_path(original, must_exist=True)
        projects = self.projects_dir.resolve(strict=True)
        if resolved.parent != projects or not resolved.is_dir():
            raise PermissionError("Project logs require a direct private project directory.")
        logs = resolved / "logs"
        if logs.is_symlink():
            raise PermissionError("Project log folders cannot be symbolic links.")
        logs.mkdir(exist_ok=True)
        if not logs.is_dir():
            raise ValueError("Project logs path is not a directory.")
        guide = logs / "README.txt"
        if not guide.exists():
            with guide.open("x", encoding="utf-8", newline="\n") as handle:
                handle.write(
                    "Daedalus project logs\n"
                    "=====================\n\n"
                    "Place project .log, .out, .err, or logs/*.txt files here. The project "
                    "diagnostics tool can scan them live or on demand without executing code. "
                    "Diagnostic reports identify the matching category and line number but do "
                    "not copy private log values.\n"
                )
        return logs

    @staticmethod
    def open_in_file_manager(path: str | os.PathLike[str]) -> None:
        target = Path(path).resolve(strict=False)
        if sys.platform == "win32":
            os.startfile(str(target))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(target)])
        else:
            subprocess.Popen(["xdg-open", str(target)])

    def describe(self) -> dict[str, Any]:
        self.bootstrap()
        marker = json.loads(self.marker_path.read_text(encoding="utf-8"))
        return {
            "source_root": str(self.source_root),
            "workspace_root": str(self.workspace_root),
            "backup_root": str(self.backup_root),
            "workspace_id": marker["id"],
            "project_count": len(self.list_projects()),
        }


_MINIMAL_PROJECT = '''"""A blank Daedalus experiment."""

import numpy as np


def main() -> None:
    rng = np.random.default_rng(47)
    print("Daedalus project ready.", rng.normal(size=3))


if __name__ == "__main__":
    main()
'''


_XOR_PROJECT = '''"""Train a tiny network on XOR using Daedalus primitives."""

from daedalus.engine.datasets import xor_dataset
from daedalus.engine.trainer import Trainer
from daedalus.layers import Linear, ReLU, Sequential
from daedalus.losses import MSELoss
from daedalus.optim import Adam


def main() -> None:
    x, y = xor_dataset()
    model = Sequential(Linear(2, 8, seed=47), ReLU(), Linear(8, 1, seed=48))
    trainer = Trainer(model, MSELoss(), Adam(model.parameters(), learning_rate=0.03))
    history = trainer.fit(x, y, epochs=1000, batch_size=4, seed=47)
    print(f"final loss: {history.loss[-1]:.6f}")
    print(model(x).data.round(3))


if __name__ == "__main__":
    main()
'''


_REGRESSION_PROJECT = '''"""Fit a deterministic one-dimensional regression dataset."""

from daedalus.engine.datasets import regression_dataset
from daedalus.engine.trainer import Trainer
from daedalus.layers import Linear, Sequential
from daedalus.losses import MSELoss
from daedalus.optim import Adam


def main() -> None:
    x, y = regression_dataset(samples=128, seed=47)
    model = Sequential(Linear(1, 16, seed=47), Linear(16, 1, seed=48))
    trainer = Trainer(model, MSELoss(), Adam(model.parameters(), learning_rate=0.01))
    history = trainer.fit(x, y, epochs=300, batch_size=16, seed=47)
    print(f"final loss: {history.loss[-1]:.6f}")


if __name__ == "__main__":
    main()
'''
