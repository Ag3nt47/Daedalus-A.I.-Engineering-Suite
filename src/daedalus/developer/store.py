"""Crash-resumable SQLite persistence for developer sessions.

Each save appends an immutable revision and advances a small head pointer in the
same transaction.  A damaged head can therefore be rolled back to the most
recent revision whose checksum and schema still validate.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from contextlib import closing
from dataclasses import dataclass, replace
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from daedalus.developer.models import (
    ARTIFACT_FILENAMES,
    SCHEMA_VERSION,
    ArtifactKind,
    ArtifactRef,
    DeveloperSession,
    ExperienceMode,
    ProjectBrief,
    TaskKind,
    utc_now,
)

MAX_SESSION_BYTES = 1_048_576
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SESSION_KEYS = {
    "schema_version",
    "id",
    "project_root",
    "mode",
    "brief",
    "answers",
    "waivers",
    "artifacts",
    "created_utc",
    "updated_utc",
    "revision",
}
_BRIEF_KEYS = {
    "project_name",
    "outcome",
    "users",
    "task_kind",
    "inputs",
    "outputs",
    "success_metric",
    "constraints",
}
_ARTIFACT_KEYS = {"kind", "relative_path", "sha256", "created_utc"}


class SessionStoreError(RuntimeError):
    pass


class ConcurrentSessionUpdate(SessionStoreError):
    pass


class SessionIntegrityError(SessionStoreError):
    pass


class SessionCatalogState(StrEnum):
    """Health of a session head without mutating its revision history."""

    HEALTHY = "healthy"
    RECOVERY_REQUIRED = "recovery_required"
    UNRECOVERABLE = "unrecoverable"


@dataclass(frozen=True, slots=True)
class StoredRevision:
    session_id: str
    revision: int
    saved_utc: str
    sha256: str


@dataclass(frozen=True, slots=True)
class SessionCatalogEntry:
    """One durable catalog row, including damaged heads that need recovery.

    ``session`` is the current head when ``state`` is ``healthy``.  For a
    recoverable damaged head it is the most recent valid revision, exposed only
    as a display preview; callers must explicitly call ``recover_last_valid``
    before editing or saving it.
    """

    session_id: str
    state: SessionCatalogState
    head_revision: int | None
    recoverable_revision: int | None
    updated_utc: str
    session: DeveloperSession | None

    @property
    def needs_recovery(self) -> bool:
        return self.state != SessionCatalogState.HEALTHY


def _timestamp(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be an ISO timestamp")
    try:
        datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an ISO timestamp") from exc
    return value


def _strict_object(value: Any, keys: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    unknown = set(value) - keys
    missing = keys - set(value)
    if unknown:
        raise ValueError(f"{name} contains unsupported fields: {', '.join(sorted(unknown))}")
    if missing:
        raise ValueError(f"{name} is missing fields: {', '.join(sorted(missing))}")
    return value


def session_to_dict(session: DeveloperSession) -> dict[str, Any]:
    artifacts = []
    for item in session.artifacts:
        expected = ARTIFACT_FILENAMES[item.kind]
        if item.relative_path != expected:
            raise ValueError(f"artifact path must be the canonical filename for {item.kind.value}")
        if not _SHA256.fullmatch(item.sha256):
            raise ValueError("artifact SHA-256 must be 64 lowercase hexadecimal characters")
        artifacts.append(
            {
                "kind": item.kind.value,
                "relative_path": item.relative_path,
                "sha256": item.sha256,
                "created_utc": _timestamp(item.created_utc, "artifact created_utc"),
            }
        )
    brief = session.brief
    return {
        "schema_version": session.schema_version,
        "id": session.id,
        "project_root": session.project_root,
        "mode": session.mode.value,
        "brief": {
            "project_name": brief.project_name,
            "outcome": brief.outcome,
            "users": brief.users,
            "task_kind": brief.task_kind.value,
            "inputs": brief.inputs,
            "outputs": brief.outputs,
            "success_metric": brief.success_metric,
            "constraints": list(brief.constraints),
        },
        "answers": dict(session.answers),
        "waivers": dict(session.waivers),
        "artifacts": artifacts,
        "created_utc": _timestamp(session.created_utc, "created_utc"),
        "updated_utc": _timestamp(session.updated_utc, "updated_utc"),
        "revision": session.revision,
    }


def session_to_json(session: DeveloperSession, *, pretty: bool = True) -> str:
    separators = None if pretty else (",", ":")
    payload = json.dumps(
        session_to_dict(session),
        indent=2 if pretty else None,
        sort_keys=True,
        separators=separators,
        ensure_ascii=False,
    )
    if len(payload.encode("utf-8")) > MAX_SESSION_BYTES:
        raise ValueError("developer session exceeds the 1 MiB persistence limit")
    return payload


def session_from_dict(
    value: Any,
    *,
    expected_project_root: Path | None = None,
) -> DeveloperSession:
    data = _strict_object(value, _SESSION_KEYS, "developer session")
    if data["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"unsupported developer session schema: {data['schema_version']}")
    brief_data = _strict_object(data["brief"], _BRIEF_KEYS, "project brief")
    constraints = brief_data["constraints"]
    if not isinstance(constraints, list) or not all(isinstance(item, str) for item in constraints):
        raise ValueError("project brief constraints must be a list of strings")
    brief = ProjectBrief(
        project_name=brief_data["project_name"],
        outcome=brief_data["outcome"],
        users=brief_data["users"],
        task_kind=TaskKind(brief_data["task_kind"]),
        inputs=brief_data["inputs"],
        outputs=brief_data["outputs"],
        success_metric=brief_data["success_metric"],
        constraints=tuple(constraints),
    )
    project_root = Path(data["project_root"])
    if not project_root.is_absolute():
        raise ValueError("developer session project_root must be absolute")
    if expected_project_root is not None:
        expected = expected_project_root.resolve(strict=False)
        if project_root.resolve(strict=False) != expected:
            raise PermissionError("imported session belongs to a different private project")
    raw_artifacts = data["artifacts"]
    if not isinstance(raw_artifacts, list) or len(raw_artifacts) > len(ArtifactKind):
        raise ValueError("developer session artifacts are invalid")
    artifacts: list[ArtifactRef] = []
    seen: set[ArtifactKind] = set()
    for raw in raw_artifacts:
        item = _strict_object(raw, _ARTIFACT_KEYS, "artifact reference")
        kind = ArtifactKind(item["kind"])
        if kind in seen:
            raise ValueError(f"duplicate artifact reference: {kind.value}")
        seen.add(kind)
        if item["relative_path"] != ARTIFACT_FILENAMES[kind]:
            raise ValueError("artifact reference uses a non-canonical path")
        if not isinstance(item["sha256"], str) or not _SHA256.fullmatch(item["sha256"]):
            raise ValueError("artifact reference has an invalid SHA-256")
        artifacts.append(
            ArtifactRef(
                kind,
                item["relative_path"],
                item["sha256"],
                _timestamp(item["created_utc"], "artifact created_utc"),
            )
        )
    answers = data["answers"]
    waivers = data["waivers"]
    if not isinstance(answers, dict) or not isinstance(waivers, dict):
        raise ValueError("developer session answers and waivers must be JSON objects")
    revision = data["revision"]
    if not isinstance(revision, int) or isinstance(revision, bool) or revision < 0:
        raise ValueError("developer session revision must be a non-negative integer")
    return DeveloperSession(
        id=data["id"],
        project_root=str(project_root.resolve(strict=False)),
        mode=ExperienceMode(data["mode"]),
        brief=brief,
        answers=answers,
        waivers=waivers,
        artifacts=tuple(artifacts),
        created_utc=_timestamp(data["created_utc"], "created_utc"),
        updated_utc=_timestamp(data["updated_utc"], "updated_utc"),
        revision=revision,
        schema_version=SCHEMA_VERSION,
    )


def session_from_json(
    payload: str | bytes,
    *,
    expected_project_root: Path | None = None,
) -> DeveloperSession:
    raw = payload.encode("utf-8") if isinstance(payload, str) else payload
    if len(raw) > MAX_SESSION_BYTES:
        raise ValueError("developer session exceeds the 1 MiB import limit")
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError("developer session is not valid UTF-8 JSON") from exc
    return session_from_dict(data, expected_project_root=expected_project_root)


def _digest(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class DeveloperSessionStore:
    """Append-only session revisions with optimistic concurrency and recovery."""

    def __init__(self, database: Path, *, allowed_root: Path | None = None) -> None:
        candidate = Path(database)
        if candidate.exists() and candidate.is_symlink():
            raise PermissionError("developer session database cannot be a symbolic link")
        resolved = candidate.resolve(strict=False)
        if allowed_root is not None:
            root = Path(allowed_root).resolve(strict=False)
            try:
                resolved.relative_to(root)
            except ValueError as exc:
                raise PermissionError("developer session database escapes its allowed root") from exc
        if resolved.exists() and not resolved.is_file():
            raise ValueError("developer session database path must be a file")
        resolved.parent.mkdir(parents=True, exist_ok=True)
        self.database = resolved
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA busy_timeout=10000")
        return connection

    def _initialize(self) -> None:
        with closing(self._connect()) as connection, connection:
            # Journal mode persists in the database. Setting it once avoids an
            # unnecessary writer-lock negotiation on every concurrent reader.
            connection.execute("PRAGMA journal_mode=WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS developer_schema (
                    version INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS developer_sessions (
                    id TEXT PRIMARY KEY,
                    head_revision INTEGER NOT NULL,
                    updated_utc TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS developer_revisions (
                    session_id TEXT NOT NULL,
                    revision INTEGER NOT NULL,
                    saved_utc TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    event TEXT NOT NULL,
                    PRIMARY KEY(session_id, revision),
                    FOREIGN KEY(session_id) REFERENCES developer_sessions(id)
                );
                """
            )
            versions = connection.execute("SELECT version FROM developer_schema").fetchall()
            if not versions:
                connection.execute(
                    "INSERT INTO developer_schema(version) VALUES (?)", (SCHEMA_VERSION,)
                )
            elif len(versions) != 1 or int(versions[0]["version"]) != SCHEMA_VERSION:
                raise SessionStoreError("unsupported developer session database schema")

    def save(
        self,
        session: DeveloperSession,
        *,
        expected_revision: int | None = None,
        event: str = "saved",
    ) -> DeveloperSession:
        event = str(event).strip()
        if event not in {"saved", "imported", "recovered"}:
            raise ValueError("unsupported developer session event")
        with closing(self._connect()) as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT head_revision FROM developer_sessions WHERE id=?", (session.id,)
            ).fetchone()
            current = int(row["head_revision"]) if row is not None else 0
            maximum_row = connection.execute(
                "SELECT MAX(revision) AS maximum FROM developer_revisions WHERE session_id=?",
                (session.id,),
            ).fetchone()
            maximum = int(maximum_row["maximum"] or 0)
            expected = session.revision if expected_revision is None else int(expected_revision)
            if expected != current:
                raise ConcurrentSessionUpdate(
                    f"session revision changed: expected {expected}, current {current}"
                )
            # Recovery can move the head backward while retaining damaged revisions
            # as forensic evidence. Never reuse one of their revision numbers.
            next_revision = maximum + 1
            persisted = replace(session, revision=next_revision, updated_utc=utc_now())
            payload = session_to_json(persisted, pretty=False)
            digest = _digest(payload)
            if row is None:
                connection.execute(
                    "INSERT INTO developer_sessions(id, head_revision, updated_utc) VALUES (?, ?, ?)",
                    (session.id, next_revision, persisted.updated_utc),
                )
            else:
                connection.execute(
                    "UPDATE developer_sessions SET head_revision=?, updated_utc=? WHERE id=?",
                    (next_revision, persisted.updated_utc, session.id),
                )
            connection.execute(
                """INSERT INTO developer_revisions
                (session_id, revision, saved_utc, payload_json, sha256, event)
                VALUES (?, ?, ?, ?, ?, ?)""",
                (session.id, next_revision, persisted.updated_utc, payload, digest, event),
            )
            connection.commit()
        return persisted

    def _load_revision(self, session_id: str, revision: int) -> DeveloperSession:
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT payload_json, sha256 FROM developer_revisions WHERE session_id=? AND revision=?",
                (session_id, revision),
            ).fetchone()
        if row is None:
            raise KeyError(session_id)
        payload = str(row["payload_json"])
        if _digest(payload) != row["sha256"]:
            raise SessionIntegrityError("developer session revision checksum failed")
        try:
            session = session_from_json(payload)
        except (TypeError, ValueError, PermissionError) as exc:
            raise SessionIntegrityError("developer session revision schema failed") from exc
        if session.id != session_id or session.revision != revision:
            raise SessionIntegrityError("developer session revision identity failed")
        return session

    def load(self, session_id: str) -> DeveloperSession:
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT head_revision FROM developer_sessions WHERE id=?", (session_id,)
            ).fetchone()
        if row is None:
            raise KeyError(session_id)
        return self._load_revision(session_id, int(row["head_revision"]))

    def _last_valid_revision(self, session_id: str) -> DeveloperSession | None:
        """Return the newest checksum/schema-valid revision without moving the head."""

        with closing(self._connect()) as connection:
            rows = connection.execute(
                """SELECT revision, payload_json, sha256 FROM developer_revisions
                WHERE session_id=? ORDER BY revision DESC""",
                (session_id,),
            ).fetchall()
        for row in rows:
            payload = str(row["payload_json"])
            if _digest(payload) != row["sha256"]:
                continue
            try:
                candidate = session_from_json(payload)
                revision = int(row["revision"])
            except (TypeError, ValueError, PermissionError):
                continue
            if candidate.id == session_id and candidate.revision == revision:
                return candidate
        return None

    def list_catalog(self) -> tuple[SessionCatalogEntry, ...]:
        """List every session row while isolating per-session head corruption.

        Healthy entries expose their current session.  A damaged entry remains
        visible with its stable ID and, when one exists, a last-valid preview.
        This method never rewinds a head; recovery remains an explicit action.
        """

        with closing(self._connect()) as connection:
            rows = connection.execute(
                """SELECT id, head_revision, updated_utc FROM developer_sessions
                ORDER BY updated_utc DESC, id"""
            ).fetchall()
        entries: list[SessionCatalogEntry] = []
        for row in rows:
            session_id = str(row["id"])
            updated_utc = str(row["updated_utc"])
            head_revision: int | None = None
            try:
                head_revision = int(row["head_revision"])
                current = self._load_revision(session_id, head_revision)
            except (KeyError, SessionIntegrityError, TypeError, ValueError):
                try:
                    preview = self._last_valid_revision(session_id)
                except (sqlite3.DatabaseError, OSError):
                    preview = None
                entries.append(
                    SessionCatalogEntry(
                        session_id,
                        (
                            SessionCatalogState.RECOVERY_REQUIRED
                            if preview is not None
                            else SessionCatalogState.UNRECOVERABLE
                        ),
                        head_revision,
                        preview.revision if preview is not None else None,
                        updated_utc,
                        preview,
                    )
                )
                continue
            entries.append(
                SessionCatalogEntry(
                    session_id,
                    SessionCatalogState.HEALTHY,
                    head_revision,
                    head_revision,
                    updated_utc,
                    current,
                )
            )
        return tuple(entries)

    def list_sessions(self) -> tuple[DeveloperSession, ...]:
        """Return healthy heads only; use :meth:`list_catalog` for recovery UI."""

        return tuple(
            entry.session
            for entry in self.list_catalog()
            if entry.state == SessionCatalogState.HEALTHY and entry.session is not None
        )

    def history(self, session_id: str) -> tuple[StoredRevision, ...]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """SELECT session_id, revision, saved_utc, sha256
                FROM developer_revisions WHERE session_id=? ORDER BY revision""",
                (session_id,),
            ).fetchall()
        if not rows:
            raise KeyError(session_id)
        return tuple(
            StoredRevision(
                str(row["session_id"]),
                int(row["revision"]),
                str(row["saved_utc"]),
                str(row["sha256"]),
            )
            for row in rows
        )

    def recover_last_valid(self, session_id: str) -> DeveloperSession:
        with closing(self._connect()) as connection:
            present = connection.execute(
                "SELECT 1 FROM developer_sessions WHERE id=?", (session_id,)
            ).fetchone()
        if present is None:
            raise KeyError(session_id)
        recovered = self._last_valid_revision(session_id)
        if recovered is None:
            raise SessionIntegrityError("no valid committed developer session revision remains")
        with closing(self._connect()) as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "UPDATE developer_sessions SET head_revision=?, updated_utc=? WHERE id=?",
                (recovered.revision, utc_now(), session_id),
            )
            connection.commit()
        return recovered

    def export_json(self, session_id: str) -> str:
        return session_to_json(self.load(session_id), pretty=True)

    def import_json(
        self,
        payload: str | bytes,
        *,
        expected_project_root: Path | None = None,
        allow_replace: bool = False,
    ) -> DeveloperSession:
        imported = session_from_json(payload, expected_project_root=expected_project_root)
        try:
            current = self.load(imported.id)
        except KeyError:
            return self.save(replace(imported, revision=0), event="imported")
        if not allow_replace:
            raise FileExistsError(f"developer session already exists: {imported.id}")
        replacement = replace(imported, revision=current.revision)
        return self.save(replacement, expected_revision=current.revision, event="imported")


__all__ = [
    "ConcurrentSessionUpdate",
    "DeveloperSessionStore",
    "MAX_SESSION_BYTES",
    "SessionCatalogEntry",
    "SessionCatalogState",
    "SessionIntegrityError",
    "SessionStoreError",
    "StoredRevision",
    "session_from_dict",
    "session_from_json",
    "session_to_dict",
    "session_to_json",
]
