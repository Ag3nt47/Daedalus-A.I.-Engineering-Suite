"""SQLite-backed training run journal with an explicit lifecycle."""

from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_TRANSITIONS = {
    "queued": {"running", "cancelled", "failed"},
    "running": {"completed", "cancelled", "failed"},
    "completed": set(),
    "cancelled": set(),
    "failed": set(),
}


def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True, slots=True)
class RunRecord:
    id: str
    project: str
    dataset: str
    status: str
    created_utc: str
    updated_utc: str
    config: dict[str, Any]
    metrics: dict[str, float]
    checkpoint: str | None
    error: str | None


class RunRegistry:
    def __init__(self, database: Path) -> None:
        self.database = database.resolve(strict=False)
        self.database.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _initialize(self) -> None:
        with closing(self._connect()) as connection, connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS schema_info (
                    version INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY,
                    project TEXT NOT NULL,
                    dataset TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_utc TEXT NOT NULL,
                    updated_utc TEXT NOT NULL,
                    config_json TEXT NOT NULL,
                    metrics_json TEXT NOT NULL DEFAULT '{}',
                    checkpoint TEXT,
                    error TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_runs_updated ON runs(updated_utc DESC);
                CREATE INDEX IF NOT EXISTS idx_runs_project ON runs(project, updated_utc DESC);
                CREATE TABLE IF NOT EXISTS run_events (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL REFERENCES runs(id),
                    timestamp_utc TEXT NOT NULL,
                    event TEXT NOT NULL,
                    detail_json TEXT NOT NULL DEFAULT '{}'
                );
                """
            )
            count = connection.execute("SELECT COUNT(*) FROM schema_info").fetchone()[0]
            if count == 0:
                connection.execute("INSERT INTO schema_info(version) VALUES (1)")

    def create_run(self, project: str, dataset: str, config: dict[str, Any]) -> str:
        if not project.strip() or not dataset.strip():
            raise ValueError("Project and dataset names are required.")
        run_id = str(uuid.uuid4())
        timestamp = _now()
        config_json = json.dumps(config, sort_keys=True, separators=(",", ":"))
        with closing(self._connect()) as connection, connection:
            connection.execute(
                """INSERT INTO runs
                (id, project, dataset, status, created_utc, updated_utc, config_json)
                VALUES (?, ?, ?, 'queued', ?, ?, ?)""",
                (run_id, project.strip(), dataset.strip(), timestamp, timestamp, config_json),
            )
            self._event(connection, run_id, "created", {"status": "queued"})
        return run_id

    @staticmethod
    def _event(
        connection: sqlite3.Connection, run_id: str, event: str, detail: dict[str, Any]
    ) -> None:
        connection.execute(
            "INSERT INTO run_events(run_id, timestamp_utc, event, detail_json) VALUES (?, ?, ?, ?)",
            (run_id, _now(), event, json.dumps(detail, sort_keys=True, separators=(",", ":"))),
        )

    def transition(
        self,
        run_id: str,
        status: str,
        *,
        metrics: dict[str, float] | None = None,
        checkpoint: str | None = None,
        error: str | None = None,
    ) -> None:
        if status not in _TRANSITIONS:
            raise ValueError(f"Unknown run status: {status}")
        with closing(self._connect()) as connection, connection:
            row = connection.execute("SELECT status FROM runs WHERE id = ?", (run_id,)).fetchone()
            if row is None:
                raise KeyError(run_id)
            current = str(row["status"])
            if status not in _TRANSITIONS[current]:
                raise ValueError(f"Invalid run transition: {current} -> {status}")
            if status == "failed" and not error:
                raise ValueError("A failed run requires a redacted error summary.")
            metrics_json = json.dumps(metrics or {}, sort_keys=True, separators=(",", ":"))
            connection.execute(
                """UPDATE runs SET status=?, updated_utc=?, metrics_json=?, checkpoint=?, error=?
                WHERE id=?""",
                (status, _now(), metrics_json, checkpoint, error, run_id),
            )
            self._event(
                connection,
                run_id,
                "transition",
                {"from": current, "to": status, "metrics": metrics or {}},
            )

    def record_metrics(self, run_id: str, metrics: dict[str, float]) -> None:
        clean = {str(name): float(value) for name, value in metrics.items()}
        with closing(self._connect()) as connection, connection:
            row = connection.execute(
                "SELECT status, metrics_json FROM runs WHERE id=?", (run_id,)
            ).fetchone()
            if row is None:
                raise KeyError(run_id)
            if row["status"] not in {"queued", "running"}:
                raise ValueError("Metrics cannot be appended to a terminal run.")
            combined = json.loads(row["metrics_json"])
            combined.update(clean)
            connection.execute(
                "UPDATE runs SET metrics_json=?, updated_utc=? WHERE id=?",
                (json.dumps(combined, sort_keys=True), _now(), run_id),
            )
            self._event(connection, run_id, "metrics", clean)

    def get(self, run_id: str) -> RunRecord:
        with closing(self._connect()) as connection, connection:
            row = connection.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
        if row is None:
            raise KeyError(run_id)
        return self._record(row)

    def list_runs(self, *, project: str | None = None, limit: int = 100) -> list[RunRecord]:
        if not 1 <= limit <= 1000:
            raise ValueError("limit must be between 1 and 1000")
        with closing(self._connect()) as connection, connection:
            if project is None:
                rows = connection.execute(
                    "SELECT * FROM runs ORDER BY updated_utc DESC LIMIT ?", (limit,)
                ).fetchall()
            else:
                rows = connection.execute(
                    "SELECT * FROM runs WHERE project=? ORDER BY updated_utc DESC LIMIT ?",
                    (project, limit),
                ).fetchall()
        return [self._record(row) for row in rows]

    @staticmethod
    def _record(row: sqlite3.Row) -> RunRecord:
        return RunRecord(
            id=row["id"],
            project=row["project"],
            dataset=row["dataset"],
            status=row["status"],
            created_utc=row["created_utc"],
            updated_utc=row["updated_utc"],
            config=json.loads(row["config_json"]),
            metrics=json.loads(row["metrics_json"]),
            checkpoint=row["checkpoint"],
            error=row["error"],
        )

    def events(self, run_id: str) -> list[dict[str, Any]]:
        with closing(self._connect()) as connection, connection:
            rows = connection.execute(
                "SELECT timestamp_utc, event, detail_json FROM run_events "
                "WHERE run_id=? ORDER BY sequence",
                (run_id,),
            ).fetchall()
        return [
            {
                "timestamp_utc": row["timestamp_utc"],
                "event": row["event"],
                "detail": json.loads(row["detail_json"]),
            }
            for row in rows
        ]
