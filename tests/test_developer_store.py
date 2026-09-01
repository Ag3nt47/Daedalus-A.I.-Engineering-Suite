from __future__ import annotations

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest

from daedalus.developer import (
    ConcurrentSessionUpdate,
    DeveloperAdvisor,
    DeveloperSessionStore,
    ExperienceMode,
    ProjectBrief,
    RecoveryPlanner,
    SessionCatalogState,
    SessionIntegrityError,
    TaskKind,
    session_from_json,
    session_to_json,
)


def make_session(project):
    brief = ProjectBrief(
        "Local Classifier",
        "Classify a local numeric record",
        "A trained reviewer",
        TaskKind.CLASSIFICATION,
        "Four finite numeric features",
        "One of two reviewed labels",
        "Held-out F1 at least 0.8",
    )
    return DeveloperAdvisor().start(project, brief, ExperienceMode.BUILDER)


def test_store_round_trip_revision_history_and_export(tmp_path) -> None:
    project = tmp_path / "workspace" / "projects" / "demo"
    project.mkdir(parents=True)
    store = DeveloperSessionStore(
        tmp_path / "workspace" / ".daedalus" / "developer.sqlite3",
        allowed_root=tmp_path / "workspace",
    )
    session = store.save(make_session(project))
    updated = DeveloperAdvisor().answer(session, "data_source", "Licensed local CSV")
    updated = store.save(updated)

    assert session.revision == 1
    assert updated.revision == 2
    assert store.load(session.id) == updated
    assert [item.revision for item in store.history(session.id)] == [1, 2]
    exported = store.export_json(session.id)
    decoded = session_from_json(exported, expected_project_root=project)
    assert decoded == updated
    assert json.loads(exported)["schema_version"] == 1


def test_optimistic_concurrency_rejects_stale_update(tmp_path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    store = DeveloperSessionStore(tmp_path / "sessions.sqlite3")
    first = store.save(make_session(project))
    store.save(DeveloperAdvisor().answer(first, "data_source", "Dataset A"))

    with pytest.raises(ConcurrentSessionUpdate):
        store.save(DeveloperAdvisor().answer(first, "data_source", "Stale Dataset B"))


def test_corrupt_head_recovers_last_valid_and_never_reuses_revision(tmp_path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    store = DeveloperSessionStore(tmp_path / "sessions.sqlite3")
    first = store.save(make_session(project))
    second = store.save(DeveloperAdvisor().answer(first, "data_source", "Dataset A"))
    with sqlite3.connect(store.database) as connection:
        connection.execute(
            "UPDATE developer_revisions SET payload_json='corrupt' WHERE session_id=? AND revision=?",
            (second.id, second.revision),
        )

    with pytest.raises(SessionIntegrityError):
        store.load(second.id)
    recovered = store.recover_last_valid(second.id)
    assert recovered.revision == 1
    third = store.save(DeveloperAdvisor().answer(recovered, "data_source", "Recovered Dataset"))
    assert third.revision == 3
    assert [item.revision for item in store.history(second.id)] == [1, 2, 3]


def test_catalog_isolates_corrupt_heads_and_keeps_recovery_ids(tmp_path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    store = DeveloperSessionStore(tmp_path / "sessions.sqlite3")
    healthy = store.save(make_session(project))
    damaged_first = store.save(make_session(project))
    damaged_head = store.save(
        DeveloperAdvisor().answer(damaged_first, "data_source", "Licensed local CSV")
    )
    with sqlite3.connect(store.database) as connection:
        connection.execute(
            "UPDATE developer_revisions SET payload_json='corrupt' "
            "WHERE session_id=? AND revision=?",
            (damaged_head.id, damaged_head.revision),
        )

    # Compatibility listing stays usable and contains only valid current heads.
    assert [session.id for session in store.list_sessions()] == [healthy.id]

    catalog = {entry.session_id: entry for entry in store.list_catalog()}
    assert catalog[healthy.id].state == SessionCatalogState.HEALTHY
    damaged = catalog[damaged_head.id]
    assert damaged.state == SessionCatalogState.RECOVERY_REQUIRED
    assert damaged.head_revision == damaged_head.revision
    assert damaged.recoverable_revision == damaged_first.revision
    assert damaged.session == damaged_first
    assert damaged.needs_recovery

    recovered = store.recover_last_valid(damaged.session_id)
    assert recovered == damaged_first
    assert {
        entry.session_id: entry.state for entry in store.list_catalog()
    }[damaged.session_id] == SessionCatalogState.HEALTHY


def test_import_is_versioned_root_bound_and_non_overwriting(tmp_path) -> None:
    project = tmp_path / "workspace" / "projects" / "one"
    other = tmp_path / "workspace" / "projects" / "other"
    project.mkdir(parents=True)
    other.mkdir()
    source_store = DeveloperSessionStore(tmp_path / "source.sqlite3")
    session = source_store.save(make_session(project))
    payload = source_store.export_json(session.id)
    destination_store = DeveloperSessionStore(tmp_path / "destination.sqlite3")

    imported = destination_store.import_json(payload, expected_project_root=project)
    assert imported.revision == 1
    with pytest.raises(FileExistsError):
        destination_store.import_json(payload, expected_project_root=project)
    with pytest.raises(PermissionError):
        DeveloperSessionStore(tmp_path / "another.sqlite3").import_json(
            payload, expected_project_root=other
        )


def test_json_schema_extra_fields_size_and_secrets_are_rejected(tmp_path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    payload = session_to_json(make_session(project))
    raw = json.loads(payload)
    raw["surprise"] = True
    with pytest.raises(ValueError):
        session_from_json(json.dumps(raw))

    raw = json.loads(payload)
    raw["schema_version"] = 999
    with pytest.raises(ValueError):
        session_from_json(json.dumps(raw))

    raw = json.loads(payload)
    raw["answers"]["data_source"] = "sk-ABCDEFGHIJKLMNOPQRSTUVWXYZ123456"
    credential_payload = json.dumps(raw)
    with pytest.raises(ValueError) as captured:
        session_from_json(credential_payload)
    assert "ABCDEFGHIJKLMNOPQRSTUVWXYZ" not in str(captured.value)

    with pytest.raises(ValueError):
        session_from_json(b"{" + b" " * 1_048_576 + b"}")


def test_database_path_is_confined_to_allowed_root(tmp_path) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    with pytest.raises(PermissionError):
        DeveloperSessionStore(tmp_path / "outside.sqlite3", allowed_root=allowed)


def test_import_replace_appends_revision_instead_of_rewriting_history(tmp_path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    store = DeveloperSessionStore(tmp_path / "sessions.sqlite3")
    original = store.save(make_session(project))
    payload = session_to_json(
        DeveloperAdvisor().answer(original, "data_source", "Imported dataset description")
    )

    replaced = store.import_json(payload, expected_project_root=project, allow_replace=True)
    assert replaced.revision == 2
    assert len(store.history(original.id)) == 2


def test_store_closes_every_connection_and_releases_database_file(
    tmp_path, monkeypatch
) -> None:
    original_connect = sqlite3.connect
    opened: list[sqlite3.Connection] = []

    def tracked_connect(*args, **kwargs):
        connection = original_connect(*args, **kwargs)
        opened.append(connection)
        return connection

    monkeypatch.setattr(sqlite3, "connect", tracked_connect)
    project = tmp_path / "workspace" / "projects" / "demo"
    project.mkdir(parents=True)
    store = DeveloperSessionStore(tmp_path / "workspace" / ".daedalus" / "sessions.sqlite3")
    saved = store.save(make_session(project))
    store.load(saved.id)
    store.list_catalog()
    store.list_sessions()
    store.history(saved.id)
    store.export_json(saved.id)

    assert opened
    for connection in opened:
        with pytest.raises(sqlite3.ProgrammingError):
            connection.execute("SELECT 1")

    moved = store.database.with_suffix(".moved")
    store.database.rename(moved)
    moved.rename(store.database)


def test_concurrent_catalog_and_recovery_inventory_do_not_hold_store_locks(
    tmp_path,
) -> None:
    workspace = tmp_path / "workspace"
    project = workspace / "projects" / "demo"
    for directory in (
        project / "runs",
        project / "checkpoints",
        workspace / "training-runs",
        workspace / "checkpoints",
        workspace / ".daedalus",
    ):
        directory.mkdir(parents=True, exist_ok=True)
    store = DeveloperSessionStore(workspace / ".daedalus" / "sessions.sqlite3")
    saved = store.save(make_session(project))
    planner = RecoveryPlanner(project, workspace, tmp_path / "missing-backup")

    def inspect(index: int):
        if index % 2:
            return store.list_catalog()[0].session_id
        return planner.inventory(session_store=store, session_id=saved.id).session_revision

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(inspect, range(32)))

    assert results[0::2] == [saved.revision] * 16
    assert results[1::2] == [saved.id] * 16
