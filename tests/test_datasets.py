import json
from pathlib import Path

import numpy as np
import pytest

from daedalus.workspace.datasets import DatasetService
from daedalus.workspace.manager import WorkspaceManager


def service(tmp_path: Path) -> DatasetService:
    source = tmp_path / "source"
    source.mkdir()
    manager = WorkspaceManager(source, tmp_path / "private", tmp_path / "backup")
    return DatasetService(manager)


def test_csv_import_and_load_round_trip(tmp_path: Path) -> None:
    datasets = service(tmp_path)
    csv_path = tmp_path / "xor.csv"
    csv_path.write_text("x1,x2,label\n0,0,0\n0,1,1\n1,0,1\n1,1,0\n", encoding="utf-8")
    metadata = datasets.import_csv(csv_path, target_column="label")
    features, targets, loaded = datasets.load(metadata.name)
    np.testing.assert_array_equal(features, [[0, 0], [0, 1], [1, 0], [1, 1]])
    np.testing.assert_array_equal(targets, [0, 1, 1, 0])
    assert loaded.rows == 4


def test_csv_import_rejects_non_numeric_data(tmp_path: Path) -> None:
    datasets = service(tmp_path)
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text("x,y\nhello,1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="non-numeric"):
        datasets.import_csv(csv_path)


def test_csv_import_detects_tampering(tmp_path: Path) -> None:
    datasets = service(tmp_path)
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("x,y\n1,2\n", encoding="utf-8")
    metadata = datasets.import_csv(csv_path)
    imported = datasets.manager.datasets_dir / metadata.file
    imported.write_text("x,y\n9,9\n", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        datasets.load(metadata.name)


def test_csv_headers_with_spaces_round_trip_consistently(tmp_path: Path) -> None:
    datasets = service(tmp_path)
    csv_path = tmp_path / "spaced.csv"
    csv_path.write_text(
        "feature one,feature-two,target value\n1,2,3\n4,5,6\n", encoding="utf-8"
    )
    metadata = datasets.import_csv(csv_path, target_column="target value")
    features, targets, _loaded = datasets.load(metadata.name)
    np.testing.assert_array_equal(features, [[1, 2], [4, 5]])
    np.testing.assert_array_equal(targets, [3, 6])


def test_csv_import_requires_feature_and_target_columns(tmp_path: Path) -> None:
    datasets = service(tmp_path)
    csv_path = tmp_path / "target-only.csv"
    csv_path.write_text("target\n1\n2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="feature column"):
        datasets.import_csv(csv_path)


@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_csv_import_rejects_non_finite_values(tmp_path: Path, value: str) -> None:
    datasets = service(tmp_path)
    csv_path = tmp_path / "nonfinite.csv"
    csv_path.write_text(f"x,y\n{value},1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite"):
        datasets.import_csv(csv_path)


def test_dataset_load_rejects_unsafe_name(tmp_path: Path) -> None:
    datasets = service(tmp_path)
    datasets.manager.bootstrap()
    with pytest.raises(ValueError, match="unsafe path"):
        datasets.load("../outside")


def test_dataset_metadata_cannot_escape_dataset_directory(tmp_path: Path) -> None:
    datasets = service(tmp_path)
    csv_path = tmp_path / "safe.csv"
    csv_path.write_text("x,y\n1,2\n", encoding="utf-8")
    metadata = datasets.import_csv(csv_path)
    metadata_path = datasets.manager.datasets_dir / f"{metadata.name}.dataset.json"
    raw = json.loads(metadata_path.read_text(encoding="utf-8"))
    raw["file"] = "../outside.csv"
    metadata_path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="unsafe data-file path"):
        datasets.load(metadata.name)


def test_failed_metadata_publication_removes_unpublished_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    datasets = service(tmp_path)
    csv_path = tmp_path / "atomic.csv"
    csv_path.write_text("x,y\n1,2\n", encoding="utf-8")
    original_replace = Path.replace

    def fail_metadata_replace(path: Path, target: Path):
        if Path(target).name == "atomic.dataset.json":
            raise OSError("simulated metadata publication failure")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_metadata_replace)
    with pytest.raises(OSError, match="simulated"):
        datasets.import_csv(csv_path)
    assert not (datasets.manager.datasets_dir / "atomic.dataset.json").exists()
    assert not list(datasets.manager.datasets_dir.glob("atomic.*.csv"))
    assert not list(datasets.manager.datasets_dir.glob(".atomic.*.importing"))


def test_interrupted_import_is_reported_with_recovery_action(tmp_path: Path) -> None:
    datasets = service(tmp_path)
    datasets.manager.bootstrap()
    orphan = datasets.manager.datasets_dir / f"data.{'a' * 32}.csv"
    orphan.write_text("x,y\n1,2\n", encoding="utf-8")
    source = tmp_path / "data.csv"
    source.write_text("x,y\n3,4\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Remove those unpublished artifacts"):
        datasets.import_csv(source)
