import json
from pathlib import Path

import numpy as np
import pytest

from daedalus.workspace.checkpoints import load_checkpoint, save_checkpoint


class Parameter:
    def __init__(self, data):
        self.data = np.asarray(data, dtype=np.float64)


def test_checkpoint_round_trip_is_non_executable(tmp_path: Path) -> None:
    parameters = [Parameter([[1, 2], [3, 4]]), Parameter([5, 6])]
    _, metadata_path = save_checkpoint(tmp_path, "test", parameters, metrics={"loss": 0.1})
    for parameter in parameters:
        parameter.data.fill(0)
    metadata = load_checkpoint(metadata_path, parameters)
    assert metadata["format"] == "daedalus-npz"
    np.testing.assert_array_equal(parameters[0].data, [[1, 2], [3, 4]])


def test_checkpoint_detects_tampering(tmp_path: Path) -> None:
    parameters = [Parameter([1, 2, 3])]
    arrays_path, metadata_path = save_checkpoint(tmp_path, "test", parameters)
    arrays_path.write_bytes(arrays_path.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        load_checkpoint(metadata_path, parameters)


def test_checkpoint_validates_every_parameter_before_mutating(tmp_path: Path) -> None:
    saved = [Parameter([1, 2]), Parameter([3, 4])]
    _, metadata_path = save_checkpoint(tmp_path, "atomic-load", saved)
    targets = [Parameter([0, 0]), Parameter([0, 0, 0])]
    with pytest.raises(ValueError, match="shape mismatch"):
        load_checkpoint(metadata_path, targets)
    np.testing.assert_array_equal(targets[0].data, [0, 0])
    np.testing.assert_array_equal(targets[1].data, [0, 0, 0])


def test_checkpoint_updates_metadata_pointer_to_immutable_generation(tmp_path: Path) -> None:
    first_arrays, metadata_path = save_checkpoint(tmp_path, "versioned", [Parameter([1, 2])])
    second_arrays, second_metadata_path = save_checkpoint(
        tmp_path, "versioned", [Parameter([9, 8])]
    )
    assert metadata_path == second_metadata_path
    assert first_arrays != second_arrays
    assert first_arrays.is_file()
    assert second_arrays.is_file()
    target = [Parameter([0, 0])]
    load_checkpoint(metadata_path, target)
    np.testing.assert_array_equal(target[0].data, [9, 8])


def test_failed_checkpoint_metadata_publish_preserves_previous_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first_arrays, metadata_path = save_checkpoint(tmp_path, "stable", [Parameter([1, 2])])
    original_replace = Path.replace

    def fail_metadata_replace(path: Path, target: Path):
        if Path(target) == metadata_path:
            raise OSError("simulated metadata pointer failure")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_metadata_replace)
    with pytest.raises(OSError, match="simulated"):
        save_checkpoint(tmp_path, "stable", [Parameter([7, 8])])
    assert first_arrays.is_file()
    assert len(list(tmp_path.glob("stable.*.npz"))) == 1
    target = [Parameter([0, 0])]
    load_checkpoint(metadata_path, target)
    np.testing.assert_array_equal(target[0].data, [1, 2])


def test_checkpoint_rejects_array_path_escape(tmp_path: Path) -> None:
    _, metadata_path = save_checkpoint(tmp_path, "safe", [Parameter([1])])
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["array_file"] = "../outside.npz"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="unsafe array-file path"):
        load_checkpoint(metadata_path, [Parameter([0])])


def test_checkpoint_loads_legacy_schema_one_pointer(tmp_path: Path) -> None:
    arrays_path, metadata_path = save_checkpoint(tmp_path, "legacy", [Parameter([4, 5])])
    legacy_arrays = tmp_path / "legacy.npz"
    arrays_path.rename(legacy_arrays)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["schema"] = 1
    metadata["array_file"] = legacy_arrays.name
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    target = [Parameter([0, 0])]
    load_checkpoint(metadata_path, target)
    np.testing.assert_array_equal(target[0].data, [4, 5])


def test_checkpoint_preserves_bounded_training_contract(tmp_path: Path) -> None:
    contract = {
        "task": "classification",
        "split": {"seed": np.int64(47), "validation_fraction": np.float64(0.2)},
        "labels": [0, 1],
        "preprocessing": {"mean": [1.0, 2.0], "scale": [0.5, 0.25]},
    }
    _, metadata_path = save_checkpoint(
        tmp_path,
        "contract",
        [Parameter([1, 2])],
        training_contract=contract,
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["training_contract"] == {
        "task": "classification",
        "split": {"seed": 47, "validation_fraction": 0.2},
        "labels": [0, 1],
        "preprocessing": {"mean": [1.0, 2.0], "scale": [0.5, 0.25]},
    }


@pytest.mark.parametrize(
    "contract, message",
    [
        ({"loss": float("nan")}, "finite"),
        ({"object": object()}, "JSON"),
        ({1: "not a string key"}, "keys"),
    ],
)
def test_checkpoint_rejects_unsafe_training_contract(
    tmp_path: Path, contract: dict, message: str
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        save_checkpoint(
            tmp_path,
            "unsafe-contract",
            [Parameter([1])],
            training_contract=contract,
        )
    assert not list(tmp_path.iterdir())
