"""Non-executable model checkpoints backed by NPZ arrays and JSON metadata."""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Protocol

import numpy as np


class ParameterLike(Protocol):
    data: np.ndarray


_SAFE_STEM = re.compile(r"[^A-Za-z0-9._-]+")
_MAX_CONTRACT_DEPTH = 12
_MAX_CONTRACT_NODES = 20_000
_MAX_CONTRACT_BYTES = 2 * 1024 * 1024


def _stem(name: str) -> str:
    value = _SAFE_STEM.sub("-", name.strip()).strip(".-")
    if not value:
        raise ValueError("Checkpoint name cannot be empty.")
    return value[:100]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validated_contract(value: Any) -> dict[str, Any]:
    """Return a bounded JSON-safe copy of a checkpoint training contract."""

    nodes = 0

    def visit(item: Any, depth: int) -> Any:
        nonlocal nodes
        nodes += 1
        if nodes > _MAX_CONTRACT_NODES:
            raise ValueError("Training contract contains too many values.")
        if depth > _MAX_CONTRACT_DEPTH:
            raise ValueError("Training contract is nested too deeply.")
        if item is None or isinstance(item, (bool, str, int)):
            return item
        if isinstance(item, (float, np.floating)):
            number = float(item)
            if not np.isfinite(number):
                raise ValueError("Training contract numbers must be finite.")
            return number
        if isinstance(item, np.integer):
            return int(item)
        if isinstance(item, dict):
            result: dict[str, Any] = {}
            for key, child in item.items():
                if not isinstance(key, str) or not key or len(key) > 200:
                    raise TypeError("Training contract keys must be non-empty bounded strings.")
                result[key] = visit(child, depth + 1)
            return result
        if isinstance(item, (list, tuple)):
            return [visit(child, depth + 1) for child in item]
        raise TypeError(
            "Training contract values must be JSON scalars, objects, arrays, or null."
        )

    cleaned = visit(value, 0)
    if not isinstance(cleaned, dict):
        raise TypeError("Training contract must be a JSON object.")
    encoded = json.dumps(cleaned, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if len(encoded.encode("utf-8")) > _MAX_CONTRACT_BYTES:
        raise ValueError("Training contract exceeds the checkpoint metadata size limit.")
    return cleaned


def save_checkpoint(
    directory: Path,
    name: str,
    parameters: Iterable[ParameterLike],
    *,
    architecture: list[dict[str, Any]] | None = None,
    metrics: dict[str, float] | None = None,
    training_contract: dict[str, Any] | None = None,
) -> tuple[Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    stem = _stem(name)
    generation = uuid.uuid4().hex
    arrays_path = directory / f"{stem}.{generation}.npz"
    metadata_path = directory / f"{stem}.json"
    temporary_arrays = directory / f".{stem}.{generation}.npz.tmp"
    temporary_metadata = directory / f".{stem}.{generation}.json.tmp"

    contract = None if training_contract is None else _validated_contract(training_contract)
    arrays = {}
    for index, parameter in enumerate(parameters):
        value = np.asarray(parameter.data)
        if value.dtype.hasobject:
            raise TypeError("Object arrays are forbidden in checkpoints.")
        arrays[f"parameter_{index:04d}"] = value
    arrays_published = False
    try:
        with temporary_arrays.open("xb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        digest = _sha256(temporary_arrays)
        metadata = {
            "format": "daedalus-npz",
            "schema": 2,
            "created_utc": datetime.now(UTC).isoformat(),
            "array_file": arrays_path.name,
            "sha256": digest,
            "parameter_count": len(arrays),
            "shapes": [list(value.shape) for value in arrays.values()],
            "dtypes": [str(value.dtype) for value in arrays.values()],
            "architecture": architecture or [],
            "metrics": metrics or {},
            "training_contract": contract,
        }
        with temporary_metadata.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(metadata, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary_arrays.replace(arrays_path)
        arrays_published = True
        # The stable metadata file is the commit pointer. Existing metadata
        # remains valid until this final atomic replacement succeeds.
        temporary_metadata.replace(metadata_path)
        return arrays_path, metadata_path
    except Exception:
        temporary_arrays.unlink(missing_ok=True)
        temporary_metadata.unlink(missing_ok=True)
        if arrays_published:
            arrays_path.unlink(missing_ok=True)
        raise


def load_checkpoint(
    metadata_path: Path, parameters: Iterable[ParameterLike]
) -> dict[str, Any]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("format") != "daedalus-npz" or metadata.get("schema") not in {1, 2}:
        raise ValueError("Unsupported or invalid Daedalus checkpoint metadata.")
    array_file = str(metadata["array_file"])
    relative_array = Path(array_file)
    if relative_array.is_absolute() or relative_array.name != array_file:
        raise ValueError("Checkpoint metadata contains an unsafe array-file path.")
    arrays_path = metadata_path.parent / relative_array
    if arrays_path.is_symlink():
        raise ValueError("Checkpoint array files cannot be symbolic links.")
    if not arrays_path.is_file():
        raise FileNotFoundError(arrays_path)
    if _sha256(arrays_path) != metadata.get("sha256"):
        raise ValueError("Checkpoint checksum does not match its metadata.")

    parameter_list = list(parameters)
    if len(parameter_list) != int(metadata["parameter_count"]):
        raise ValueError("Checkpoint parameter count does not match the model.")
    loaded_values: list[np.ndarray] = []
    with np.load(arrays_path, allow_pickle=False) as archive:
        expected_keys = [f"parameter_{index:04d}" for index in range(len(parameter_list))]
        if archive.files != expected_keys:
            raise ValueError("Checkpoint array index is incomplete or reordered.")
        for index, parameter in enumerate(parameter_list):
            value = np.array(archive[expected_keys[index]], copy=True)
            target = np.asarray(parameter.data)
            if value.shape != target.shape:
                raise ValueError(
                    f"Parameter {index} shape mismatch: checkpoint={value.shape}, "
                    f"model={target.shape}"
                )
            if value.dtype.hasobject:
                raise TypeError("Object arrays are forbidden in checkpoints.")
            if not target.flags.writeable:
                raise ValueError(f"Parameter {index} is not writable.")
            if not np.can_cast(value.dtype, target.dtype, casting="same_kind"):
                raise ValueError(
                    f"Parameter {index} dtype mismatch: checkpoint={value.dtype}, model={target.dtype}"
                )
            loaded_values.append(value.astype(target.dtype, copy=False))

    originals = [np.array(parameter.data, copy=True) for parameter in parameter_list]
    try:
        for parameter, value in zip(parameter_list, loaded_values, strict=True):
            parameter.data[...] = value
    except Exception:
        for parameter, original in zip(parameter_list, originals, strict=True):
            parameter.data[...] = original
        raise
    return metadata


def list_checkpoints(directory: Path) -> list[dict[str, Any]]:
    if not directory.exists():
        return []
    results = []
    for metadata_path in sorted(directory.glob("*.json"), reverse=True):
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("format") == "daedalus-npz":
                metadata["metadata_path"] = str(metadata_path)
                results.append(metadata)
        except (OSError, ValueError, TypeError):
            continue
    return results
