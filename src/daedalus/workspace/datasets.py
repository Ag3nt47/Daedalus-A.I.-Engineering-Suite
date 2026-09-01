"""Validated numeric CSV ingestion for private Daedalus datasets."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import shutil
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from daedalus.workspace.manager import WorkspaceManager

_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")
_GENERATION_FILE = re.compile(r"^(?P<name>.+)\.(?P<generation>[0-9a-f]{32})\.csv$")


def _dataset_name(value: str, *, strict: bool = False) -> str:
    raw = value.strip()
    cleaned = _SAFE_NAME.sub("-", raw).strip(".-")[:100]
    if not cleaned:
        raise ValueError("Dataset name must contain at least one letter or number.")
    if strict and cleaned != raw:
        raise ValueError("Dataset name contains unsafe path characters or is too long.")
    return cleaned


@dataclass(frozen=True, slots=True)
class DatasetMetadata:
    schema: int
    name: str
    file: str
    sha256: str
    imported_utc: str
    rows: int
    columns: tuple[str, ...]
    feature_columns: tuple[str, ...]
    target_column: str
    delimiter: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class DatasetService:
    def __init__(self, manager: WorkspaceManager, *, maximum_bytes: int = 256 * 1024 * 1024):
        self.manager = manager
        self.maximum_bytes = maximum_bytes

    def import_csv(
        self,
        source: str | os.PathLike[str],
        *,
        name: str | None = None,
        target_column: str | None = None,
        delimiter: str = ",",
    ) -> DatasetMetadata:
        self.manager.bootstrap()
        source_path = Path(source).resolve(strict=True)
        if not source_path.is_file() or source_path.is_symlink():
            raise ValueError("Dataset source must be a regular, non-symlink file.")
        if source_path.suffix.casefold() != ".csv":
            raise ValueError("This importer accepts CSV files only.")
        if source_path.stat().st_size > self.maximum_bytes:
            raise ValueError("Dataset exceeds the configured import size limit.")
        if len(delimiter) != 1:
            raise ValueError("CSV delimiter must be one character.")

        columns, row_count = self._validate_numeric_csv(source_path, delimiter)
        resolved_target = target_column.strip() if target_column else columns[-1]
        if resolved_target not in columns:
            raise ValueError(f"Target column {resolved_target!r} is not present in the CSV header.")
        dataset_name = _dataset_name(name or source_path.stem)
        metadata_path = self.manager.datasets_dir / f"{dataset_name}.dataset.json"
        if metadata_path.exists():
            raise FileExistsError(f"Dataset {dataset_name!r} already exists.")
        interrupted = self._interrupted_import_artifacts(dataset_name)
        if interrupted:
            names = ", ".join(path.name for path in interrupted[:5])
            raise RuntimeError(
                f"An interrupted import for dataset {dataset_name!r} needs review: {names}. "
                "Remove those unpublished artifacts or choose a different dataset name."
            )

        generation = uuid.uuid4().hex
        destination = self.manager.datasets_dir / f"{dataset_name}.{generation}.csv"
        temporary = self.manager.datasets_dir / f".{dataset_name}.{generation}.csv.importing"
        temporary_metadata = (
            self.manager.datasets_dir / f".{dataset_name}.{generation}.dataset.json.importing"
        )
        data_published = False
        try:
            shutil.copy2(source_path, temporary)
            if _sha256(source_path) != _sha256(temporary):
                raise OSError("Dataset checksum changed during import.")
            metadata = DatasetMetadata(
                schema=1,
                name=dataset_name,
                file=destination.name,
                sha256=_sha256(temporary),
                imported_utc=datetime.now(UTC).isoformat(),
                rows=row_count,
                columns=tuple(columns),
                feature_columns=tuple(column for column in columns if column != resolved_target),
                target_column=resolved_target,
                delimiter=delimiter,
            )
            with temporary_metadata.open("x", encoding="utf-8", newline="\n") as handle:
                json.dump(asdict(metadata), handle, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(destination)
            data_published = True
            temporary_metadata.replace(metadata_path)
            return metadata
        except Exception:
            temporary.unlink(missing_ok=True)
            temporary_metadata.unlink(missing_ok=True)
            if data_published and not metadata_path.exists():
                destination.unlink(missing_ok=True)
            raise

    def _interrupted_import_artifacts(self, dataset_name: str) -> list[Path]:
        results: list[Path] = []
        legacy = self.manager.datasets_dir / f"{dataset_name}.csv"
        if legacy.exists():
            results.append(legacy)
        for path in self.manager.datasets_dir.iterdir():
            name = path.name
            generated = _GENERATION_FILE.fullmatch(name)
            if generated and generated.group("name") == dataset_name:
                results.append(path)
            elif name.startswith(f".{dataset_name}.") and name.endswith(
                (".csv.importing", ".dataset.json.importing")
            ):
                results.append(path)
        return sorted(results, key=lambda path: path.name.casefold())

    @staticmethod
    def _validate_numeric_csv(path: Path, delimiter: str) -> tuple[list[str], int]:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle, delimiter=delimiter)
            try:
                columns = [value.strip() for value in next(reader)]
            except StopIteration as exc:
                raise ValueError("CSV file is empty.") from exc
            if not columns or any(not column for column in columns):
                raise ValueError("Every CSV column needs a non-empty header.")
            if len(set(columns)) != len(columns):
                raise ValueError("CSV headers must be unique.")
            if len(columns) < 2:
                raise ValueError("CSV needs at least one feature column and one target column.")
            rows = 0
            for line_number, row in enumerate(reader, start=2):
                if not row or all(not value.strip() for value in row):
                    continue
                if len(row) != len(columns):
                    raise ValueError(
                        f"CSV line {line_number} has {len(row)} values; expected {len(columns)}."
                    )
                try:
                    numbers = [float(value) for value in row]
                except ValueError as exc:
                    raise ValueError(f"CSV line {line_number} contains a non-numeric value.") from exc
                if not all(math.isfinite(value) for value in numbers):
                    raise ValueError(f"CSV line {line_number} contains a non-finite value.")
                rows += 1
            if rows == 0:
                raise ValueError("CSV contains no numeric data rows.")
            return columns, rows

    def load(self, name: str) -> tuple[Any, Any, DatasetMetadata]:
        self.manager.bootstrap()
        safe_name = _dataset_name(name, strict=True)
        metadata_path = self.manager.datasets_dir / f"{safe_name}.dataset.json"
        metadata, dataset_path = self._read_metadata(metadata_path, expected_name=safe_name)
        if _sha256(dataset_path) != metadata.sha256:
            raise ValueError("Dataset checksum does not match its import metadata.")
        columns, matrix = self._load_numeric_csv(dataset_path, metadata.delimiter)
        if tuple(columns) != metadata.columns:
            raise ValueError("Dataset header does not match its import metadata.")
        if len(matrix) != metadata.rows:
            raise ValueError("Dataset row count does not match its import metadata.")
        target_index = columns.index(metadata.target_column)
        feature_indices = [columns.index(column) for column in metadata.feature_columns]
        features = matrix[:, feature_indices]
        targets = matrix[:, target_index]
        return features, targets, metadata

    def list_datasets(self) -> list[DatasetMetadata]:
        self.manager.bootstrap()
        results = []
        for path in sorted(self.manager.datasets_dir.glob("*.dataset.json")):
            try:
                expected_name = path.name.removesuffix(".dataset.json")
                metadata, _dataset_path = self._read_metadata(
                    path, expected_name=_dataset_name(expected_name, strict=True)
                )
                results.append(metadata)
            except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
                continue
        return results

    def _read_metadata(
        self, metadata_path: Path, *, expected_name: str
    ) -> tuple[DatasetMetadata, Path]:
        raw = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("Dataset metadata must be a JSON object.")
        try:
            metadata = DatasetMetadata(
                **{
                    **raw,
                    "columns": tuple(raw["columns"]),
                    "feature_columns": tuple(raw["feature_columns"]),
                }
            )
        except (TypeError, KeyError) as exc:
            raise ValueError("Dataset metadata is incomplete or malformed.") from exc
        if metadata.schema != 1 or metadata.name != expected_name:
            raise ValueError("Dataset metadata identity or schema is invalid.")
        if len(metadata.delimiter) != 1:
            raise ValueError("Dataset metadata contains an invalid delimiter.")
        if len(metadata.columns) < 2 or not metadata.feature_columns:
            raise ValueError("Dataset metadata needs feature and target columns.")
        if metadata.target_column not in metadata.columns:
            raise ValueError("Dataset target column is absent from metadata columns.")
        if tuple(column for column in metadata.columns if column != metadata.target_column) != (
            metadata.feature_columns
        ):
            raise ValueError("Dataset feature columns are inconsistent with its target column.")
        file_name = Path(metadata.file)
        generated = _GENERATION_FILE.fullmatch(metadata.file)
        legacy_name = f"{expected_name}.csv"
        if (
            file_name.is_absolute()
            or file_name.name != metadata.file
            or not (
                metadata.file == legacy_name
                or (generated is not None and generated.group("name") == expected_name)
            )
        ):
            raise ValueError("Dataset metadata contains an unsafe data-file path.")
        unresolved = self.manager.datasets_dir / metadata.file
        if unresolved.is_symlink():
            raise ValueError("Dataset data files cannot be symbolic links.")
        dataset_path = unresolved.resolve(strict=True)
        if dataset_path.parent != self.manager.datasets_dir.resolve(strict=True):
            raise ValueError("Dataset data file escapes the private dataset directory.")
        if not dataset_path.is_file():
            raise FileNotFoundError(dataset_path)
        return metadata, dataset_path

    @staticmethod
    def _load_numeric_csv(path: Path, delimiter: str) -> tuple[list[str], Any]:
        # Metadata-only paths (for example opening Training Lab) should not pay
        # NumPy's import cost. Numeric allocation begins only when a user loads
        # a dataset for analysis or training.
        import numpy as np

        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle, delimiter=delimiter)
            try:
                columns = [value.strip() for value in next(reader)]
            except StopIteration as exc:
                raise ValueError("CSV file is empty.") from exc
            rows: list[list[float]] = []
            for line_number, row in enumerate(reader, start=2):
                if not row or all(not value.strip() for value in row):
                    continue
                if len(row) != len(columns):
                    raise ValueError(
                        f"CSV line {line_number} has {len(row)} values; expected {len(columns)}."
                    )
                try:
                    values = [float(value) for value in row]
                except ValueError as exc:
                    raise ValueError(
                        f"CSV line {line_number} contains a non-numeric value."
                    ) from exc
                if not all(math.isfinite(value) for value in values):
                    raise ValueError(f"CSV line {line_number} contains a non-finite value.")
                rows.append(values)
        if not rows:
            raise ValueError("CSV contains no numeric data rows.")
        return columns, np.asarray(rows, dtype=np.float64)
