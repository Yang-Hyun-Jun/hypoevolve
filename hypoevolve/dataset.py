from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

from .config import _ensure_mapping, _parse_simple_yaml


class DatasetSchemaError(ValueError):
    pass


@dataclass(slots=True)
class ColumnSpec:
    name: str
    description: str | None = None


@dataclass(slots=True)
class DataFile:
    entity: str
    path: str


@dataclass(slots=True)
class IndexSpec:
    name: str | None = None
    dtype: str | None = None


@dataclass(slots=True)
class DatasetSchema:
    files: List[DataFile] = field(default_factory=list)
    index: IndexSpec = field(default_factory=IndexSpec)
    columns: List[ColumnSpec] = field(default_factory=list)
    description: str | None = None


class DatasetAccessor:
    def __init__(self, schema: DatasetSchema):
        self.schema = schema

    def entities(self) -> List[str]:
        return [item.entity for item in self.schema.files]

    def file_map(self) -> Dict[str, Path]:
        return {item.entity: Path(item.path) for item in self.schema.files}

    def column_names(self) -> List[str]:
        return [column.name for column in self.schema.columns]

    def column_descriptions(self) -> Dict[str, str]:
        return {column.name: column.description or "" for column in self.schema.columns}

    def load_dataframe(self, entity: str):
        file_path = self.file_map().get(entity)
        if file_path is None:
            raise DatasetSchemaError(f"Unknown entity: {entity}")
        return self._read_parquet_dataframe(file_path)

    def load_all_dataframes(self) -> Dict[str, object]:
        return {
            entity: self._read_parquet_dataframe(path)
            for entity, path in self.file_map().items()
        }

    def head(self, entity: str, n: int = 5):
        return self.load_dataframe(entity).head(n)

    def summary(self) -> Dict[str, object]:
        return {
            "description": self.schema.description,
            "index_name": self.schema.index.name,
            "index_dtype": self.schema.index.dtype,
            "entities": self.entities(),
            "columns": self.column_names(),
            "column_descriptions": self.column_descriptions(),
            "files": {entity: str(path) for entity, path in self.file_map().items()},
        }

    def _read_parquet_dataframe(self, path: Path):
        if path.suffix.lower() != ".parquet":
            raise DatasetSchemaError("DatasetAccessor only supports .parquet files")
        try:
            pandas = importlib.import_module("pandas")
        except Exception as exc:
            raise DatasetSchemaError(
                "Reading parquet requires pandas to be installed"
            ) from exc
        return pandas.read_parquet(path)


def load_dataset_schema(path: str | Path) -> DatasetSchema:
    schema_path = Path(path)
    if not schema_path.exists():
        raise DatasetSchemaError(f"Dataset schema file not found: {schema_path}")
    raw = _parse_simple_yaml(schema_path.read_text(encoding="utf-8"))
    schema = dataset_schema_from_dict(raw)
    base_dir = schema_path.resolve().parent
    resolved_files = []
    for item in schema.files:
        file_path = Path(item.path)
        if not file_path.is_absolute():
            file_path = base_dir / file_path
        resolved_files.append(DataFile(entity=item.entity, path=str(file_path)))
    schema.files = resolved_files
    return schema


def dataset_schema_from_dict(data: Dict[str, object]) -> DatasetSchema:
    _ensure_mapping(data, "dataset schema")

    files_raw = data.get("files", [])
    columns_raw = data.get("columns", [])
    index_raw = data.get("index", {})

    if not isinstance(files_raw, list):
        raise DatasetSchemaError("files must be a list")
    if not isinstance(columns_raw, list):
        raise DatasetSchemaError("columns must be a list")
    if not isinstance(index_raw, dict):
        raise DatasetSchemaError("index must be a mapping")

    files: List[DataFile] = []
    for item in files_raw:
        _ensure_mapping(item, "dataset file entry")
        if "entity" not in item or "path" not in item:
            raise DatasetSchemaError("Each file entry must include 'entity' and 'path'")
        files.append(DataFile(entity=str(item["entity"]), path=str(item["path"])))

    columns: List[ColumnSpec] = []
    for item in columns_raw:
        _ensure_mapping(item, "column spec")
        if "name" not in item:
            raise DatasetSchemaError("Each column spec must include 'name'")
        columns.append(
            ColumnSpec(
                name=str(item["name"]),
                description=str(item["description"])
                if item.get("description") is not None
                else None,
            )
        )

    index = IndexSpec(
        name=str(index_raw.get("name")) if index_raw.get("name") is not None else None,
        dtype=str(index_raw.get("dtype"))
        if index_raw.get("dtype") is not None
        else None,
    )

    return DatasetSchema(
        files=files,
        index=index,
        columns=columns,
        description=str(data["description"])
        if data.get("description") is not None
        else None,
    )
