import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from hypoevolve.dataset import (
    ColumnSpec,
    DataFile,
    DatasetAccessor,
    DatasetSchema,
    DatasetSchemaError,
    IndexSpec,
    dataset_schema_from_dict,
    load_dataset_schema,
)


class TestHypoEvolveDataset(unittest.TestCase):
    def test_dataset_schema_from_dict_builds_schema(self):
        schema = dataset_schema_from_dict(
            {
                "description": "OHLCV dataset",
                "index": {"name": "close_time", "dtype": "datetime64[us]"},
                "files": [
                    {"entity": "BTCUSDT", "path": "BTCUSDT.parquet"},
                    {"entity": "ETHUSDT", "path": "ETHUSDT.parquet"},
                ],
                "columns": [
                    {"name": "CLOSE", "description": "close price"},
                    {"name": "VOLUME", "description": "traded volume"},
                ],
            }
        )
        self.assertEqual(schema.index.name, "close_time")
        self.assertEqual(schema.index.dtype, "datetime64[us]")
        self.assertEqual(len(schema.files), 2)
        self.assertEqual(schema.columns[0].name, "CLOSE")

    def test_load_dataset_schema_from_yaml(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dataset.yaml"
            path.write_text(
                "description: OHLCV dataset\n"
                "index:\n"
                "  name: close_time\n"
                "  dtype: datetime64[us]\n"
                "files:\n"
                "  -\n"
                "    entity: BTCUSDT\n"
                f"    path: {tmp}/BTCUSDT.parquet\n"
                "columns:\n"
                "  -\n"
                "    name: CLOSE\n"
                "    description: close price\n",
                encoding="utf-8",
            )
            schema = load_dataset_schema(path)
            self.assertEqual(schema.description, "OHLCV dataset")
            self.assertEqual(schema.index.name, "close_time")
            self.assertEqual(schema.files[0].entity, "BTCUSDT")

    def test_load_dataset_schema_resolves_relative_file_paths_from_schema_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dataset.yaml"
            path.write_text(
                "description: OHLCV dataset\n"
                "files:\n"
                "  -\n"
                "    entity: BTCUSDT\n"
                "    path: data/BTCUSDT.parquet\n",
                encoding="utf-8",
            )
            schema = load_dataset_schema(path)
            self.assertEqual(
                schema.files[0].path,
                str((Path(tmp) / "data" / "BTCUSDT.parquet").resolve()),
            )

    def test_dataset_accessor_reads_dataframe_and_summary(self):
        class FakeFrame:
            def __init__(self):
                self.rows = [{"close_time": "2022-01-01", "CLOSE": 100.0}, {"close_time": "2022-01-02", "CLOSE": 101.0}]

            def head(self, n=5):
                return self.rows[:n]

        class FakePandas:
            @staticmethod
            def read_parquet(path):
                return FakeFrame()

        schema = DatasetSchema(
            files=[
                DataFile(entity="BTCUSDT", path="BTCUSDT.parquet"),
                DataFile(entity="ETHUSDT", path="ETHUSDT.parquet"),
            ],
            index=IndexSpec(name="close_time", dtype="datetime64[us]"),
            columns=[ColumnSpec(name="CLOSE", description="close price")],
            description="Per-asset OHLCV",
        )
        accessor = DatasetAccessor(schema)
        with patch("importlib.import_module", side_effect=lambda name: FakePandas if name == "pandas" else (_ for _ in ()).throw(ImportError())):
            frame = accessor.load_dataframe("BTCUSDT")
            self.assertEqual(accessor.entities(), ["BTCUSDT", "ETHUSDT"])
            self.assertEqual(accessor.column_names(), ["CLOSE"])
            self.assertEqual(accessor.column_descriptions()["CLOSE"], "close price")
            self.assertIsInstance(frame, FakeFrame)
            self.assertEqual(accessor.head("BTCUSDT", 1)[0]["CLOSE"], 100.0)
            summary = accessor.summary()
            self.assertEqual(summary["index_name"], "close_time")
            self.assertEqual(summary["index_dtype"], "datetime64[us]")
            self.assertIn("files", summary)

    def test_dataset_accessor_rejects_unknown_entity(self):
        schema = DatasetSchema(files=[], columns=[])
        accessor = DatasetAccessor(schema)
        with self.assertRaises(DatasetSchemaError):
            accessor.load_dataframe("BTCUSDT")

    def test_dataset_accessor_rejects_non_parquet_file(self):
        schema = DatasetSchema(
            files=[DataFile(entity="BTCUSDT", path="BTCUSDT.csv")],
            columns=[ColumnSpec(name="CLOSE")],
        )
        accessor = DatasetAccessor(schema)
        with self.assertRaises(DatasetSchemaError):
            accessor.load_dataframe("BTCUSDT")

    def test_dataset_accessor_rejects_parquet_without_pandas(self):
        schema = DatasetSchema(
            files=[DataFile(entity="SOLUSDT", path="SOLUSDT.parquet")],
            columns=[ColumnSpec(name="CLOSE")],
        )
        accessor = DatasetAccessor(schema)
        with patch("importlib.import_module", side_effect=ImportError()):
            with self.assertRaises(DatasetSchemaError):
                accessor.load_dataframe("SOLUSDT")

    def test_invalid_dataset_schema_fails(self):
        with self.assertRaises(DatasetSchemaError):
            dataset_schema_from_dict({"files": {"bad": "shape"}})
