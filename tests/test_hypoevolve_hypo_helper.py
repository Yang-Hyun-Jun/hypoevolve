import tempfile
import unittest
from pathlib import Path

from hypoevolve.hypo.helper import get_labels, get_nodes
from hypoevolve.hypo.nodes.nodes import DATA


class TestHypoHelper(unittest.TestCase):
    def test_get_labels_reads_dataset_schema_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            schema_path = Path(tmp) / "dataset.yaml"
            schema_path.write_text(
                "description: test\n"
                "index:\n"
                "  name: close_time\n"
                "  dtype: datetime64[us]\n"
                "files:\n"
                "  -\n"
                "    entity: BTCUSDT\n"
                "    path: data/BTCUSDT.parquet\n"
                "columns:\n"
                "  -\n"
                "    name: ALPHA\n"
                "  -\n"
                "    name: BETA\n",
                encoding="utf-8",
            )

            self.assertEqual(get_labels(schema_path), ["ALPHA", "BETA"])

    def test_get_nodes_uses_dataset_schema_columns_for_data_nodes(self):
        with tempfile.TemporaryDirectory() as tmp:
            schema_path = Path(tmp) / "dataset.yaml"
            schema_path.write_text(
                "description: test\n"
                "index:\n"
                "  name: close_time\n"
                "  dtype: datetime64[us]\n"
                "files:\n"
                "  -\n"
                "    entity: BTCUSDT\n"
                "    path: data/BTCUSDT.parquet\n"
                "columns:\n"
                "  -\n"
                "    name: ALPHA\n"
                "  -\n"
                "    name: BETA\n",
                encoding="utf-8",
            )

            data_labels = [
                node.label
                for node in get_nodes(schema_path)
                if isinstance(node, DATA)
            ]

        self.assertEqual(data_labels, ["ALPHA", "BETA"])


if __name__ == "__main__":
    unittest.main()
