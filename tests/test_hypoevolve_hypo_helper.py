import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from hypoevolve.hypo.helper import get_labels, get_nodes
from hypoevolve.hypo.nodes.nodes import DATA
from hypoevolve.hypo.tree.generator import HypoTreeGenerator


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
            data_labels = [node.label for node in get_nodes(schema_path) if isinstance(node, DATA)]
        self.assertEqual(data_labels, ["ALPHA", "BETA"])

    def test_get_labels_rejects_schema_without_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            schema_path = Path(tmp) / "dataset.yaml"
            schema_path.write_text(
                "description: test\n"
                "files:\n"
                "  -\n"
                "    entity: BTCUSDT\n"
                "    path: data/BTCUSDT.parquet\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                get_labels(schema_path)

    def test_get_nodes_contains_core_operator_nodes_before_data_nodes(self):
        with tempfile.TemporaryDirectory() as tmp:
            schema_path = Path(tmp) / "dataset.yaml"
            schema_path.write_text(
                "description: test\n"
                "files:\n"
                "  -\n"
                "    entity: BTCUSDT\n"
                "    path: data/BTCUSDT.parquet\n"
                "columns:\n"
                "  -\n"
                "    name: ALPHA\n",
                encoding="utf-8",
            )
            generated_nodes = get_nodes(schema_path)
            labels = [getattr(node, "label", None) for node in generated_nodes]
        self.assertIn("ALPHA", labels)
        self.assertGreater(len(generated_nodes), 1)

    def test_get_tree_generator_builds_generator_from_schema_nodes(self):
        with patch("hypoevolve.hypo.helper.get_nodes", return_value=["NODE_A", "NODE_B"]):
            from hypoevolve.hypo.helper import get_tree_generator

            generator = get_tree_generator("dataset.yaml")

        self.assertIsInstance(generator, HypoTreeGenerator)
        self.assertEqual(generator.nodes, ["NODE_A", "NODE_B"])


if __name__ == "__main__":
    unittest.main()
