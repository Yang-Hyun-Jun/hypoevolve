import tempfile
import unittest
from pathlib import Path

from hypoevolve.config import ConfigError, HypoEvolveConfig, load_config


class TestHypoEvolveConfig(unittest.TestCase):
    def test_load_defaults_when_no_path(self):
        config = load_config(None)
        self.assertIsInstance(config, HypoEvolveConfig)
        self.assertEqual(config.archive.top_k, 5)

    def test_load_minimal_yaml(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "hypoevolve.yaml"
            path.write_text("search:\n  iterations: 3\narchive:\n  top_k: 5\n", encoding="utf-8")
            config = load_config(path)
            self.assertEqual(config.search.iterations, 3)
            self.assertEqual(config.archive.top_k, 5)

    def test_invalid_top_k_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "hypoevolve.yaml"
            path.write_text("archive:\n  top_k: 7\n", encoding="utf-8")
            with self.assertRaises(ConfigError):
                load_config(path)

    def test_worker_config_loads(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "hypoevolve.yaml"
            path.write_text("archive:\n  top_k: 5\nworkers:\n  enabled: true\n  count: 2\n", encoding="utf-8")
            config = load_config(path)
            self.assertTrue(config.workers.enabled)
            self.assertEqual(config.workers.count, 2)

    def test_invalid_worker_count_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "hypoevolve.yaml"
            path.write_text("archive:\n  top_k: 5\nworkers:\n  count: 0\n", encoding="utf-8")
            with self.assertRaises(ConfigError):
                load_config(path)


if __name__ == "__main__":
    unittest.main()
