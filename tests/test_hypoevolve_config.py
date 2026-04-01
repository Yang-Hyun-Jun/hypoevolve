import tempfile
import unittest
from pathlib import Path

from hypoevolve.config import ConfigError, HypoEvolveConfig, load_config


class TestHypoEvolveConfig(unittest.TestCase):
    def test_load_defaults_when_no_path(self):
        config = load_config(None)
        self.assertIsInstance(config, HypoEvolveConfig)
        self.assertEqual(config.archive.coverage_bins, [0.05, 0.15, 0.30])
        self.assertEqual(config.archive.complexity_bins, [3, 5, 8])

    def test_load_minimal_yaml(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "hypoevolve.yaml"
            path.write_text("search:\n  iterations: 3\n", encoding="utf-8")
            config = load_config(path)
            self.assertEqual(config.search.iterations, 3)
            self.assertEqual(config.archive.coverage_bins, [0.05, 0.15, 0.30])

    def test_archive_bins_load_from_yaml(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "hypoevolve.yaml"
            path.write_text(
                "archive:\n"
                "  coverage_bins:\n"
                "    - 0.02\n"
                "    - 0.10\n"
                "    - 0.25\n"
                "  complexity_bins:\n"
                "    - 2\n"
                "    - 4\n"
                "    - 7\n",
                encoding="utf-8",
            )
            config = load_config(path)
            self.assertEqual(config.archive.coverage_bins, [0.02, 0.10, 0.25])
            self.assertEqual(config.archive.complexity_bins, [2, 4, 7])

    def test_worker_config_loads(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "hypoevolve.yaml"
            path.write_text("workers:\n  enabled: true\n  count: 2\n", encoding="utf-8")
            config = load_config(path)
            self.assertTrue(config.workers.enabled)
            self.assertEqual(config.workers.count, 2)

    def test_llm_evaluator_config_loads(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "hypoevolve.yaml"
            path.write_text(
                "evaluator:\n"
                "  dataset_schema_path: dataset.yaml\n"
                "search:\n"
                "  steering_retries: 2\n",
                encoding="utf-8",
            )
            config = load_config(path)
            self.assertEqual(config.evaluator.dataset_schema_path, "dataset.yaml")
            self.assertEqual(config.search.steering_retries, 2)
            self.assertEqual(config.search.random_steering_prob, 0.2)

    def test_random_steering_prob_loads_and_validates(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "hypoevolve.yaml"
            path.write_text(
                "search:\n"
                "  random_steering_prob: 0.4\n",
                encoding="utf-8",
            )
            config = load_config(path)
            self.assertEqual(config.search.random_steering_prob, 0.4)

            path.write_text(
                "search:\n"
                "  random_steering_prob: 1.5\n",
                encoding="utf-8",
            )
            with self.assertRaises(ConfigError):
                load_config(path)

    def test_invalid_worker_count_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "hypoevolve.yaml"
            path.write_text("workers:\n  count: 0\n", encoding="utf-8")
            with self.assertRaises(ConfigError):
                load_config(path)

    def test_invalid_archive_bins_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "hypoevolve.yaml"
            path.write_text(
                "archive:\n"
                "  coverage_bins:\n"
                "    - 0.20\n"
                "    - 0.10\n",
                encoding="utf-8",
            )
            with self.assertRaises(ConfigError):
                load_config(path)

            path.write_text(
                "archive:\n"
                "  complexity_bins:\n"
                "    - 3\n"
                "    - 0\n",
                encoding="utf-8",
            )
            with self.assertRaises(ConfigError):
                load_config(path)

    def test_unknown_archive_keys_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "hypoevolve.yaml"
            path.write_text("archive:\n  top_k: 5\n", encoding="utf-8")
            with self.assertRaises(ConfigError):
                load_config(path)


if __name__ == "__main__":
    unittest.main()
