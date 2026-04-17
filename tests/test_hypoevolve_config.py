import tempfile
import unittest
from pathlib import Path

from hypoevolve.config import (
    ConfigError,
    HypoEvolveConfig,
    _ensure_mapping,
    _filter_known,
    _validate_archive_bins,
    load_config,
)
from hypoevolve.dataset import load_dataset_schema


class TestHypoEvolveConfig(unittest.TestCase):
    def test_load_defaults_when_no_path(self):
        config = load_config(None)
        self.assertIsInstance(config, HypoEvolveConfig)
        self.assertEqual(config.llm.model, "DeepSeek-R1-Distill-Qwen-14B")
        self.assertEqual(config.llm.api_base, "http://127.0.0.1:8000/v1")
        self.assertEqual(config.archive.coverage_bins, [0.05, 0.15, 0.30])
        self.assertEqual(config.archive.complexity_bins, [3, 5, 8])

    def test_load_minimal_yaml(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "hypoevolve.yaml"
            path.write_text("search:\n  iterations: 3\n", encoding="utf-8")
            config = load_config(path)
            self.assertEqual(config.search.iterations, 3)
            self.assertEqual(config.archive.coverage_bins, [0.05, 0.15, 0.30])
            self.assertEqual(config.output.top_k_evaluator_code_artifacts, 5)

    def test_llm_api_key_loads_from_yaml(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "hypoevolve.yaml"
            path.write_text(
                "llm:\n"
                "  api_key: EMPTY\n"
                "  api_base: http://127.0.0.1:8000/v1\n"
                "  model: DeepSeek-R1-Distill-Qwen-14B\n",
                encoding="utf-8",
            )
            config = load_config(path)
            self.assertEqual(config.llm.api_key, "EMPTY")
            self.assertEqual(config.llm.api_base, "http://127.0.0.1:8000/v1")
            self.assertEqual(config.llm.model, "DeepSeek-R1-Distill-Qwen-14B")

    def test_output_top_k_evaluator_code_artifacts_loads_and_validates(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "hypoevolve.yaml"
            path.write_text(
                "output:\n"
                "  top_k_evaluator_code_artifacts: 3\n",
                encoding="utf-8",
            )
            config = load_config(path)
            self.assertEqual(config.output.top_k_evaluator_code_artifacts, 3)

            path.write_text(
                "output:\n"
                "  top_k_evaluator_code_artifacts: 0\n",
                encoding="utf-8",
            )
            with self.assertRaises(ConfigError):
                load_config(path)

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

    def test_archive_per_cell_top_k_loads_and_validates(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "hypoevolve.yaml"
            path.write_text(
                "archive:\n"
                "  per_cell_top_k: 7\n",
                encoding="utf-8",
            )
            config = load_config(path)
            self.assertEqual(config.archive.per_cell_top_k, 7)

            path.write_text(
                "archive:\n"
                "  per_cell_top_k: 0\n",
                encoding="utf-8",
            )
            with self.assertRaises(ConfigError):
                load_config(path)

    def test_archive_parent_sampling_mode_loads_and_validates(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "hypoevolve.yaml"
            path.write_text(
                "archive:\n"
                "  parent_sampling_mode: random\n",
                encoding="utf-8",
            )
            config = load_config(path)
            self.assertEqual(config.archive.parent_sampling_mode, "random")

            path.write_text(
                "archive:\n"
                "  parent_sampling_mode: weighted\n",
                encoding="utf-8",
            )
            with self.assertRaises(ConfigError):
                load_config(path)

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

    def test_repo_example_config_points_to_repo_dataset_contract(self):
        config = load_config(Path("hypoevolve.yaml"))
        self.assertEqual(config.evaluator.dataset_schema_path, "dataset.yaml")

        schema = load_dataset_schema(config.evaluator.dataset_schema_path)
        self.assertEqual(schema.description.startswith("Daily oil-market feature parquet dataset"), True)
        self.assertEqual(schema.index.name, "date")
        self.assertEqual(schema.index.dtype, "datetime64[us]")
        self.assertGreater(len(schema.columns), 5)

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


    def test_filter_known_accepts_none_and_rejects_unknown_keys(self):
        self.assertEqual(_filter_known(None, {"a"}), {})
        self.assertEqual(_filter_known({"a": 1}, {"a"}), {"a": 1})
        with self.assertRaises(ConfigError):
            _filter_known({"b": 2}, {"a"})

    def test_validate_archive_bins_rejects_invalid_values(self):
        with self.assertRaises(ConfigError):
            _validate_archive_bins([], [1, 2])
        with self.assertRaises(ConfigError):
            _validate_archive_bins([0.2, 0.1], [1, 2])
        with self.assertRaises(ConfigError):
            _validate_archive_bins([1.2], [1, 2])
        with self.assertRaises(ConfigError):
            _validate_archive_bins([0.2], [1.5])

    def test_ensure_mapping_wraps_yaml_error_as_config_error(self):
        _ensure_mapping({}, "root")
        with self.assertRaises(ConfigError):
            _ensure_mapping([], "root")


if __name__ == "__main__":
    unittest.main()
