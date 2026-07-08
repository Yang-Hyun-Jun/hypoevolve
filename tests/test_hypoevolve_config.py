import tempfile
import unittest
from pathlib import Path

from hypoevolve.core.config import (
    ConfigError,
    HypoEvolveConfig,
    _ensure_mapping,
    _filter_known,
    load_config,
    load_runtime_config,
    resolve_config_path,
)
from hypoevolve.data.dataset import load_dataset_schema


class TestHypoEvolveConfig(unittest.TestCase):
    def test_load_defaults_when_no_path(self):
        config = load_config(None)
        self.assertIsInstance(config, HypoEvolveConfig)
        self.assertEqual(config.llm.model, "DeepSeek-R1-Distill-Qwen-14B")
        self.assertEqual(config.llm.api_base, "http://127.0.0.1:8000/v1")
        self.assertEqual(config.archive.capacity, 64)
        self.assertAlmostEqual(config.archive.gamma, 0.3)
        self.assertAlmostEqual(config.archive.eps, 1e-2)

    def test_load_minimal_yaml(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "hypoevolve.yaml"
            path.write_text("search:\n  iterations: 3\n", encoding="utf-8")
            config = load_config(path)
            self.assertEqual(config.search.iterations, 3)
            self.assertEqual(config.archive.capacity, 64)
            self.assertEqual(config.output.top_k_evaluator_code_artifacts, 5)

    def test_resolve_config_path_uses_explicit_or_default_location(self):
        self.assertEqual(
            resolve_config_path("custom.yaml", "ignored.yaml"),
            Path("custom.yaml"),
        )
        self.assertEqual(
            resolve_config_path(None, "hypoevolve.local.yaml"),
            Path("hypoevolve.local.yaml"),
        )

    def test_load_runtime_config_rejects_explicit_missing_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ConfigError):
                load_runtime_config(Path(tmp) / "missing.yaml")

    def test_load_runtime_config_allows_missing_default_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = load_runtime_config(None, default_path=Path(tmp) / "missing.yaml")
        self.assertEqual(config.search.iterations, 5)

    def test_load_runtime_config_reads_present_default_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            default_path = Path(tmp) / "hypoevolve.yaml"
            default_path.write_text("search:\n  iterations: 7\n", encoding="utf-8")

            config = load_runtime_config(None, default_path=default_path)

        self.assertEqual(config.search.iterations, 7)

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

    def test_archive_settings_load_from_yaml(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "hypoevolve.yaml"
            path.write_text(
                "archive:\n"
                "  capacity: 32\n"
                "  gamma: 0.5\n"
                "  eps: 0.001\n",
                encoding="utf-8",
            )
            config = load_config(path)
            self.assertEqual(config.archive.capacity, 32)
            self.assertAlmostEqual(config.archive.gamma, 0.5)
            self.assertAlmostEqual(config.archive.eps, 0.001)

    def test_archive_capacity_must_be_positive(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "hypoevolve.yaml"
            path.write_text("archive:\n  capacity: 0\n", encoding="utf-8")
            with self.assertRaises(ConfigError):
                load_config(path)

    def test_archive_gamma_must_be_non_negative(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "hypoevolve.yaml"
            path.write_text("archive:\n  gamma: -0.1\n", encoding="utf-8")
            with self.assertRaises(ConfigError):
                load_config(path)

    def test_archive_eps_must_be_positive(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "hypoevolve.yaml"
            path.write_text("archive:\n  eps: 0\n", encoding="utf-8")
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

    def test_ensure_mapping_wraps_yaml_error_as_config_error(self):
        _ensure_mapping({}, "root")
        with self.assertRaises(ConfigError):
            _ensure_mapping([], "root")


if __name__ == "__main__":
    unittest.main()
