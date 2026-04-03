import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from click.testing import CliRunner

from hypoevolve import cli


class TestHypoEvolveCLI(unittest.TestCase):
    def setUp(self) -> None:
        self.runner = CliRunner()

    def test_help_output_exists(self):
        result = self.runner.invoke(cli.app, ["--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("HypoEvolve", result.output)
        self.assertIn("Quick start:", result.output)
        self.assertIn("Commands:", result.output)

    def test_version_option_exists(self):
        result = self.runner.invoke(cli.app, ["--version"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("hypoevolve", result.output.lower())

    def test_main_without_args_prints_banner_and_help(self):
        result = self.runner.invoke(cli.app, [])
        self.assertEqual(result.exit_code, 1)
        self.assertIn("HypoEvolve", result.output)
        self.assertIn("run", result.output)
        self.assertIn("doctor", result.output)

    def test_main_wrapper_without_args_returns_error_code(self):
        out = StringIO()
        with redirect_stdout(out):
            code = cli.main([])
        self.assertEqual(code, 1)
        self.assertIn("HypoEvolve", out.getvalue())

    def test_run_subcommand_happy_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "hypoevolve.yaml"
            config_path.write_text(
                "search:\n  iterations: 1\n",
                encoding="utf-8",
            )
            fake_result = SimpleNamespace(
                run_dir=Path(tmp) / "run1",
                best_hypothesis=SimpleNamespace(),
                best_metrics={"combined_score": 0.9},
            )
            with patch("hypoevolve.cli.HypoEvolveController") as controller_cls, patch(
                "hypoevolve.cli.render_pretty", return_value="IMPLIES(\n  A,\n  B\n)"
            ):
                controller_cls.return_value.run.return_value = fake_result
                result = self.runner.invoke(
                    cli.app,
                    ["run", "if A then B", "--config", str(config_path)],
                )
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Run Summary", result.output)
        self.assertIn("Metric highlights", result.output)
        self.assertIn("Best hypothesis", result.output)
        self.assertIn('"combined_score": 0.9', result.output)

    def test_doctor_subcommand(self):
        result = self.runner.invoke(cli.app, ["doctor"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Diagnostics", result.output)
        self.assertIn("Python", result.output)

    def test_render_subcommand_tree_mode(self):
        with patch("hypoevolve.cli.LLMClient"), patch(
            "hypoevolve.cli.parse_hypothesis_text", return_value=SimpleNamespace()
        ), patch("hypoevolve.cli.render_tree", return_value="ROOT\n└── A"):
            result = self.runner.invoke(
                cli.app,
                ["render", "if A then B", "--tree"],
            )
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Rendered hypothesis", result.output)
        self.assertIn("ROOT", result.output)

    def test_inspect_subcommand_displays_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload_path = Path(tmp) / "best.json"
            payload_path.write_text(
                '{"hypothesis": {"kind": "atomic", "name": "A", "type": "boolean", "source": "primitive", "params": {}}, "metrics": {"combined_score": 0.5}}',
                encoding="utf-8",
            )
            with patch("hypoevolve.cli.hypothesis_from_dict", return_value=SimpleNamespace()), patch(
                "hypoevolve.cli.render_pretty", return_value="A"
            ):
                result = self.runner.invoke(cli.app, ["inspect", str(payload_path)])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Artifact", result.output)
        self.assertIn("Metrics", result.output)
        self.assertIn('"combined_score": 0.5', result.output)

    def test_run_subcommand_with_workers(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "hypoevolve.yaml"
            config_path.write_text(
                "search:\n  iterations: 1\nworkers:\n  enabled: true\n  count: 2\n",
                encoding="utf-8",
            )
            fake_result = SimpleNamespace(
                run_dir=Path(tmp) / "run2",
                best_hypothesis=SimpleNamespace(),
                best_metrics={"combined_score": 0.8},
            )
            with patch("hypoevolve.cli.HypoEvolveController") as controller_cls, patch(
                "hypoevolve.cli.render_pretty", return_value="SUPPORT(\n  A,\n  B\n)"
            ):
                controller_cls.return_value.run.return_value = fake_result
                result = self.runner.invoke(
                    cli.app,
                    [
                        "run",
                        "if A then B",
                        "--config",
                        str(config_path),
                        "--workers",
                        "1",
                    ],
                )
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Workers", result.output)
        self.assertIn("1", result.output)

    def test_load_runtime_config_rejects_explicit_missing_config(self):
        with self.runner.isolated_filesystem():
            with self.assertRaises(cli.ConfigError):
                cli._load_runtime_config("missing.yaml")

    def test_load_runtime_config_allows_missing_default_file(self):
        with self.runner.isolated_filesystem():
            config = cli._load_runtime_config(None)
        self.assertEqual(config.search.iterations, 5)


if __name__ == "__main__":
    unittest.main()
