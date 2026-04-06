import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
import os
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
                report_path=Path(tmp) / "run1" / "report" / "report.md",
                seed_hypothesis=SimpleNamespace(),
                best_hypothesis=SimpleNamespace(),
                best_metrics={"combined_score": 0.9},
            )
            with patch("hypoevolve.cli.HypoEvolveController") as controller_cls, patch(
                "hypoevolve.cli.render_pretty",
                side_effect=[
                    "IMPLIES(\n  SEED_A,\n  SEED_B\n)",
                    "IMPLIES(\n  BEST_A,\n  BEST_B\n)",
                ],
            ):
                controller_cls.return_value.run.return_value = fake_result
                result = self.runner.invoke(
                    cli.app,
                    ["run", "if A then B", "--config", str(config_path)],
                )
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Run Summary", result.output)
        self.assertIn("Report", result.output)
        self.assertIn("Metric highlights", result.output)
        self.assertIn("Initial hypothesis", result.output)
        self.assertIn("Best hypothesis", result.output)
        self.assertIn("IMPLIES(\n  SEED_A,\n  SEED_B\n)", result.output)
        self.assertIn("IMPLIES(\n  BEST_A,\n  BEST_B\n)", result.output)
        self.assertLess(
            result.output.index("Initial hypothesis"),
            result.output.index("Best hypothesis"),
        )
        self.assertIn('"combined_score": 0.9', result.output)

    def test_run_subcommand_without_hypothesis_uses_generated_seed_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "hypoevolve.yaml"
            config_path.write_text("search:\n  iterations: 1\n", encoding="utf-8")
            fake_result = SimpleNamespace(
                run_dir=Path(tmp) / "run-seed",
                report_path=Path(tmp) / "run-seed" / "report" / "report.md",
                seed_hypothesis=SimpleNamespace(),
                best_hypothesis=SimpleNamespace(),
                best_metrics={"combined_score": 0.7},
                seed_input_text="Generated seed hypothesis.",
                seed_generated=True,
            )
            with patch("hypoevolve.cli.HypoEvolveController") as controller_cls, patch(
                "hypoevolve.cli.render_pretty",
                side_effect=[
                    "IMPLIES(\n  SEED_A,\n  SEED_B\n)",
                    "IMPLIES(\n  BEST_A,\n  BEST_B\n)",
                ],
            ):
                controller_cls.return_value.run.return_value = fake_result
                result = self.runner.invoke(
                    cli.app,
                    ["run", "--config", str(config_path)],
                )

        self.assertEqual(result.exit_code, 0)
        self.assertIn("Seed source", result.output)
        self.assertIn("generated", result.output)
        self.assertIn("Seed hypothesis", result.output)
        self.assertIn("Generated seed hypothesis.", result.output)
        self.assertIn("Initial hypothesis", result.output)
        controller_cls.return_value.run.assert_called_once_with(None)

    def test_seed_subcommand_prints_trees_and_hypothesis(self):
        fake_result = SimpleNamespace(
            tree_a=SimpleNamespace(render=lambda return_str=False: "TREE A"),
            tree_b=SimpleNamespace(render=lambda return_str=False: "TREE B"),
            hypothesis="Generated hypothesis.",
        )
        with patch("hypoevolve.cli.generate_random_tree_pair_hypothesis", return_value=fake_result):
            result = self.runner.invoke(cli.app, ["seed"])

        self.assertEqual(result.exit_code, 0)
        self.assertIn("Feature tree A", result.output)
        self.assertIn("TREE A", result.output)
        self.assertIn("Feature tree B", result.output)
        self.assertIn("TREE B", result.output)
        self.assertIn("Generated hypothesis", result.output)
        self.assertIn("Generated hypothesis.", result.output)

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

    def test_inspect_subcommand_json_mode_returns_raw_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload_path = Path(tmp) / "best.json"
            payload_path.write_text(
                '{"metrics": {"combined_score": 0.5}, "hello": "world"}',
                encoding="utf-8",
            )
            result = self.runner.invoke(cli.app, ["inspect", str(payload_path), "--json"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn('"hello": "world"', result.output)
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
                report_path=Path(tmp) / "run2" / "report" / "report.md",
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

    def test_runs_latest_json_returns_newest_run_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "hypoevolve.yaml"
            runs_dir = Path(tmp) / "runs"
            runs_dir.mkdir()
            older = runs_dir / "older"
            newer = runs_dir / "newer"
            older.mkdir()
            newer.mkdir()
            os.utime(older, (1, 1))
            os.utime(newer, (2, 2))
            config_path.write_text(f"output:\n  base_dir: {runs_dir}\n", encoding="utf-8")
            result = self.runner.invoke(
                cli.app,
                ["runs", "latest", "--config", str(config_path), "--json"],
            )
        self.assertEqual(result.exit_code, 0)
        self.assertIn(str(newer), result.output)

    def test_status_json_reads_run_summary_and_report_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run1"
            (run_dir / "report").mkdir(parents=True)
            (run_dir / "run_summary.json").write_text(
                '{"iterations_requested": 10, "best_score": 0.7, "best_hypothesis_nl": "If A then B.", "archive_size": 4, "duplicate_skips_total": 2}',
                encoding="utf-8",
            )
            (run_dir / "checkpoint.json").write_text(
                '{"iteration": 10, "archive_size": 4}',
                encoding="utf-8",
            )
            (run_dir / "score_history.json").write_text("[]", encoding="utf-8")
            (run_dir / "report" / "report.md").write_text("# report\n", encoding="utf-8")
            result = self.runner.invoke(cli.app, ["status", str(run_dir), "--json"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn('"status": "completed"', result.output)
        self.assertIn('"best_score": 0.7', result.output)
        self.assertIn('"report_path"', result.output)

    def test_status_json_marks_failed_when_checkpoint_and_summary_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run1"
            run_dir.mkdir()
            result = self.runner.invoke(cli.app, ["status", str(run_dir), "--json"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn('"status": "failed"', result.output)

    def test_report_json_regenerates_missing_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run1"
            run_dir.mkdir()
            (run_dir / "best.json").write_text(
                '{"hypothesis": {"root": {"kind": "atomic", "name": "A", "type": "boolean", "source": "primitive", "params": {}}, "params": {}}, "metrics": {"combined_score": 0.5}}',
                encoding="utf-8",
            )
            (run_dir / "checkpoint.json").write_text(
                '{"iteration": 0, "archive_size": 1, "archive": [], "best_hypothesis": {"root": {"kind": "atomic", "name": "A", "type": "boolean", "source": "primitive", "params": {}}, "params": {}}, "best_metrics": {"combined_score": 0.5}}',
                encoding="utf-8",
            )
            (run_dir / "run_summary.json").write_text(
                '{"iterations_requested": 1, "best_score": 0.5, "best_hypothesis_nl": "A", "archive_size": 1, "duplicate_skips_total": 0, "best_fingerprint": ""}',
                encoding="utf-8",
            )
            (run_dir / "score_history.json").write_text(
                '[{"iteration": 0, "score": 0.5, "best_updated": true, "hypothesis_nl": "A"}]',
                encoding="utf-8",
            )
            result = self.runner.invoke(cli.app, ["report", str(run_dir), "--json"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn('"report_path"', result.output)
        self.assertIn("report.md", result.output)

    def test_runs_status_json_resolves_run_id_under_configured_base_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "hypoevolve.yaml"
            runs_dir = Path(tmp) / "runs"
            run_id = "abcd1234"
            run_dir = runs_dir / run_id
            run_dir.mkdir(parents=True)
            (run_dir / "checkpoint.json").write_text(
                '{"iteration": 3, "archive_size": 2, "best_metrics": {"combined_score": 0.4}}',
                encoding="utf-8",
            )
            config_path.write_text(f"output:\n  base_dir: {runs_dir}\n", encoding="utf-8")
            result = self.runner.invoke(
                cli.app,
                ["runs", "status", run_id, "--config", str(config_path), "--json"],
            )
        self.assertEqual(result.exit_code, 0)
        self.assertIn('"status": "running"', result.output)
        self.assertIn('"current_iteration": 3', result.output)
        self.assertIn(str(run_dir), result.output)

    def test_runs_report_json_resolves_run_id_under_configured_base_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "hypoevolve.yaml"
            runs_dir = Path(tmp) / "runs"
            run_id = "abcd1234"
            run_dir = runs_dir / run_id
            run_dir.mkdir(parents=True)
            (run_dir / "best.json").write_text(
                '{"hypothesis": {"root": {"kind": "atomic", "name": "A", "type": "boolean", "source": "primitive", "params": {}}, "params": {}}, "metrics": {"combined_score": 0.5}}',
                encoding="utf-8",
            )
            (run_dir / "checkpoint.json").write_text(
                '{"iteration": 0, "archive_size": 1, "archive": [], "best_hypothesis": {"root": {"kind": "atomic", "name": "A", "type": "boolean", "source": "primitive", "params": {}}, "params": {}}, "best_metrics": {"combined_score": 0.5}}',
                encoding="utf-8",
            )
            (run_dir / "run_summary.json").write_text(
                '{"iterations_requested": 1, "best_score": 0.5, "best_hypothesis_nl": "A", "archive_size": 1, "duplicate_skips_total": 0, "best_fingerprint": ""}',
                encoding="utf-8",
            )
            (run_dir / "score_history.json").write_text(
                '[{"iteration": 0, "score": 0.5, "best_updated": true, "hypothesis_nl": "A"}]',
                encoding="utf-8",
            )
            config_path.write_text(f"output:\n  base_dir: {runs_dir}\n", encoding="utf-8")
            result = self.runner.invoke(
                cli.app,
                ["runs", "report", run_id, "--config", str(config_path), "--json"],
            )
        self.assertEqual(result.exit_code, 0)
        self.assertIn('"run_id": "abcd1234"', result.output)
        self.assertIn('"report_path"', result.output)
        self.assertIn("report.md", result.output)

    def test_runs_status_errors_for_missing_run_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "hypoevolve.yaml"
            runs_dir = Path(tmp) / "runs"
            runs_dir.mkdir()
            config_path.write_text(f"output:\n  base_dir: {runs_dir}\n", encoding="utf-8")
            result = self.runner.invoke(
                cli.app,
                ["runs", "status", "missing", "--config", str(config_path)],
            )
        self.assertEqual(result.exit_code, 1)
        self.assertIn("Run id not found", result.output)


if __name__ == "__main__":
    unittest.main()
