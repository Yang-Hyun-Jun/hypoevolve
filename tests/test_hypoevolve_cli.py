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
        self.assertIn(
            'if signal A weakens then event B becomes more likely', result.output
        )
        self.assertNotIn("BTC momentum drops", result.output)

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
            with patch("hypoevolve.cli.commands.HypoEvolveController") as controller_cls, patch(
                "hypoevolve.cli.commands.render_pretty",
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
            with patch("hypoevolve.cli.commands.HypoEvolveController") as controller_cls, patch(
                "hypoevolve.cli.commands.render_pretty",
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
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "hypoevolve.yaml"
            dataset_path = Path(tmp) / "seed-dataset.yaml"
            dataset_path.write_text(
                "description: seed dataset\n"
                "index:\n"
                "  name: close_time\n"
                "  dtype: datetime64[us]\n"
                "files:\n"
                "  -\n"
                "    entity: BTCUSDT\n"
                "    path: data/BTCUSDT.parquet\n"
                "columns:\n"
                "  -\n"
                "    name: CLOSE\n",
                encoding="utf-8",
            )
            config_path.write_text(
                "evaluator:\n"
                f"  dataset_schema_path: {dataset_path}\n",
                encoding="utf-8",
            )
            with patch(
                "hypoevolve.cli.commands.generate_random_tree_pair_hypothesis",
                return_value=fake_result,
            ) as generate_mock:
                result = self.runner.invoke(
                    cli.app, ["seed", "--config", str(config_path)]
                )

        self.assertEqual(result.exit_code, 0)
        self.assertIn("Feature tree A", result.output)
        self.assertIn("TREE A", result.output)
        self.assertIn("Feature tree B", result.output)
        self.assertIn("TREE B", result.output)
        self.assertIn("Generated hypothesis", result.output)
        self.assertIn("Generated hypothesis.", result.output)
        generate_mock.assert_called_once()
        self.assertEqual(
            generate_mock.call_args.kwargs["dataset_schema_path"],
            str(dataset_path),
        )

    def test_doctor_subcommand(self):
        result = self.runner.invoke(cli.app, ["doctor"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Diagnostics", result.output)
        self.assertIn("Python", result.output)

    def test_doctor_reports_archive_kind_from_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "hypoevolve.yaml"
            config_path.write_text(
                (
                    "archive:\n"
                    "  kind: coulomb\n"
                    "  coulomb:\n"
                    "    capacity: 16\n"
                    "    gamma: 0.5\n"
                    "    eps: 0.02\n"
                    "evaluator:\n"
                    "  dataset_schema_path: dataset.yaml\n"
                ),
                encoding="utf-8",
            )
            result = self.runner.invoke(
                cli.app, ["doctor", "--config", str(config_path)]
            )
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Coulomb", result.output)
        self.assertIn("Coulomb capacity", result.output)
        self.assertIn("Coulomb gamma", result.output)
        # Coulomb kind should NOT print the MAP-Elites-specific fields.
        self.assertNotIn("Coverage bins", result.output)
        self.assertNotIn("Complexity bins", result.output)

    def test_doctor_still_reports_map_elites_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "hypoevolve.yaml"
            config_path.write_text(
                "evaluator:\n  dataset_schema_path: dataset.yaml\n",
                encoding="utf-8",
            )
            result = self.runner.invoke(
                cli.app, ["doctor", "--config", str(config_path)]
            )
        self.assertEqual(result.exit_code, 0)
        self.assertIn("MAP-Elites", result.output)
        self.assertIn("Coverage bins", result.output)
        self.assertIn("Complexity bins", result.output)

    def test_render_subcommand_tree_mode(self):
        with patch("hypoevolve.cli.commands.LLMClient"), patch(
            "hypoevolve.cli.commands.parse_hypothesis_text", return_value=SimpleNamespace()
        ), patch("hypoevolve.cli.commands.render_tree", return_value="ROOT\n└── A"):
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
                '{"hypothesis": {"kind": "atomic", "name": "A"}, "metrics": {"combined_score": 0.5}}',
                encoding="utf-8",
            )
            with patch("hypoevolve.cli.commands.hypothesis_from_dict", return_value=SimpleNamespace()), patch(
                "hypoevolve.cli.commands.render_pretty", return_value="A"
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
            with patch("hypoevolve.cli.commands.HypoEvolveController") as controller_cls, patch(
                "hypoevolve.cli.commands.render_pretty", return_value="SUPPORT(\n  A,\n  B\n)"
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
            config_path.write_text(f"output:\n  base_dir: {tmp}/runs\n", encoding="utf-8")
            result = self.runner.invoke(
                cli.app,
                ["runs", "latest", "--config", str(config_path), "--json"],
            )
        self.assertEqual(result.exit_code, 0)
        self.assertIn(str(newer), result.output)

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
            config_path.write_text(f"output:\n  base_dir: {tmp}/runs\n", encoding="utf-8")
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
                '{"hypothesis": {"root": {"kind": "atomic", "name": "A"}}, "metrics": {"combined_score": 0.5}}',
                encoding="utf-8",
            )
            (run_dir / "checkpoint.json").write_text(
                '{"iteration": 0, "archive_size": 1, "archive": [], "best_hypothesis": {"root": {"kind": "atomic", "name": "A"}}, "best_metrics": {"combined_score": 0.5}}',
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
            config_path.write_text(f"output:\n  base_dir: {tmp}/runs\n", encoding="utf-8")
            result = self.runner.invoke(
                cli.app,
                ["runs", "report", run_id, "--config", str(config_path), "--json"],
            )
        self.assertEqual(result.exit_code, 0)
        self.assertIn('"run_id": "abcd1234"', result.output)
        self.assertIn('"report_path"', result.output)
        self.assertIn('report.md', result.output)

    def test_runs_status_errors_for_missing_run_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "hypoevolve.yaml"
            runs_dir = Path(tmp) / "runs"
            runs_dir.mkdir()
            config_path.write_text(f"output:\n  base_dir: {tmp}/runs\n", encoding="utf-8")
            result = self.runner.invoke(
                cli.app,
                ["runs", "status", "missing", "--config", str(config_path)],
            )
        self.assertEqual(result.exit_code, 1)
        self.assertIn("Run id not found", result.output)

    def test_render_helpers_cover_banner_kv_and_metric_output(self):
        banner = cli._render_banner()
        self.assertIn('HypoEvolve', banner)
        self.assertIn('LLM-guided ELG hypothesis evolution', banner)
        section = cli._render_kv_section('Section', [('Alpha', '1'), ('Beta', '2')])
        self.assertIn('Section:', section)
        self.assertIn('Alpha', section)
        out = StringIO()
        with redirect_stdout(out):
            cli._echo_metric_highlights({'combined_score': 0.7, 'precision': 0.6, 'coverage': 0.3, 'uplift': 0.2})
        rendered = out.getvalue()
        self.assertIn('Metric highlights', rendered)
        self.assertIn('0.7000', rendered)

    def test_latest_run_dir_and_run_dir_from_id_behave_as_expected(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            older = base / 'older'
            newer = base / 'newer'
            older.mkdir()
            newer.mkdir()
            os.utime(older, (1, 1))
            os.utime(newer, (2, 2))
            self.assertEqual(cli._latest_run_dir(base), newer)
            self.assertEqual(cli._run_dir_from_id(base, 'older'), older)
            with self.assertRaises(cli.ConfigError):
                cli._run_dir_from_id(base, 'missing')

    def test_status_payload_prefers_summary_and_backfills_from_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / 'run1'
            run_dir.mkdir()
            (run_dir / 'checkpoint.json').write_text('{"iteration": 2, "archive_size": 3, "best_metrics": {"combined_score": 0.4}}', encoding='utf-8')
            (run_dir / 'score_history.json').write_text('[{"iteration": 1, "best_updated": true, "hypothesis_nl": "If A then B."}]', encoding='utf-8')
            payload = cli._status_payload(run_dir)
        self.assertEqual(payload['status'], 'running')
        self.assertEqual(payload['current_iteration'], 2)
        self.assertEqual(payload['best_score'], 0.4)
        self.assertEqual(payload['best_hypothesis_nl'], 'If A then B.')

    def test_read_json_if_exists_and_resolve_runs_base_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'payload.json'
            path.write_text('{"hello": "world"}', encoding='utf-8')
            self.assertEqual(cli._read_json_if_exists(path), {'hello': 'world'})
            self.assertIsNone(cli._read_json_if_exists(Path(tmp) / 'missing.json'))
            config_path = Path(tmp) / 'hypoevolve.yaml'
            config_path.write_text(f"output:\n  base_dir: {tmp}/runs\n", encoding='utf-8')
            self.assertEqual(cli._resolve_runs_base_dir(str(config_path)), Path(tmp) / 'runs')


    def test_detect_version_returns_dev_when_package_metadata_missing(self):
        with patch('hypoevolve.cli.metadata.version', side_effect=cli.metadata.PackageNotFoundError):
            self.assertEqual(cli._detect_version(), 'dev')

    def test_echo_helpers_emit_expected_text(self):
        echo_calls = []
        secho_calls = []

        with patch("hypoevolve.cli.click.echo", side_effect=lambda message="": echo_calls.append(message)), patch(
            "hypoevolve.cli.click.secho",
            side_effect=lambda message="", **kwargs: secho_calls.append((message, kwargs)),
        ):
            cli._echo_banner()
            cli._echo_json({"hello": "world"})
            cli._echo_block("Title", "Body")
            cli._echo_error(Exception("boom"))

        self.assertTrue(any("HypoEvolve" in call for call in echo_calls))
        self.assertTrue(any('"hello": "world"' in call for call in echo_calls))
        self.assertIn(("◆ Title", {"fg": "cyan", "bold": True}), secho_calls)
        self.assertIn("Body", echo_calls)
        self.assertIn(("Error: boom", {"fg": "red", "err": True}), secho_calls)

    def test_echo_error_prints_nested_error_list_items(self):
        class FakeError(Exception):
            def __init__(self):
                super().__init__("failed")
                self.errors = ["first", "second"]

        echo_calls = []
        secho_calls = []
        with patch("hypoevolve.cli.click.echo", side_effect=lambda message="": echo_calls.append(message)), patch(
            "hypoevolve.cli.click.secho",
            side_effect=lambda message="", **kwargs: secho_calls.append((message, kwargs)),
        ):
            cli._echo_error(FakeError())

        self.assertIn(("Error: failed", {"fg": "red", "err": True}), secho_calls)
        self.assertIn("  - first", echo_calls)
        self.assertIn("  - second", echo_calls)


if __name__ == "__main__":
    unittest.main()
