import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from hypoevolve import cli


class TestHypoEvolveCLI(unittest.TestCase):
    def test_help_output_exists(self):
        with patch("sys.argv", ["hypoevolve", "--help"]):
            with self.assertRaises(SystemExit) as ctx:
                cli.parse_args()
            self.assertEqual(ctx.exception.code, 0)

    def test_run_subcommand_happy_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "hypoevolve.yaml"
            config_path.write_text("archive:\n  top_k: 5\nsearch:\n  iterations: 1\n", encoding="utf-8")
            out = io.StringIO()
            with patch("sys.argv", ["hypoevolve", "run", "if A then B", "--config", str(config_path)]):
                with redirect_stdout(out):
                    code = cli.main()
            self.assertEqual(code, 0)
            self.assertIn("Best hypothesis:", out.getvalue())

    def test_doctor_subcommand(self):
        out = io.StringIO()
        with patch("sys.argv", ["hypoevolve", "doctor"]):
            with redirect_stdout(out):
                code = cli.main()
        self.assertEqual(code, 0)
        self.assertIn("python:", out.getvalue())

    def test_run_subcommand_with_workers(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "hypoevolve.yaml"
            config_path.write_text("archive:\n  top_k: 5\nsearch:\n  iterations: 1\nworkers:\n  enabled: true\n  count: 2\n", encoding="utf-8")
            out = io.StringIO()
            with patch("sys.argv", ["hypoevolve", "run", "if A then B", "--config", str(config_path), "--workers", "1"]):
                with redirect_stdout(out):
                    code = cli.main()
            self.assertEqual(code, 0)
            self.assertIn("Best hypothesis:", out.getvalue())


if __name__ == "__main__":
    unittest.main()
