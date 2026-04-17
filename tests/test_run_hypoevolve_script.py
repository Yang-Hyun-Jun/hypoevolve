import os
import stat
import subprocess
import tempfile
import time
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_hypoevolve.sh"


def _write_executable(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


class TestRunHypoevolveScript(unittest.TestCase):
    def test_inner_mode_runs_hypoevolve_sequentially(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            bin_dir = tmp_path / "bin"
            bin_dir.mkdir()
            calls_path = tmp_path / "calls.txt"

            _write_executable(
                bin_dir / "date",
                """#!/bin/sh
if [ "$1" = "+%Y%m%d_%H%M%S" ]; then
  echo "20260417_063000"
elif [ "$1" = "-Is" ]; then
  echo "2026-04-17T06:30:00+00:00"
else
  echo "unexpected date args: $@" >&2
  exit 1
fi
""",
            )
            _write_executable(
                bin_dir / "hypoevolve",
                """#!/bin/sh
printf '%s\\n' "$*" >> "$HYPO_TEST_CALLS"
""",
            )

            env = os.environ.copy()
            env.update(
                {
                    "PATH": f"{bin_dir}:{env['PATH']}",
                    "RUNS": "2",
                    "HYPOEVOLVE_BATCH_INNER": "1",
                    "HYPO_TEST_CALLS": str(calls_path),
                }
            )

            result = subprocess.run(
                [str(SCRIPT_PATH), "--config", "hypoevolve.yaml"],
                cwd=tmp,
                env=env,
                text=True,
                capture_output=True,
                check=True,
            )

            self.assertIn("========== hypoevolve run 1/2 (2026-04-17T06:30:00+00:00) ==========", result.stdout)
            self.assertIn("========== hypoevolve run 2/2 (2026-04-17T06:30:00+00:00) ==========", result.stdout)
            self.assertIn("========== batch finished: 2 runs completed (2026-04-17T06:30:00+00:00) ==========", result.stdout)
            self.assertEqual(
                calls_path.read_text(encoding="utf-8").splitlines(),
                ["run --config hypoevolve.yaml", "run --config hypoevolve.yaml"],
            )

    def test_outer_mode_detaches_with_log_file_and_reinvokes_self(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            bin_dir = tmp_path / "bin"
            bin_dir.mkdir()
            nohup_args_path = tmp_path / "nohup_args.txt"

            _write_executable(
                bin_dir / "date",
                """#!/bin/sh
if [ "$1" = "+%Y%m%d_%H%M%S" ]; then
  echo "20260417_063000"
elif [ "$1" = "-Is" ]; then
  echo "2026-04-17T06:30:00+00:00"
else
  echo "unexpected date args: $@" >&2
  exit 1
fi
""",
            )
            _write_executable(
                bin_dir / "nohup",
                """#!/bin/sh
printf '%s\\n' "$@" > "$HYPO_TEST_NOHUP_ARGS"
exit 0
""",
            )

            env = os.environ.copy()
            env.update(
                {
                    "PATH": f"{bin_dir}:{env['PATH']}",
                    "RUNS": "3",
                    "HYPO_TEST_NOHUP_ARGS": str(nohup_args_path),
                }
            )

            result = subprocess.run(
                [str(SCRIPT_PATH), "--config", "hypoevolve.yaml"],
                cwd=tmp,
                env=env,
                text=True,
                capture_output=True,
                check=True,
            )

            self.assertIn("Starting 3 sequential hypoevolve runs in background (nohup)...", result.stdout)
            self.assertIn("Log file: hypoevolve_batch_20260417_063000.log", result.stdout)
            self.assertIn("PID:", result.stdout)

            log_path = tmp_path / "hypoevolve_batch_20260417_063000.log"
            self.assertTrue(log_path.exists())

            for _ in range(20):
                if nohup_args_path.exists():
                    break
                time.sleep(0.05)
            self.assertTrue(nohup_args_path.exists(), msg="fake nohup was not invoked")
            self.assertEqual(
                nohup_args_path.read_text(encoding="utf-8").splitlines(),
                [str(SCRIPT_PATH), "--config", "hypoevolve.yaml"],
            )


if __name__ == "__main__":
    unittest.main()
