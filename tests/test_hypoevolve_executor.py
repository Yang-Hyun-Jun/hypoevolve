import tempfile
import unittest
from pathlib import Path

from hypoevolve.executor import ExecutionResult, LocalSubprocessExecutor


class TestHypoEvolveExecutor(unittest.TestCase):
    def test_execute_simple_code(self):
        executor = LocalSubprocessExecutor(timeout=5, cleanup=False)
        result = executor.execute("print('hello')")
        self.assertIsInstance(result, ExecutionResult)
        self.assertEqual(result.exit_code, 0)
        self.assertFalse(result.timed_out)
        self.assertEqual(result.stdout.strip(), "hello")
        self.assertTrue(Path(result.work_dir).exists())

    def test_execute_captures_stderr(self):
        executor = LocalSubprocessExecutor(timeout=5, cleanup=False)
        result = executor.execute(
            "import sys\nprint('warn', file=sys.stderr)\nraise SystemExit(2)"
        )
        self.assertEqual(result.exit_code, 2)
        self.assertIn("warn", result.stderr)

    def test_execute_timeout(self):
        executor = LocalSubprocessExecutor(timeout=1, cleanup=False)
        result = executor.execute(
            "import time\ntime.sleep(2)\nprint('done')"
        )
        self.assertTrue(result.timed_out)
        self.assertEqual(result.exit_code, -1)

    def test_execute_supports_extra_files(self):
        executor = LocalSubprocessExecutor(timeout=5, cleanup=False)
        result = executor.execute(
            "from helper import value\nprint(value)",
            files={"helper.py": "value = 123\n"},
        )
        self.assertEqual(result.exit_code, 0)
        self.assertEqual(result.stdout.strip(), "123")

    def test_cleanup_removes_work_dir(self):
        executor = LocalSubprocessExecutor(timeout=5, cleanup=True)
        result = executor.execute("print('x')")
        self.assertFalse(Path(result.work_dir).exists())


    def test_write_files_materializes_nested_support_files(self):
        executor = LocalSubprocessExecutor(timeout=5, cleanup=False)
        with tempfile.TemporaryDirectory() as tmp:
            work_dir = Path(tmp)
            executor._write_files(work_dir, "print('hello')", {"pkg/helper.py": "value = 1\n"})
            self.assertTrue((work_dir / "main.py").exists())
            self.assertTrue((work_dir / "pkg" / "helper.py").exists())


if __name__ == "__main__":
    unittest.main()
