from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Protocol


@dataclass(slots=True)
class ExecutionResult:
    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool
    duration_sec: float
    work_dir: str


class CodeExecutor(Protocol):
    def execute(self, code: str, files: Optional[Dict[str, str]] = None) -> ExecutionResult:
        ...


class LocalSubprocessExecutor:
    def __init__(
        self,
        python_bin: Optional[str] = None,
        timeout: int = 30,
        cleanup: bool = False,
    ):
        self.python_bin = python_bin or sys.executable
        self.timeout = timeout
        self.cleanup = cleanup

    def execute(self, code: str, files: Optional[Dict[str, str]] = None) -> ExecutionResult:
        start = time.monotonic()
        work_dir = Path(tempfile.mkdtemp(prefix="hypoevolve-exec-"))
        self._write_files(work_dir, code, files or {})

        try:
            completed = subprocess.run(
                [self.python_bin, "main.py"],
                cwd=work_dir,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            result = ExecutionResult(
                stdout=completed.stdout,
                stderr=completed.stderr,
                exit_code=completed.returncode,
                timed_out=False,
                duration_sec=time.monotonic() - start,
                work_dir=str(work_dir),
            )
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout if isinstance(exc.stdout, str) else (exc.stdout.decode() if exc.stdout else "")
            stderr = exc.stderr if isinstance(exc.stderr, str) else (exc.stderr.decode() if exc.stderr else "")
            result = ExecutionResult(
                stdout=stdout,
                stderr=stderr,
                exit_code=-1,
                timed_out=True,
                duration_sec=time.monotonic() - start,
                work_dir=str(work_dir),
            )
        finally:
            if self.cleanup:
                shutil.rmtree(work_dir, ignore_errors=True)

        return result

    def _write_files(self, work_dir: Path, code: str, files: Dict[str, str]) -> None:
        (work_dir / "main.py").write_text(code, encoding="utf-8")
        for relative_path, content in files.items():
            target = work_dir / relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
