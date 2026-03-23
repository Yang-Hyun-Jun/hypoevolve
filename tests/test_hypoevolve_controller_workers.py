import tempfile
import unittest
import json

from hypoevolve.config import HypoEvolveConfig
from hypoevolve.controller import HypoEvolveController


class ImmediateFuture:
    def __init__(self, result):
        self._result = result
        self._done = True

    def done(self):
        return self._done

    def result(self):
        return self._result


class FakeExecutor:
    def __init__(self, max_workers=1):
        self.max_workers = max_workers

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def submit(self, fn, task):
        return ImmediateFuture(fn(task))


class DelayedFuture:
    def __init__(self, result, polls_before_done: int):
        self._result = result
        self._polls_before_done = polls_before_done
        self._polls = 0

    def done(self):
        self._polls += 1
        return self._polls > self._polls_before_done

    def result(self):
        return self._result


class DelayedExecutor(FakeExecutor):
    def __init__(self, delays):
        super().__init__(max_workers=len(delays))
        self.delays = list(delays)

    def submit(self, fn, task):
        delay = self.delays.pop(0) if self.delays else 0
        return DelayedFuture(fn(task), delay)


class TestHypoEvolveControllerWorkers(unittest.TestCase):
    def test_worker_enabled_run_completes_and_writes_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = HypoEvolveConfig()
            config.search.iterations = 3
            config.output.base_dir = tmp
            config.workers.enabled = True
            config.workers.count = 2
            controller = HypoEvolveController(config, executor_factory=FakeExecutor)
            result = controller.run("if A then B")
            self.assertTrue((result.run_dir / "trace.jsonl").exists())
            self.assertTrue((result.run_dir / "checkpoint.json").exists())
            self.assertTrue((result.run_dir / "best.json").exists())

    def test_single_process_fallback_when_workers_disabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = HypoEvolveConfig()
            config.search.iterations = 1
            config.output.base_dir = tmp
            config.workers.enabled = False
            controller = HypoEvolveController(config)
            result = controller.run("if A then B")
            self.assertTrue(result.run_dir.exists())

    def test_single_process_fallback_when_worker_count_is_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = HypoEvolveConfig()
            config.search.iterations = 1
            config.output.base_dir = tmp
            config.workers.enabled = True
            config.workers.count = 1
            controller = HypoEvolveController(config, executor_factory=FakeExecutor)
            result = controller.run("if A then B")
            self.assertTrue(result.run_dir.exists())

    def test_worker_results_are_reflected_as_completed(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = HypoEvolveConfig()
            config.search.iterations = 3
            config.output.base_dir = tmp
            config.workers.enabled = True
            config.workers.count = 2
            controller = HypoEvolveController(
                config,
                executor_factory=lambda max_workers=2: DelayedExecutor([2, 0, 0]),
            )
            result = controller.run("if A then B")
            trace_path = result.run_dir / "trace.jsonl"
            lines = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(lines[0]["iteration"], 0)
            self.assertEqual(lines[1]["iteration"], 2)
