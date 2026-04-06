import tempfile
import unittest
import json
from pathlib import Path
from unittest.mock import patch

from hypoevolve.config import HypoEvolveConfig
from hypoevolve.controller import HypoEvolveController
from hypoevolve.workers import WorkerResult


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
        class FakeLLM:
            def generate_json(self, system, user, **kwargs):
                return {
                    "child_hypothesis": {
                        "kind": "relation",
                        "type": "IMPLIES",
                        "inputs": [
                            {"kind": "atomic", "name": "A2", "type": "abstract", "source": "semantic", "params": {}},
                            {"kind": "atomic", "name": "B", "type": "abstract", "source": "semantic", "params": {}},
                        ],
                        "params": {},
                    },
                    "domain_reason": "Tightening the stress condition is plausible from a crypto downside-regime perspective.",
                    "score_reason": "Tightening one atomic condition is a local change that may improve precision.",
                    "operation_score_rankings": {"replace_atomic_feature": 1, "append_atomic": 2},
                    "mutation_summary": "Applied a replace_atomic-style change in the condition side while keeping the overall relation structure.",
                } if "mutation_summary" in system else {
                    "kind": "relation",
                    "type": "IMPLIES",
                    "inputs": [
                        {"kind": "atomic", "name": "A", "type": "abstract", "source": "semantic", "params": {}},
                        {"kind": "atomic", "name": "B", "type": "abstract", "source": "semantic", "params": {}},
                    ],
                    "params": {},
                }

            def generate_text(self, system, user, **kwargs):
                return "If A then B."

        def fake_run_worker_task(task):
            return WorkerResult(
                child_hypothesis=task.parent_hypothesis,
                metrics={"combined_score": 0.6},
                iteration=task.iteration,
                mutation_summary="Applied a change_relation_type-style local mutation.",
                parent_score=task.parent_score,
                domain_reason="Use a plausible local relation-type mutation.",
                score_reason="Use a local relation-type mutation.",
                operation_score_rankings={"change_relation_type": 1},
            )

        with tempfile.TemporaryDirectory() as tmp:
            config = HypoEvolveConfig()
            config.search.iterations = 3
            config.output.base_dir = tmp
            config.workers.enabled = True
            config.workers.count = 2
            config.evaluator.dataset_schema_path = str(Path(tmp) / "dataset.yaml")
            Path(config.evaluator.dataset_schema_path).write_text(
                "description: test\nindex:\n  name: close_time\n  dtype: datetime64[us]\nfiles:\n  -\n    entity: BTCUSDT\n    path: /tmp/BTCUSDT.parquet\ncolumns:\n  -\n    name: CLOSE\n",
                encoding="utf-8",
            )
            with patch("hypoevolve.controller.run_worker_task", side_effect=fake_run_worker_task):
                controller = HypoEvolveController(config, llm_client=FakeLLM(), evaluator=type("FakeEvaluator", (), {"evaluate": lambda self, hypothesis: {"combined_score": 0.5}})(), executor_factory=FakeExecutor)
                result = controller.run("if A then B")
            self.assertTrue((result.run_dir / "trace.jsonl").exists())
            self.assertTrue((result.run_dir / "checkpoint.json").exists())
            self.assertTrue((result.run_dir / "best.json").exists())
            self.assertTrue((result.run_dir / "run_summary.json").exists())
            self.assertTrue((result.run_dir / "score_history.json").exists())
            self.assertTrue((result.run_dir / "report" / "report.md").exists())

    def test_single_process_fallback_when_workers_disabled(self):
        class FakeLLM:
            def generate_json(self, system, user, **kwargs):
                return {
                    "child_hypothesis": {
                        "kind": "relation",
                        "type": "IMPLIES",
                        "inputs": [
                            {"kind": "atomic", "name": "A2", "type": "abstract", "source": "semantic", "params": {}},
                            {"kind": "atomic", "name": "B", "type": "abstract", "source": "semantic", "params": {}},
                        ],
                        "params": {},
                    },
                    "domain_reason": "Tightening the stress condition is plausible from a crypto downside-regime perspective.",
                    "score_reason": "Tightening one atomic condition is a local change that may improve precision.",
                    "operation_score_rankings": {"replace_atomic_feature": 1, "append_atomic": 2},
                    "mutation_summary": "Applied a replace_atomic-style change in the condition side while keeping the overall relation structure.",
                } if "mutation_summary" in system else {
                    "kind": "relation",
                    "type": "IMPLIES",
                    "inputs": [
                        {"kind": "atomic", "name": "A", "type": "abstract", "source": "semantic", "params": {}},
                        {"kind": "atomic", "name": "B", "type": "abstract", "source": "semantic", "params": {}},
                    ],
                    "params": {},
                }
            def generate_text(self, system, user, **kwargs):
                return "If A then B."
        with tempfile.TemporaryDirectory() as tmp:
            config = HypoEvolveConfig()
            config.search.iterations = 1
            config.output.base_dir = tmp
            config.workers.enabled = False
            config.evaluator.dataset_schema_path = str(Path(tmp) / "dataset.yaml")
            Path(config.evaluator.dataset_schema_path).write_text(
                "description: test\nindex:\n  name: close_time\n  dtype: datetime64[us]\nfiles:\n  -\n    entity: BTCUSDT\n    path: /tmp/BTCUSDT.parquet\ncolumns:\n  -\n    name: CLOSE\n",
                encoding="utf-8",
            )
            controller = HypoEvolveController(config, llm_client=FakeLLM(), evaluator=type("FakeEvaluator", (), {"evaluate": lambda self, hypothesis: {"combined_score": 0.5}})())
            result = controller.run("if A then B")
            self.assertTrue(result.run_dir.exists())

    def test_single_process_fallback_when_worker_count_is_one(self):
        class FakeLLM:
            def generate_json(self, system, user, **kwargs):
                return {
                    "child_hypothesis": {
                        "kind": "relation",
                        "type": "IMPLIES",
                        "inputs": [
                            {"kind": "atomic", "name": "A2", "type": "abstract", "source": "semantic", "params": {}},
                            {"kind": "atomic", "name": "B", "type": "abstract", "source": "semantic", "params": {}},
                        ],
                        "params": {},
                    },
                    "domain_reason": "Tightening the stress condition is plausible from a crypto downside-regime perspective.",
                    "score_reason": "Tightening one atomic condition is a local change that may improve precision.",
                    "operation_score_rankings": {"replace_atomic_feature": 1, "append_atomic": 2},
                    "mutation_summary": "Applied a replace_atomic-style change in the condition side while keeping the overall relation structure.",
                } if "mutation_summary" in system else {
                    "kind": "relation",
                    "type": "IMPLIES",
                    "inputs": [
                        {"kind": "atomic", "name": "A", "type": "abstract", "source": "semantic", "params": {}},
                        {"kind": "atomic", "name": "B", "type": "abstract", "source": "semantic", "params": {}},
                    ],
                    "params": {},
                }
            def generate_text(self, system, user, **kwargs):
                return "If A then B."
        with tempfile.TemporaryDirectory() as tmp:
            config = HypoEvolveConfig()
            config.search.iterations = 1
            config.output.base_dir = tmp
            config.workers.enabled = True
            config.workers.count = 1
            config.evaluator.dataset_schema_path = str(Path(tmp) / "dataset.yaml")
            Path(config.evaluator.dataset_schema_path).write_text(
                "description: test\nindex:\n  name: close_time\n  dtype: datetime64[us]\nfiles:\n  -\n    entity: BTCUSDT\n    path: /tmp/BTCUSDT.parquet\ncolumns:\n  -\n    name: CLOSE\n",
                encoding="utf-8",
            )
            controller = HypoEvolveController(config, llm_client=FakeLLM(), evaluator=type("FakeEvaluator", (), {"evaluate": lambda self, hypothesis: {"combined_score": 0.5}})(), executor_factory=FakeExecutor)
            result = controller.run("if A then B")
            self.assertTrue(result.run_dir.exists())

    def test_worker_results_are_reflected_as_completed(self):
        class FakeLLM:
            def generate_json(self, system, user, **kwargs):
                return {
                    "child_hypothesis": {
                        "kind": "relation",
                        "type": "IMPLIES",
                        "inputs": [
                            {"kind": "atomic", "name": "A2", "type": "abstract", "source": "semantic", "params": {}},
                            {"kind": "atomic", "name": "B", "type": "abstract", "source": "semantic", "params": {}},
                        ],
                        "params": {},
                    },
                    "domain_reason": "Tightening the stress condition is plausible from a crypto downside-regime perspective.",
                    "score_reason": "Tightening one atomic condition is a local change that may improve precision.",
                    "operation_score_rankings": {"replace_atomic_feature": 1, "append_atomic": 2},
                    "mutation_summary": "Applied a replace_atomic-style change in the condition side while keeping the overall relation structure.",
                } if "mutation_summary" in system else {
                    "kind": "relation",
                    "type": "IMPLIES",
                    "inputs": [
                        {"kind": "atomic", "name": "A", "type": "abstract", "source": "semantic", "params": {}},
                        {"kind": "atomic", "name": "B", "type": "abstract", "source": "semantic", "params": {}},
                    ],
                    "params": {},
                }
            def generate_text(self, system, user, **kwargs):
                return "If A then B."
        def fake_run_worker_task(task):
            return WorkerResult(
                child_hypothesis=task.parent_hypothesis,
                metrics={"combined_score": 0.6},
                iteration=task.iteration,
                mutation_summary="Applied a change_relation_type-style local mutation.",
                parent_score=task.parent_score,
                domain_reason="Use a plausible local relation-type mutation.",
                score_reason="Use a local relation-type mutation.",
                operation_score_rankings={"change_relation_type": 1},
            )
        with tempfile.TemporaryDirectory() as tmp:
            config = HypoEvolveConfig()
            config.search.iterations = 3
            config.output.base_dir = tmp
            config.workers.enabled = True
            config.workers.count = 2
            config.evaluator.dataset_schema_path = str(Path(tmp) / "dataset.yaml")
            Path(config.evaluator.dataset_schema_path).write_text(
                "description: test\nindex:\n  name: close_time\n  dtype: datetime64[us]\nfiles:\n  -\n    entity: BTCUSDT\n    path: /tmp/BTCUSDT.parquet\ncolumns:\n  -\n    name: CLOSE\n",
                encoding="utf-8",
            )
            with patch("hypoevolve.controller.run_worker_task", side_effect=fake_run_worker_task):
                controller = HypoEvolveController(
                    config,
                    llm_client=FakeLLM(),
                    evaluator=type("FakeEvaluator", (), {"evaluate": lambda self, hypothesis: {"combined_score": 0.5}})(),
                    executor_factory=lambda max_workers=2: DelayedExecutor([2, 0, 0]),
                )
                result = controller.run("if A then B")
            trace_path = result.run_dir / "trace.jsonl"
            lines = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(lines[0]["iteration"], 0)
            self.assertEqual(lines[1]["iteration"], 2)

    def test_worker_steering_failure_is_skipped_not_fatal(self):
        class FakeLLM:
            def generate_json(self, system, user, **kwargs):
                return {
                    "kind": "relation",
                    "type": "IMPLIES",
                    "inputs": [
                        {"kind": "atomic", "name": "A", "type": "abstract", "source": "semantic", "params": {}},
                        {"kind": "atomic", "name": "B", "type": "abstract", "source": "semantic", "params": {}},
                    ],
                    "params": {},
                }

            def generate_text(self, system, user, **kwargs):
                return "If A then B."

        def fake_run_worker_task(task):
            if task.iteration == 1:
                return WorkerResult(
                    child_hypothesis={},
                    metrics={},
                    iteration=task.iteration,
                    mutation_summary="steering_failed",
                    parent_score=task.parent_score,
                    skipped_steering_error=True,
                    steering_error="Failed to steer mutation via LLM",
                )
            return WorkerResult(
                child_hypothesis=task.parent_hypothesis,
                metrics={"combined_score": 0.6},
                iteration=task.iteration,
                mutation_summary="Applied a change_relation_type-style local mutation.",
                parent_score=task.parent_score,
                domain_reason="Use a plausible local relation-type mutation.",
                score_reason="Use a local relation-type mutation.",
                operation_score_rankings={"change_relation_type": 1},
            )

        with tempfile.TemporaryDirectory() as tmp:
            config = HypoEvolveConfig()
            config.search.iterations = 2
            config.output.base_dir = tmp
            config.workers.enabled = True
            config.workers.count = 1 + 1
            config.evaluator.dataset_schema_path = str(Path(tmp) / "dataset.yaml")
            Path(config.evaluator.dataset_schema_path).write_text(
                "description: test\nindex:\n  name: close_time\n  dtype: datetime64[us]\nfiles:\n  -\n    entity: BTCUSDT\n    path: /tmp/BTCUSDT.parquet\ncolumns:\n  -\n    name: CLOSE\n",
                encoding="utf-8",
            )
            with patch("hypoevolve.controller.run_worker_task", side_effect=fake_run_worker_task):
                controller = HypoEvolveController(
                    config,
                    llm_client=FakeLLM(),
                    evaluator=type("FakeEvaluator", (), {"evaluate": lambda self, hypothesis: {"combined_score": 0.5}})(),
                    executor_factory=FakeExecutor,
                )
                result = controller.run("if A then B")

            history = json.loads(
                (result.run_dir / "score_history.json").read_text(encoding="utf-8")
            )
            self.assertEqual(history[1]["status"], "skipped_steering_error")
            self.assertEqual(history[2]["status"], "evaluated")
