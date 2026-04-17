import tempfile
import unittest
import json
from concurrent.futures import Future
from pathlib import Path
from threading import Timer
from unittest.mock import Mock, patch

from elg import AtomicNode, Hypothesis, fingerprint
from hypoevolve.archive import MAPElitesArchive
from hypoevolve.artifacts import RunArtifactRecorder
from hypoevolve.config import HypoEvolveConfig
from hypoevolve.controller import HypoEvolveController
from hypoevolve.workers import WorkerResult


class ImmediateFuture(Future):
    def __init__(self, result):
        super().__init__()
        self.set_result(result)


class FakeExecutor:
    def __init__(self, max_workers=1):
        self.max_workers = max_workers

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def submit(self, fn, task):
        return ImmediateFuture(fn(task))


class DelayedFuture(Future):
    def __init__(self, result, polls_before_done: int):
        super().__init__()
        delay_sec = max(0.0, 0.01 * polls_before_done)
        self._timer = Timer(delay_sec, lambda: self.set_result(result))
        self._timer.start()


class DelayedExecutor(FakeExecutor):
    def __init__(self, delays):
        super().__init__(max_workers=len(delays))
        self.delays = list(delays)

    def submit(self, fn, task):
        delay = self.delays.pop(0) if self.delays else 0
        return DelayedFuture(fn(task), delay)


class TestHypoEvolveControllerWorkers(unittest.TestCase):
    def test_run_builds_followup_worker_task_from_recent_history_and_known_fingerprints(self):
        config = HypoEvolveConfig()
        config.search.iterations = 3
        config.search.random_steering_prob = 0.0
        config.workers.enabled = True
        config.workers.count = 2
        config.evaluator.parameters = {"temperature": 0.2}

        seed = Hypothesis(root=AtomicNode("A"))
        seen_tasks: list[object] = []

        def fake_run_worker_task(task):
            seen_tasks.append(task)
            child = Hypothesis(root=AtomicNode(f"C{task.iteration}"))
            return WorkerResult(
                child_hypothesis=child.to_dict(),
                metrics={"combined_score": 0.5 + (0.1 * task.iteration)},
                iteration=task.iteration,
                mutation_summary=f"mutation {task.iteration}",
                parent_score=task.parent_score,
                domain_reason=f"domain {task.iteration}",
                score_reason=f"score {task.iteration}",
                operation_score_rankings={"replace_atomic_feature": 1},
                random_steering=task.use_random_steering,
                child_fingerprint=fingerprint(child),
            )

        with tempfile.TemporaryDirectory() as tmp:
            config.output.base_dir = tmp
            controller = HypoEvolveController(
                config,
                llm_client=object(),
                evaluator=Mock(),
                executor_factory=lambda max_workers: DelayedExecutor([0, 2]),
            )
            controller.evaluator.evaluate.return_value = {"combined_score": 0.5}
            controller.evaluator.last_evaluation_artifacts = {}

            with (
                patch("hypoevolve.controller.parse_hypothesis_text", return_value=seed),
                patch("hypoevolve.controller.llm_make_hypothesis_measurable", return_value=seed),
                patch("hypoevolve.controller.llm_hypothesis_to_natural_language", return_value="A"),
                patch("hypoevolve.controller.run_worker_task", side_effect=fake_run_worker_task),
            ):
                controller.run("if A then B")

        self.assertEqual(len(seen_tasks), 3)
        self.assertEqual(seen_tasks[0].recent_history, [])
        self.assertEqual(seen_tasks[0].seen_fingerprints, [fingerprint(seed)])
        self.assertFalse(seen_tasks[0].use_random_steering)
        self.assertEqual(seen_tasks[0].evaluator_parameters, {"temperature": 0.2})

        self.assertEqual(len(seen_tasks[2].recent_history), 1)
        self.assertAlmostEqual(seen_tasks[2].recent_history[0]["score_delta"], 0.1)
        self.assertEqual(seen_tasks[2].recent_history[0]["result_hypothesis"], "C1")
        self.assertTrue(seen_tasks[2].recent_history[0]["steered"])
        self.assertEqual(seen_tasks[2].recent_history[0]["domain_reason"], "domain 1")
        self.assertEqual(seen_tasks[2].recent_history[0]["score_reason"], "score 1")
        self.assertEqual(
            seen_tasks[2].recent_history[0]["operation_score_rankings"],
            {"replace_atomic_feature": 1},
        )
        self.assertEqual(seen_tasks[2].recent_history[0]["mutation_summary"], "mutation 1")
        self.assertFalse(seen_tasks[2].recent_history[0]["random_steering"])
        self.assertEqual(
            seen_tasks[2].seen_fingerprints,
            sorted([fingerprint(seed), fingerprint(Hypothesis(root=AtomicNode("C1")))]),
        )
        self.assertEqual(
            seen_tasks[2].top_hypotheses[0]["fingerprint"],
            fingerprint(Hypothesis(root=AtomicNode("C1"))),
        )

    def test_worker_loop_preserves_followup_task_context_and_known_fingerprint_count(self):
        config = HypoEvolveConfig()
        config.search.iterations = 3
        config.search.random_steering_prob = 0.0
        config.workers.enabled = True
        config.workers.count = 2
        config.evaluator.parameters = {"temperature": 0.2}

        seed = Hypothesis(root=AtomicNode("A"))
        seen_tasks: list[object] = []

        def fake_run_worker_task(task):
            seen_tasks.append(task)
            child = Hypothesis(root=AtomicNode(f"C{task.iteration}"))
            return WorkerResult(
                child_hypothesis=child.to_dict(),
                metrics={"combined_score": 0.5 + (0.1 * task.iteration)},
                iteration=task.iteration,
                mutation_summary=f"mutation {task.iteration}",
                parent_score=task.parent_score,
                domain_reason=f"domain {task.iteration}",
                score_reason=f"score {task.iteration}",
                operation_score_rankings={"replace_atomic_feature": 1},
                random_steering=task.use_random_steering,
                child_fingerprint=fingerprint(child),
            )

        with tempfile.TemporaryDirectory() as tmp:
            controller = HypoEvolveController(
                config,
                llm_client=object(),
                evaluator=Mock(),
                executor_factory=lambda max_workers: DelayedExecutor([0, 2]),
            )
            archive = MAPElitesArchive()
            archive.add(seed, {"combined_score": 0.5}, iteration=0, metadata={"source": "seed"})
            known_fingerprints = {fingerprint(seed)}
            recorder = RunArtifactRecorder(
                run_dir=Path(tmp),
                seed_input_text="if A then B",
                worker_count=2,
                workers_enabled=True,
                dataset_schema_path="dataset.yaml",
            )

            with patch("hypoevolve.controller.run_worker_task", side_effect=fake_run_worker_task):
                known_fingerprint_count = controller._run_worker_iterations(
                    archive=archive,
                    recorder=recorder,
                    known_fingerprints=known_fingerprints,
                )

        self.assertEqual(known_fingerprint_count, 4)
        self.assertEqual(len(seen_tasks), 3)
        self.assertEqual(seen_tasks[0].recent_history, [])
        self.assertEqual(seen_tasks[0].seen_fingerprints, [fingerprint(seed)])
        self.assertEqual(len(seen_tasks[2].recent_history), 1)
        self.assertAlmostEqual(seen_tasks[2].recent_history[0]["score_delta"], 0.1)
        self.assertEqual(seen_tasks[2].recent_history[0]["result_hypothesis"], "C1")
        self.assertEqual(
            known_fingerprints,
            {
                fingerprint(seed),
                fingerprint(Hypothesis(root=AtomicNode("C1"))),
                fingerprint(Hypothesis(root=AtomicNode("C2"))),
                fingerprint(Hypothesis(root=AtomicNode("C3"))),
            },
        )

    def test_run_uses_single_process_path_in_mocked_run_when_worker_count_is_one(self):
        config = HypoEvolveConfig()
        config.workers.enabled = True
        config.workers.count = 1
        controller = HypoEvolveController(
            config,
            llm_client=Mock(),
            evaluator=Mock(),
            executor_factory=FakeExecutor,
        )
        controller.evaluator.evaluate.return_value = {"combined_score": 0.5}
        controller.evaluator.last_evaluation_artifacts = {}
        archive = MAPElitesArchive()
        archive.add(Hypothesis(root=AtomicNode("A")), {"combined_score": 0.5})
        recorder = Mock(duplicate_skips_solo=0, duplicate_skips_worker=0)
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()
            recorder.finalize.return_value = run_dir / "report.md"
            with (
                patch("hypoevolve.controller.create_run_dir", return_value=run_dir),
                patch("hypoevolve.controller.RunArtifactRecorder", return_value=recorder),
                patch("hypoevolve.controller.configure_logger"),
                patch("hypoevolve.controller.log_info_event"),
                patch("hypoevolve.controller.parse_hypothesis_text", return_value=Hypothesis(root=AtomicNode("A"))),
                patch("hypoevolve.controller.llm_make_hypothesis_measurable", return_value=Hypothesis(root=AtomicNode("A"))),
                patch("hypoevolve.controller.run_worker_task") as run_with_workers,
                patch("hypoevolve.controller.llm_hypothesis_to_natural_language", return_value="A"),
            ):
                result = controller.run("if A then B")

        self.assertEqual(result.run_dir, run_dir)
        self.assertEqual(result.seed_hypothesis.root.name, "A")
        self.assertEqual(result.best_hypothesis.root.name, "A")
        self.assertEqual(result.best_metrics["combined_score"], 0.5)
        run_with_workers.assert_not_called()

    def test_run_calls_worker_path_in_mocked_run_when_workers_enabled(self):
        config = HypoEvolveConfig()
        config.workers.enabled = True
        config.workers.count = 2
        controller = HypoEvolveController(
            config,
            llm_client=Mock(),
            evaluator=Mock(),
            executor_factory=FakeExecutor,
        )
        controller.evaluator.evaluate.return_value = {"combined_score": 0.5}
        controller.evaluator.last_evaluation_artifacts = {}
        archive = MAPElitesArchive()
        archive.add(Hypothesis(root=AtomicNode("A")), {"combined_score": 0.5})
        recorder = Mock(duplicate_skips_solo=0, duplicate_skips_worker=0)
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()
            recorder.finalize.return_value = run_dir / "report.md"
            with (
                patch("hypoevolve.controller.create_run_dir", return_value=run_dir),
                patch("hypoevolve.controller.RunArtifactRecorder", return_value=recorder),
                patch("hypoevolve.controller.configure_logger"),
                patch("hypoevolve.controller.log_info_event"),
                patch("hypoevolve.controller.parse_hypothesis_text", return_value=Hypothesis(root=AtomicNode("A"))),
                patch("hypoevolve.controller.llm_make_hypothesis_measurable", return_value=Hypothesis(root=AtomicNode("A"))),
                patch("hypoevolve.controller.run_worker_task", return_value=WorkerResult(
                    child_hypothesis=Hypothesis(root=AtomicNode("A")).to_dict(),
                    metrics={"combined_score": 0.5},
                    iteration=1,
                    mutation_summary="worker",
                    parent_score=0.5,
                )) as run_with_workers,
                patch("hypoevolve.controller.llm_hypothesis_to_natural_language", return_value="A"),
            ):
                result = controller.run("if A then B")

        self.assertEqual(result.run_dir, run_dir)
        self.assertEqual(result.seed_hypothesis.root.name, "A")
        self.assertEqual(result.best_hypothesis.root.name, "A")
        self.assertEqual(result.best_metrics["combined_score"], 0.5)
        run_with_workers.assert_called()

    def test_run_with_workers_passes_known_fingerprint_count_through_finalize_and_completion_log(self):
        config = HypoEvolveConfig()
        config.workers.enabled = True
        config.workers.count = 2
        controller = HypoEvolveController(
            config,
            llm_client=Mock(),
            evaluator=Mock(),
            executor_factory=FakeExecutor,
        )
        controller.evaluator.evaluate.return_value = {"combined_score": 0.5}
        controller.evaluator.last_evaluation_artifacts = {}
        archive = MAPElitesArchive()
        archive.add(Hypothesis(root=AtomicNode("A")), {"combined_score": 0.5})
        recorder = Mock(duplicate_skips_solo=0, duplicate_skips_worker=0)
        call_order: list[tuple[str, int | None]] = []

        def finalize_side_effect(*, archive, known_fingerprint_count, **kwargs):
            call_order.append(("finalize", known_fingerprint_count))
            return run_dir / "report.md"

        def log_side_effect(event_name, **kwargs):
            if event_name == "run.done":
                call_order.append(("log_complete", 1))

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()
            with (
                patch("hypoevolve.controller.create_run_dir", return_value=run_dir),
                patch("hypoevolve.controller.RunArtifactRecorder", return_value=recorder),
                patch("hypoevolve.controller.configure_logger"),
                patch("hypoevolve.controller.log_info_event", side_effect=log_side_effect) as log_info_event,
                patch("hypoevolve.controller.parse_hypothesis_text", return_value=Hypothesis(root=AtomicNode("A"))),
                patch("hypoevolve.controller.llm_make_hypothesis_measurable", return_value=Hypothesis(root=AtomicNode("A"))),
                patch("hypoevolve.controller.run_worker_task", return_value=WorkerResult(
                    child_hypothesis=Hypothesis(root=AtomicNode("A")).to_dict(),
                    metrics={"combined_score": 0.5},
                    iteration=1,
                    mutation_summary="worker",
                    parent_score=0.5,
                )),
                patch.object(recorder, "finalize", side_effect=finalize_side_effect) as finalize_run,
                patch("hypoevolve.controller.llm_hypothesis_to_natural_language", return_value="A"),
            ):
                result = controller.run("if A then B")

        self.assertEqual(result.run_dir, run_dir)
        self.assertEqual(result.seed_hypothesis.root.name, "A")
        self.assertEqual(result.best_hypothesis.root.name, "A")
        self.assertEqual(result.best_metrics["combined_score"], 0.5)
        self.assertEqual(result.report_path, run_dir / "report.md")
        self.assertEqual(
            call_order,
            [("finalize", 1), ("log_complete", 1)],
        )
        finalize_run.assert_called_once()
        self.assertTrue(any(call.args[0] == "run.done" for call in log_info_event.call_args_list))

    def test_run_writes_expected_artifacts_in_real_run_when_workers_enabled(self):
        class FakeLLM:
            def generate_json(self, system, user, **kwargs):
                return {
                    "child_hypothesis": {
                        "kind": "relation",
                        "name": "IMPLIES",
                        "inputs": [
                            {"kind": "atomic", "name": "A2"},
                            {"kind": "atomic", "name": "B"},
                        ],
                    },
                    "domain_reason": "Tightening the stress condition is plausible from a crypto downside-regime perspective.",
                    "score_reason": "Tightening one atomic condition is a local change that may improve precision.",
                    "operation_score_rankings": {"replace_atomic_feature": 1, "append_atomic": 2},
                    "mutation_summary": "Applied a replace_atomic-style change in the condition side while keeping the overall relation structure.",
                } if "mutation_summary" in system else {
                    "kind": "relation",
                    "name": "IMPLIES",
                    "inputs": [
                        {"kind": "atomic", "name": "A"},
                        {"kind": "atomic", "name": "B"},
                    ],
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

    def test_run_uses_single_process_path_in_real_run_when_workers_disabled(self):
        class FakeLLM:
            def generate_json(self, system, user, **kwargs):
                return {
                    "child_hypothesis": {
                        "kind": "relation",
                        "name": "IMPLIES",
                        "inputs": [
                            {"kind": "atomic", "name": "A2"},
                            {"kind": "atomic", "name": "B"},
                        ],
                    },
                    "domain_reason": "Tightening the stress condition is plausible from a crypto downside-regime perspective.",
                    "score_reason": "Tightening one atomic condition is a local change that may improve precision.",
                    "operation_score_rankings": {"replace_atomic_feature": 1, "append_atomic": 2},
                    "mutation_summary": "Applied a replace_atomic-style change in the condition side while keeping the overall relation structure.",
                } if "mutation_summary" in system else {
                    "kind": "relation",
                    "name": "IMPLIES",
                    "inputs": [
                        {"kind": "atomic", "name": "A"},
                        {"kind": "atomic", "name": "B"},
                    ],
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

    def test_run_uses_single_process_path_in_real_run_when_worker_count_is_one(self):
        class FakeLLM:
            def generate_json(self, system, user, **kwargs):
                return {
                    "child_hypothesis": {
                        "kind": "relation",
                        "name": "IMPLIES",
                        "inputs": [
                            {"kind": "atomic", "name": "A2"},
                            {"kind": "atomic", "name": "B"},
                        ],
                    },
                    "domain_reason": "Tightening the stress condition is plausible from a crypto downside-regime perspective.",
                    "score_reason": "Tightening one atomic condition is a local change that may improve precision.",
                    "operation_score_rankings": {"replace_atomic_feature": 1, "append_atomic": 2},
                    "mutation_summary": "Applied a replace_atomic-style change in the condition side while keeping the overall relation structure.",
                } if "mutation_summary" in system else {
                    "kind": "relation",
                    "name": "IMPLIES",
                    "inputs": [
                        {"kind": "atomic", "name": "A"},
                        {"kind": "atomic", "name": "B"},
                    ],
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

    def test_run_with_workers_records_completed_results_in_trace_jsonl(self):
        class FakeLLM:
            def generate_json(self, system, user, **kwargs):
                return {
                    "child_hypothesis": {
                        "kind": "relation",
                        "name": "IMPLIES",
                        "inputs": [
                            {"kind": "atomic", "name": "A2"},
                            {"kind": "atomic", "name": "B"},
                        ],
                    },
                    "domain_reason": "Tightening the stress condition is plausible from a crypto downside-regime perspective.",
                    "score_reason": "Tightening one atomic condition is a local change that may improve precision.",
                    "operation_score_rankings": {"replace_atomic_feature": 1, "append_atomic": 2},
                    "mutation_summary": "Applied a replace_atomic-style change in the condition side while keeping the overall relation structure.",
                } if "mutation_summary" in system else {
                    "kind": "relation",
                    "name": "IMPLIES",
                    "inputs": [
                        {"kind": "atomic", "name": "A"},
                        {"kind": "atomic", "name": "B"},
                    ],
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

    def test_run_with_workers_records_skipped_steering_error_in_score_history(self):
        class FakeLLM:
            def generate_json(self, system, user, **kwargs):
                return {
                    "kind": "relation",
                    "name": "IMPLIES",
                    "inputs": [
                        {"kind": "atomic", "name": "A"},
                        {"kind": "atomic", "name": "B"},
                    ],
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
            self.assertNotIn("score", history[1])
