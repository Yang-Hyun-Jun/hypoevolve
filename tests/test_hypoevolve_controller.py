import tempfile
import unittest
import json
from pathlib import Path
from unittest.mock import Mock, patch

from elg import AtomicNode, Hypothesis, fingerprint
from hypoevolve.archive import MAPElitesArchive
from hypoevolve.artifacts import RunArtifactRecorder
from hypoevolve.config import HypoEvolveConfig
from hypoevolve.controller import (
    HypoEvolveController,
    _build_steering_metadata,
    _record_completed_child,
    _record_skip,
)
from hypoevolve.parser import ParseError


class TestHypoEvolveController(unittest.TestCase):
    def test_bootstrap_seed_preserves_seed_archive_contract(self):
        config = HypoEvolveConfig()
        config.archive.coverage_bins = [0.1, 0.5, 0.9]
        config.archive.complexity_bins = [1, 3, 5]
        config.archive.per_cell_top_k = 4
        config.archive.parent_sampling_mode = "random"
        controller = HypoEvolveController(
            config,
            evaluator=type(
                "FakeEvaluator",
                (),
                {
                    "evaluate": lambda self, hypothesis: {"combined_score": 0.5},
                    "last_evaluation_artifacts": {"attempt": 1},
                },
            )(),
            llm_client=object(),
        )
        recorder = Mock()
        seed = Hypothesis(root=AtomicNode("A"))

        with (
            patch("hypoevolve.controller.parse_hypothesis_text", return_value=seed),
            patch("hypoevolve.controller.llm_make_hypothesis_measurable", return_value=seed),
        ):
            seed_state = controller._bootstrap_seed(
                seed_input_text="if A then B",
                recorder=recorder,
            )

        self.assertEqual(seed_state.hypothesis, seed)
        self.assertEqual(seed_state.known_fingerprints, {seed_state.archive.best.fingerprint})
        self.assertEqual(len(seed_state.known_fingerprints), 1)
        self.assertEqual(seed_state.archive.coverage_bins, [0.1, 0.5, 0.9])
        self.assertEqual(seed_state.archive.complexity_bins, [1, 3, 5])
        self.assertEqual(seed_state.archive.per_cell_top_k, 4)
        self.assertEqual(seed_state.archive.parent_sampling_mode, "random")
        recorder.record_seed.assert_called_once()

    def test_choose_mutation_uses_recent_history_tail_and_top_archive_entries(self):
        config = HypoEvolveConfig()
        config.search.random_steering_prob = 1.0
        controller = HypoEvolveController(
            config,
            evaluator=Mock(),
            llm_client=object(),
        )
        archive = MAPElitesArchive()
        for index, score in enumerate((0.9, 0.8, 0.7, 0.6), start=1):
            archive.add(
                Hypothesis(root=AtomicNode(f"A{index}")),
                {"combined_score": score},
                iteration=index,
                metadata={"source": "seed"},
            )
        parent_entry = archive.entries[0]
        recent_history = [
            {"score_delta": 0.1, "result_hypothesis": "H1"},
            {"score_delta": 0.2, "result_hypothesis": "H2"},
            {"score_delta": 0.3, "result_hypothesis": "H3"},
        ]
        fake_child = Hypothesis(root=AtomicNode("B"))
        fake_decision = type(
            "FakeDecision",
            (),
            {
                "child_hypothesis": fake_child,
                "domain_reason": "Local plausibility.",
                "score_reason": "Likely score improvement.",
                "operation_score_rankings": {"replace_atomic_feature": 1},
                "mutation_summary": "Applied a local mutation.",
            },
        )()

        with patch(
            "hypoevolve.controller.steer_mutation",
            return_value=fake_decision,
        ) as steer_mutation_mock:
            child, metadata = controller._choose_mutation(
                parent_entry=parent_entry,
                recent_history=recent_history,
                archive=archive,
            )

        self.assertEqual(child, fake_child)
        self.assertTrue(metadata["random_steering"])
        self.assertEqual(metadata["mutation_summary"], "Applied a local mutation.")
        steer_mutation_mock.assert_called_once()
        _, kwargs = steer_mutation_mock.call_args
        self.assertEqual(kwargs["recent_history"], recent_history[-2:])
        self.assertEqual(len(kwargs["top_hypotheses"]), 3)
        self.assertEqual(kwargs["top_hypotheses"][0].fingerprint, archive.entries[0].fingerprint)
        self.assertTrue(kwargs["use_random_steering"])

    def test_single_process_loop_preserves_history_and_duplicate_skip_contract(self):
        config = HypoEvolveConfig()
        config.search.iterations = 2
        seed = Hypothesis(root=AtomicNode("A"))
        child = Hypothesis(root=AtomicNode("B"))
        controller = HypoEvolveController(
            config,
            evaluator=Mock(),
            llm_client=object(),
        )
        controller.evaluator.evaluate.return_value = {"combined_score": 0.8}
        controller.evaluator.last_evaluation_artifacts = {"attempt": 1}
        archive = MAPElitesArchive()
        archive.add(seed, {"combined_score": 0.5}, iteration=0, metadata={"source": "seed"})
        parent_entry = archive.best
        known_fingerprints = {parent_entry.fingerprint}

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            recorder = RunArtifactRecorder(
                run_dir=run_dir,
                seed_input_text="if A then B",
                worker_count=1,
                workers_enabled=False,
                dataset_schema_path="dataset.yaml",
            )
            with (
                patch.object(
                    archive,
                    "sample_parent",
                    side_effect=[parent_entry, parent_entry],
                ),
                patch.object(
                    controller,
                    "_choose_mutation",
                    side_effect=[
                        (
                            child,
                            {
                                "steered": True,
                                "domain_reason": "Local plausibility.",
                                "score_reason": "Likely score improvement.",
                                "operation_score_rankings": {"replace_atomic_feature": 1},
                                "mutation_summary": "Applied a local mutation.",
                                "random_steering": False,
                            },
                        ),
                        (
                            child,
                            {
                                "steered": True,
                                "domain_reason": "Duplicate proposal.",
                                "score_reason": "Same fingerprint.",
                                "operation_score_rankings": {"replace_atomic_feature": 1},
                                "mutation_summary": "Proposed an already known hypothesis.",
                                "random_steering": False,
                            },
                        ),
                    ],
                ),
            ):
                controller._run_single_process_iterations(
                    archive=archive,
                    recorder=recorder,
                    known_fingerprints=known_fingerprints,
                )
                trace = [
                    json.loads(line)
                    for line in (run_dir / "trace.jsonl").read_text(encoding="utf-8").splitlines()
                ]

        self.assertEqual(
            known_fingerprints,
            {fingerprint(seed), fingerprint(child)},
        )
        self.assertEqual(
            [entry["status"] for entry in recorder.score_history],
            ["evaluated", "skipped_duplicate"],
        )
        self.assertEqual(len(trace), 1)
        self.assertEqual(trace[0]["child"], child.to_dict())
        self.assertAlmostEqual(trace[0]["metrics"]["combined_score"], 0.8)
        self.assertEqual(recorder.duplicate_skips_solo, 1)

    def test_build_steering_metadata_normalizes_result_shapes(self):
        class StubDecision:
            domain_reason = "Local plausibility."
            score_reason = "Likely score improvement."
            operation_score_rankings = {"replace_atomic_feature": 1}
            mutation_summary = "Applied a local mutation."

        metadata = _build_steering_metadata(
            StubDecision(),
            random_steering=True,
        )

        self.assertEqual(
            metadata,
            {
                "steered": True,
                "domain_reason": "Local plausibility.",
                "score_reason": "Likely score improvement.",
                "operation_score_rankings": {"replace_atomic_feature": 1},
                "mutation_summary": "Applied a local mutation.",
                "random_steering": True,
            },
        )

    def test_record_completed_child_reuses_common_postprocessing_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            archive = MAPElitesArchive()
            parent = Hypothesis(root=AtomicNode("A"))
            child = Hypothesis(root=AtomicNode("B"))
            archive.add(parent, {"combined_score": 0.5}, iteration=0, metadata={"source": "seed"})
            parent_entry = archive.best
            recorder = RunArtifactRecorder(
                run_dir=run_dir,
                seed_input_text="if A then B",
                worker_count=2,
                workers_enabled=True,
                dataset_schema_path="dataset.yaml",
            )

            state = _record_completed_child(
                archive=archive,
                recorder=recorder,
                iteration=1,
                parent_entry=parent_entry,
                child_hypothesis=child,
                child_metrics={"combined_score": 0.8, "precision": 0.7},
                steering_metadata={
                    "steered": True,
                    "mutation_summary": "Replaced one atomic predicate.",
                    "domain_reason": "Local change.",
                    "score_reason": "Improves precision.",
                    "operation_score_rankings": {"replace_atomic_feature": 1},
                    "random_steering": False,
                },
                evaluation_artifacts={
                    "candidate_code": "print('candidate')",
                    "wrapper_code": "print('wrapper')",
                    "attempt": 1,
                },
                worker_mode=True,
            )

            self.assertEqual(state.child_fingerprint, archive.best.fingerprint)
            self.assertEqual(
                state.history_entry["result_hypothesis"],
                child.root.name,
            )
            self.assertAlmostEqual(state.history_entry["score_delta"], 0.3)
            self.assertEqual(recorder.score_history[0]["status"], "evaluated")
            self.assertTrue(recorder.score_history[0]["best_updated"])
            self.assertTrue(recorder.score_history[0]["worker_mode"])
            self.assertEqual(
                recorder.evaluation_artifact_cache[state.child_fingerprint]["attempt"],
                1,
            )

            trace = json.loads((run_dir / "trace.jsonl").read_text(encoding="utf-8").splitlines()[0])
            checkpoint = json.loads((run_dir / "checkpoint.json").read_text(encoding="utf-8"))
            iteration_artifact = json.loads(
                (run_dir / "artifacts" / "iteration_0001.json").read_text(encoding="utf-8")
            )

            self.assertTrue(trace["metadata"]["worker_mode"])
            self.assertEqual(checkpoint["iteration"], 1)
            self.assertTrue(iteration_artifact["worker_mode"])

    def test_record_skip_reuses_common_skip_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            archive = MAPElitesArchive()
            parent = Hypothesis(root=AtomicNode("A"))
            archive.add(parent, {"combined_score": 0.5}, iteration=0, metadata={"source": "seed"})
            parent_entry = archive.best
            recorder = RunArtifactRecorder(
                run_dir=run_dir,
                seed_input_text="if A then B",
                worker_count=1,
                workers_enabled=False,
                dataset_schema_path="dataset.yaml",
            )

            _record_skip(
                archive=archive,
                recorder=recorder,
                iteration=1,
                parent_entry=parent_entry,
                worker_mode=False,
                error="Failed to steer mutation via LLM",
            )
            _record_skip(
                archive=archive,
                recorder=recorder,
                iteration=2,
                parent_entry=parent_entry,
                worker_mode=True,
                child_fingerprint="child-fp-123",
            )

            sampling_stats = archive.sampling_stats(parent_entry.fingerprint)
            self.assertEqual(sampling_stats["pulls"], 0)
            self.assertEqual(sampling_stats["total_reward"], 0.0)
            self.assertEqual(sampling_stats["last_reward"], 0.0)
            self.assertEqual(
                [entry["status"] for entry in recorder.score_history],
                ["skipped_steering_error", "skipped_duplicate"],
            )
            self.assertEqual(recorder.score_history[0]["error"], "Failed to steer mutation via LLM")
            self.assertTrue(recorder.score_history[1]["worker_mode"])
            self.assertEqual(recorder.score_history[1]["child_fingerprint"], "child-fp-123")
            self.assertEqual(recorder.duplicate_skips_worker, 1)

    def test_finalize_run_result_preserves_finalize_then_logging_and_nl_fallback(self):
        config = HypoEvolveConfig()
        controller = HypoEvolveController(
            config,
            evaluator=Mock(),
            llm_client=object(),
        )
        archive = MAPElitesArchive()
        best = archive.add(
            Hypothesis(root=AtomicNode("A")),
            {"combined_score": 0.5},
            iteration=0,
            metadata={"source": "seed"},
        )
        recorder = Mock(duplicate_skips_solo=1, duplicate_skips_worker=2)
        call_order: list[tuple[str, str | int | None]] = []

        def finalize_side_effect(**kwargs):
            call_order.append(("finalize", kwargs["known_fingerprint_count"]))
            self.assertEqual(kwargs["best_hypothesis_nl"], "A")
            return Path("/tmp/final-report.md")

        def log_side_effect(event_name, **kwargs):
            if event_name in {"run.duplicate_summary", "run.done"}:
                call_order.append((event_name, None))

        recorder.finalize.side_effect = finalize_side_effect

        with patch(
            "hypoevolve.controller.llm_hypothesis_to_natural_language",
            side_effect=ParseError("nl fail"),
        ), patch(
            "hypoevolve.controller.log_info_event",
            side_effect=log_side_effect,
        ):
            result = controller._finalize_run_result(
                archive=archive,
                recorder=recorder,
                known_fingerprint_count=4,
                run_dir=Path("/tmp/run-1"),
                seed_hypothesis=best.hypothesis,
                seed_input_text="if A then B",
                seed_generated=False,
            )

        self.assertEqual(
            call_order,
            [("finalize", 4), ("run.duplicate_summary", None), ("run.done", None)],
        )
        self.assertEqual(result.report_path, Path("/tmp/final-report.md"))
        self.assertEqual(result.best_hypothesis.root.name, "A")
        self.assertEqual(result.best_metrics["combined_score"], 0.5)

    def test_prepare_run_preserves_setup_contract(self):
        config = HypoEvolveConfig()
        config.search.iterations = 3
        config.workers.count = 2
        controller = HypoEvolveController(
            config,
            evaluator=Mock(),
            llm_client=object(),
        )
        recorder = Mock()
        call_order: list[str] = []

        with (
            patch.object(
                controller,
                "_resolve_seed_input_text",
                return_value=("if A then B", False),
            ),
            patch("hypoevolve.controller.create_run_dir", return_value=Path("/tmp/run-1")),
            patch("hypoevolve.controller.RunArtifactRecorder", return_value=recorder) as recorder_cls,
            patch("hypoevolve.controller.configure_logger", side_effect=lambda *args, **kwargs: call_order.append("configure_logger")),
            patch(
                "hypoevolve.controller.log_info_event",
                side_effect=lambda event_name, **kwargs: call_order.append(event_name),
            ),
        ):
            prepared = controller._prepare_run("if A then B")

        self.assertEqual(prepared.seed_input_text, "if A then B")
        self.assertFalse(prepared.seed_generated)
        self.assertEqual(prepared.run_dir, Path("/tmp/run-1"))
        self.assertIs(prepared.recorder, recorder)
        recorder_cls.assert_called_once_with(
            run_dir=Path("/tmp/run-1"),
            seed_input_text="if A then B",
            worker_count=2,
            workers_enabled=config.workers.enabled,
            dataset_schema_path=config.evaluator.dataset_schema_path,
            top_k_code_artifacts=config.output.top_k_evaluator_code_artifacts,
        )
        self.assertEqual(call_order, ["configure_logger", "run.start"])

    def test_execute_search_branch_preserves_known_fingerprint_count_in_single_process_mode(self):
        config = HypoEvolveConfig()
        config.workers.enabled = False
        controller = HypoEvolveController(
            config,
            evaluator=Mock(),
            llm_client=object(),
        )
        archive = MAPElitesArchive()
        seed = Hypothesis(root=AtomicNode("A"))
        archive.add(seed, {"combined_score": 0.5}, iteration=0, metadata={"source": "seed"})
        seed_state = type(
            "SeedState",
            (),
            {
                "archive": archive,
                "known_fingerprints": {fingerprint(seed)},
            },
        )()
        recorder = Mock()

        with patch.object(controller, "_run_single_process_iterations") as single_process_mock:
            known_fingerprint_count = controller._execute_search_branch(
                seed_state=seed_state,
                recorder=recorder,
            )

        single_process_mock.assert_called_once_with(
            archive=archive,
            recorder=recorder,
            known_fingerprints=seed_state.known_fingerprints,
        )
        self.assertEqual(known_fingerprint_count, 1)

    def test_execute_search_branch_uses_worker_returned_known_fingerprint_count(self):
        config = HypoEvolveConfig()
        config.workers.enabled = True
        config.workers.count = 2
        controller = HypoEvolveController(
            config,
            evaluator=Mock(),
            llm_client=object(),
        )
        archive = MAPElitesArchive()
        seed = Hypothesis(root=AtomicNode("A"))
        archive.add(seed, {"combined_score": 0.5}, iteration=0, metadata={"source": "seed"})
        seed_state = type(
            "SeedState",
            (),
            {
                "archive": archive,
                "known_fingerprints": {fingerprint(seed)},
            },
        )()
        recorder = Mock()

        with patch.object(controller, "_run_worker_iterations", return_value=4) as worker_mock:
            known_fingerprint_count = controller._execute_search_branch(
                seed_state=seed_state,
                recorder=recorder,
            )

        worker_mock.assert_called_once_with(
            archive=archive,
            recorder=recorder,
            known_fingerprints=seed_state.known_fingerprints,
        )
        self.assertEqual(known_fingerprint_count, 4)

    def test_run_writes_expected_artifacts_and_report(self):
        class FakeLLM:
            def generate_json(self, system, user, **kwargs):
                if "mutation_summary" in system:
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
                    }
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
        with tempfile.TemporaryDirectory() as tmp:
            config = HypoEvolveConfig()
            config.search.iterations = 2
            config.output.base_dir = tmp
            config.evaluator.dataset_schema_path = str(Path(tmp) / "dataset.yaml")
            Path(config.evaluator.dataset_schema_path).write_text(
                "description: test\nindex:\n  name: close_time\n  dtype: datetime64[us]\nfiles:\n  -\n    entity: BTCUSDT\n    path: /tmp/BTCUSDT.parquet\ncolumns:\n  -\n    name: CLOSE\n",
                encoding="utf-8",
            )
            fake_evaluator = type(
                "FakeEvaluator",
                (),
                {"evaluate": lambda self, hypothesis: {"combined_score": 0.5}},
            )()
            controller = HypoEvolveController(config, evaluator=fake_evaluator, llm_client=FakeLLM())
            result = controller.run('if A then B')
            self.assertTrue(result.run_dir.exists())
            self.assertTrue((result.run_dir / 'trace.jsonl').exists())
            self.assertTrue((result.run_dir / 'checkpoint.json').exists())
            self.assertTrue((result.run_dir / 'best.json').exists())
            self.assertTrue((result.run_dir / 'run_summary.json').exists())
            self.assertTrue((result.run_dir / 'score_history.json').exists())
            self.assertTrue((result.run_dir / 'report' / 'report.md').exists())
            self.assertTrue((result.run_dir / 'report' / 'assets' / 'score_progression.svg').exists())
            report_text = (result.run_dir / 'report' / 'report.md').read_text(encoding='utf-8')
            self.assertIn('Best ELG', report_text)
            self.assertIn('If A then B.', report_text)
            self.assertIn('combined_score', result.best_metrics)

    def test_run_records_mutation_steering_metadata_in_trace_when_enabled(self):
        class FakeLLM:
            def __init__(self):
                self.calls = 0

            def generate_json(self, system, user, **kwargs):
                self.calls += 1
                if "mutation_summary" in system:
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
                    }
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

        with tempfile.TemporaryDirectory() as tmp:
            config = HypoEvolveConfig()
            config.search.iterations = 1
            config.output.base_dir = tmp
            config.evaluator.dataset_schema_path = str(Path(tmp) / "dataset.yaml")
            Path(config.evaluator.dataset_schema_path).write_text(
                "description: test\nindex:\n  name: close_time\n  dtype: datetime64[us]\nfiles:\n  -\n    entity: BTCUSDT\n    path: /tmp/BTCUSDT.parquet\ncolumns:\n  -\n    name: CLOSE\n",
                encoding="utf-8",
            )
            fake_evaluator = type(
                "FakeEvaluator",
                (),
                {"evaluate": lambda self, hypothesis: {"combined_score": 0.5}},
            )()
            controller = HypoEvolveController(config, evaluator=fake_evaluator, llm_client=FakeLLM())
            result = controller.run("if A then B")
            trace = [
                json.loads(line)
                for line in (result.run_dir / "trace.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertTrue(trace[1]["metadata"]["steered"])
            self.assertIn("replace_atomic-style", trace[1]["metadata"]["mutation_summary"])

    def test_run_records_skipped_iteration_in_score_history_when_steering_errors(self):
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

        with tempfile.TemporaryDirectory() as tmp:
            config = HypoEvolveConfig()
            config.search.iterations = 1
            config.output.base_dir = tmp
            config.evaluator.dataset_schema_path = str(Path(tmp) / "dataset.yaml")
            Path(config.evaluator.dataset_schema_path).write_text(
                "description: test\nindex:\n  name: close_time\n  dtype: datetime64[us]\nfiles:\n  -\n    entity: BTCUSDT\n    path: /tmp/BTCUSDT.parquet\ncolumns:\n  -\n    name: CLOSE\n",
                encoding="utf-8",
            )
            fake_evaluator = type(
                "FakeEvaluator",
                (),
                {"evaluate": lambda self, hypothesis: {"combined_score": 0.5}},
            )()
            controller = HypoEvolveController(
                config, evaluator=fake_evaluator, llm_client=FakeLLM()
            )
            with patch(
                "hypoevolve.controller.steer_mutation",
                side_effect=ParseError("Failed to steer mutation via LLM"),
            ):
                result = controller.run("if A then B")

            history = json.loads(
                (result.run_dir / "score_history.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(history), 2)
            self.assertEqual(history[1]["status"], "skipped_steering_error")
            self.assertIn("Failed to steer mutation via LLM", history[1]["error"])

    def test_run_skips_duplicate_child_before_child_evaluation(self):
        config = HypoEvolveConfig()
        config.search.iterations = 1

        seed = Hypothesis(root=AtomicNode("A"))
        fake_decision = type(
            "FakeDecision",
            (),
            {
                "child_hypothesis": Hypothesis(root=AtomicNode("A")),
                "domain_reason": "Duplicate child.",
                "score_reason": "Should skip repeated eval.",
                "operation_score_rankings": {"replace_atomic_feature": 1},
                "mutation_summary": "Proposed an already known hypothesis.",
            },
        )()

        call_count = 0

        def fake_evaluate(hypothesis, evaluator):
            nonlocal call_count
            call_count += 1
            if call_count > 1:
                raise AssertionError("duplicate child should skip evaluation")
            return {"combined_score": 0.5}

        with tempfile.TemporaryDirectory() as tmp, patch(
            "hypoevolve.controller.parse_hypothesis_text",
            return_value=seed,
        ), patch(
            "hypoevolve.controller.llm_make_hypothesis_measurable",
            return_value=seed,
        ), patch(
            "hypoevolve.controller.llm_hypothesis_to_natural_language",
            return_value="A",
        ), patch(
            "hypoevolve.controller.steer_mutation",
            return_value=fake_decision,
        ):
            config.output.base_dir = tmp
            fake_evaluator = type(
                "FakeEvaluator",
                (),
                {
                    "evaluate": lambda self, hypothesis: fake_evaluate(hypothesis, self),
                    "last_evaluation_artifacts": {},
                },
            )()
            controller = HypoEvolveController(
                config,
                evaluator=fake_evaluator,
                llm_client=object(),
            )
            result = controller.run("if A then B")

            trace_lines = (result.run_dir / "trace.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
            score_history = json.loads(
                (result.run_dir / "score_history.json").read_text(encoding="utf-8")
            )
            log_text = (result.run_dir / "hypoevolve.log").read_text(
                encoding="utf-8"
            )

        self.assertEqual(call_count, 1)
        self.assertEqual(len(trace_lines), 1)
        self.assertEqual([entry["status"] for entry in score_history], ["seed", "skipped_duplicate"])
        self.assertNotIn("score", score_history[1])
        self.assertFalse((result.run_dir / "artifacts" / "iteration_0001.json").exists())
        self.assertIn("event=run.duplicate_summary total_skips=1", log_text)

    def test_run_generates_initial_hypothesis_when_seed_input_is_missing(self):
        config = HypoEvolveConfig()
        config.search.iterations = 1

        seed = Hypothesis(root=AtomicNode("A"))
        fake_tree_result = type(
            "FakeTreeResult",
            (),
            {"hypothesis": "Generated seed hypothesis."},
        )()

        with tempfile.TemporaryDirectory() as tmp, patch(
            "hypoevolve.controller.generate_random_tree_pair_hypothesis",
            return_value=fake_tree_result,
        ) as generate_seed_mock, patch(
            "hypoevolve.controller.parse_hypothesis_text",
            return_value=seed,
        ) as parse_mock, patch(
            "hypoevolve.controller.llm_make_hypothesis_measurable",
            return_value=seed,
        ), patch(
            "hypoevolve.controller.llm_hypothesis_to_natural_language",
            return_value="A",
        ), patch(
            "hypoevolve.controller.steer_mutation",
            return_value=type(
                "FakeDecision",
                (),
                {
                    "child_hypothesis": Hypothesis(root=AtomicNode("B")),
                    "domain_reason": "",
                    "score_reason": "",
                    "operation_score_rankings": {"replace_atomic_feature": 1},
                    "mutation_summary": "Applied a replace_atomic-style local mutation.",
                },
            )(),
        ):
            config.output.base_dir = tmp
            fake_evaluator = type(
                "FakeEvaluator",
                (),
                {"evaluate": lambda self, hypothesis: {"combined_score": 0.5}, "last_evaluation_artifacts": {}},
            )()
            controller = HypoEvolveController(
                config,
                evaluator=fake_evaluator,
                llm_client=object(),
            )
            result = controller.run()

        generate_seed_mock.assert_called_once()
        self.assertEqual(
            generate_seed_mock.call_args.kwargs["dataset_schema_path"],
            config.evaluator.dataset_schema_path,
        )
        parse_mock.assert_called_once_with(
            "Generated seed hypothesis.",
            llm=controller.llm_client,
            retries=config.parser.retries,
        )
        self.assertTrue(result.seed_generated)
        self.assertEqual(result.seed_input_text, "Generated seed hypothesis.")
        self.assertIsNotNone(result.seed_hypothesis)

    def test_run_seed_bootstrap_uses_configured_archive_bins_and_sampling_mode(self):
        config = HypoEvolveConfig()
        config.archive.coverage_bins = [0.1, 0.5, 0.9]
        config.archive.complexity_bins = [1, 3, 5]
        config.archive.per_cell_top_k = 4
        config.archive.parent_sampling_mode = "random"
        controller = HypoEvolveController(
            config,
            evaluator=type("FakeEvaluator", (), {"evaluate": lambda self, hypothesis: {"combined_score": 0.5}, "last_evaluation_artifacts": {}})(),
            llm_client=object(),
        )
        with tempfile.TemporaryDirectory() as tmp:
            config.output.base_dir = tmp
            with (
                patch("hypoevolve.controller.parse_hypothesis_text", return_value=Hypothesis(root=AtomicNode("A"))),
                patch("hypoevolve.controller.llm_make_hypothesis_measurable", return_value=Hypothesis(root=AtomicNode("A"))),
                patch("hypoevolve.controller.RunArtifactRecorder.record_seed") as record_seed,
                patch("hypoevolve.controller.llm_hypothesis_to_natural_language", return_value="A"),
                patch("hypoevolve.controller.RunArtifactRecorder.finalize", return_value=Path(tmp) / "report.md"),
            ):
                controller.run("if A then B")

        archive = record_seed.call_args.kwargs["archive"]
        self.assertEqual(archive.coverage_bins, [0.1, 0.5, 0.9])
        self.assertEqual(archive.complexity_bins, [1, 3, 5])
        self.assertEqual(archive.per_cell_top_k, 4)
        self.assertEqual(archive.parent_sampling_mode, "random")

    def test_archive_best_fields_use_stable_empty_defaults_before_first_archive_entry(self):
        controller = HypoEvolveController(
            HypoEvolveConfig(),
            evaluator=type("FakeEvaluator", (), {"evaluate": lambda self, hypothesis: {}})(),
            llm_client=object(),
        )
        archive = MAPElitesArchive()

        self.assertEqual(archive.best.score if archive.best else 0.0, 0.0)
        self.assertIsNone(archive.best.cell if archive.best else None)

        archive.add(Hypothesis(root=AtomicNode("A")), {"combined_score": 0.7})
        self.assertEqual(archive.best.score if archive.best else 0.0, 0.7)
        self.assertEqual(archive.best.cell if archive.best else None, archive.best.cell)
