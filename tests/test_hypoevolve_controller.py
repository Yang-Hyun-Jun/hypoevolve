import tempfile
import unittest
import json
from pathlib import Path
from unittest.mock import patch

from elg import AtomicNode, Hypothesis, fingerprint
from hypoevolve.archive import MAPElitesArchive
from hypoevolve.artifacts import RunArtifactRecorder
from hypoevolve.config import HypoEvolveConfig
from hypoevolve.controller import HypoEvolveController


class TestHypoEvolveController(unittest.TestCase):
    def test_happy_path_run_completes_and_writes_outputs(self):
        class FakeLLM:
            def generate_json(self, system, user, **kwargs):
                if "mutation_summary" in system:
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
                    }
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

    def test_run_uses_mutation_steering_when_enabled(self):
        class FakeLLM:
            def __init__(self):
                self.calls = 0

            def generate_json(self, system, user, **kwargs):
                self.calls += 1
                if "mutation_summary" in system:
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
                    }
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

    def test_choose_mutation_reuses_cached_parent_hypothesis_nl(self):
        config = HypoEvolveConfig()
        controller = HypoEvolveController(
            config,
            evaluator=type("FakeEvaluator", (), {"evaluate": lambda self, hypothesis: {}})(),
            llm_client=object(),
        )
        archive = MAPElitesArchive()
        parent = Hypothesis(root=AtomicNode("A"))
        entry = archive.add(
            parent,
            {"combined_score": 0.1},
            metadata={"hypothesis_nl": "Cached A."},
        )
        fake_decision = type(
            "FakeDecision",
            (),
            {
                "child_hypothesis": Hypothesis(root=AtomicNode("B")),
                "domain_reason": "Cached domain rationale.",
                "score_reason": "Use cached score rationale.",
                "operation_score_rankings": {"replace_atomic_feature": 1},
                "mutation_summary": "Applied a replace_atomic-style local mutation.",
            },
        )()

        with patch(
            "hypoevolve.controller.llm_hypothesis_to_natural_language",
            side_effect=AssertionError("should reuse cached NL"),
        ), patch(
            "hypoevolve.controller.steer_mutation",
            return_value=fake_decision,
        ) as steer_mutation_mock:
            controller._choose_mutation(entry, [], archive)

        self.assertEqual(
            steer_mutation_mock.call_args.kwargs["parent_hypothesis_nl"], "Cached A."
        )

    def test_reflect_result_stores_child_hypothesis_nl_in_archive_metadata(self):
        config = HypoEvolveConfig()
        controller = HypoEvolveController(
            config,
            evaluator=type("FakeEvaluator", (), {"evaluate": lambda self, hypothesis: {}})(),
            llm_client=object(),
        )
        archive = MAPElitesArchive()
        parent = Hypothesis(root=AtomicNode("A"))
        archive.add(parent, {"combined_score": 0.1}, metadata={"hypothesis_nl": "A"})
        child = Hypothesis(root=AtomicNode("B"))

        with tempfile.TemporaryDirectory() as tmp, patch(
            "hypoevolve.controller.llm_hypothesis_to_natural_language",
            return_value="B in natural language.",
        ):
            run_dir = Path(tmp)
            (run_dir / "artifacts").mkdir(exist_ok=True)
            recorder = RunArtifactRecorder(
                run_dir=run_dir,
                seed_input_text="if A then B",
                worker_count=1,
                workers_enabled=False,
                dataset_schema_path="dataset.yaml",
            )
            descriptor, best_updated = controller._record_archive_result(
                archive=archive,
                iteration=1,
                child_hypothesis=child,
                child_metrics={"combined_score": 0.2},
                metadata={
                    "mutation_summary": "Applied a replace_atomic-style local mutation."
                },
            )
            recorder.record_iteration_result(
                archive=archive,
                iteration=1,
                parent_hypothesis=parent,
                parent_fingerprint=archive.entries[-1].fingerprint,
                child_hypothesis=child,
                child_metrics={"combined_score": 0.2},
                metadata={
                    "mutation_summary": "Applied a replace_atomic-style local mutation.",
                    "hypothesis_nl": "B in natural language.",
                },
                descriptor=descriptor,
                best_updated=best_updated,
            )

            checkpoint = json.loads(
                (run_dir / "checkpoint.json").read_text(encoding="utf-8")
            )

        self.assertEqual(
            archive.best.metadata["hypothesis_nl"], "B in natural language."
        )
        self.assertEqual(
            checkpoint["archive"][0]["metadata"]["hypothesis_nl"],
            "B in natural language.",
        )

    def test_make_worker_task_threads_cached_parent_hypothesis_nl(self):
        config = HypoEvolveConfig()
        controller = HypoEvolveController(
            config,
            evaluator=type("FakeEvaluator", (), {"evaluate": lambda self, hypothesis: {}})(),
            llm_client=object(),
        )
        archive = MAPElitesArchive()
        archive.add(
            Hypothesis(root=AtomicNode("A")),
            {"combined_score": 0.1},
            metadata={"hypothesis_nl": "Cached A."},
        )

        task, _parent = controller._make_worker_task(archive, iteration=1, recent_history=[])

        self.assertEqual(task.parent_hypothesis_nl, "Cached A.")
        self.assertEqual(task.seen_fingerprints, [fingerprint(Hypothesis(root=AtomicNode("A")))])

    def test_choose_mutation_can_use_random_steering_prompt(self):
        config = HypoEvolveConfig()
        config.search.random_steering_prob = 1.0
        controller = HypoEvolveController(
            config,
            evaluator=type("FakeEvaluator", (), {"evaluate": lambda self, hypothesis: {}})(),
            llm_client=object(),
        )
        archive = MAPElitesArchive()
        parent = Hypothesis(root=AtomicNode("A"))
        entry = archive.add(
            parent,
            {"combined_score": 0.1},
            metadata={"hypothesis_nl": "Cached A."},
        )
        fake_decision = type(
            "FakeDecision",
            (),
            {
                "child_hypothesis": Hypothesis(root=AtomicNode("B")),
                "domain_reason": "",
                "score_reason": "",
                "mutation_summary": "Applied three exploratory local mutations.",
            },
        )()

        with patch(
            "hypoevolve.controller.steer_mutation",
            return_value=fake_decision,
        ) as steer_mutation_mock:
            _child, metadata = controller._choose_mutation(entry, [], archive)

        self.assertTrue(steer_mutation_mock.call_args.kwargs["use_random_steering"])
        self.assertTrue(metadata["random_steering"])

    def test_run_skips_duplicate_child_before_evaluation(self):
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
        ), patch(
            "hypoevolve.controller.evaluate_hypothesis",
            side_effect=fake_evaluate,
        ):
            config.output.base_dir = tmp
            controller = HypoEvolveController(
                config,
                evaluator=type(
                    "FakeEvaluator",
                    (),
                    {"evaluate": lambda self, hypothesis: {"combined_score": 0.5}},
                )(),
                llm_client=object(),
            )
            result = controller.run("if A then B")

            trace_lines = (result.run_dir / "trace.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
            log_text = (result.run_dir / "hypoevolve.log").read_text(
                encoding="utf-8"
            )

        self.assertEqual(call_count, 1)
        self.assertEqual(len(trace_lines), 1)
        self.assertIn("[run.duplicate_summary] total_skips=1", log_text)

    def test_run_without_seed_generates_initial_hypothesis(self):
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
        ), patch(
            "hypoevolve.controller.evaluate_hypothesis",
            return_value={"combined_score": 0.5},
        ):
            config.output.base_dir = tmp
            controller = HypoEvolveController(
                config,
                evaluator=type(
                    "FakeEvaluator",
                    (),
                    {"evaluate": lambda self, hypothesis: {"combined_score": 0.5}},
                )(),
                llm_client=object(),
            )
            result = controller.run()

        generate_seed_mock.assert_called_once()
        parse_mock.assert_called_once_with(
            "Generated seed hypothesis.",
            llm=controller.llm_client,
            retries=config.parser.retries,
        )
        self.assertTrue(result.seed_generated)
        self.assertEqual(result.seed_input_text, "Generated seed hypothesis.")
