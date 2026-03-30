import tempfile
import unittest
import json
from pathlib import Path
from unittest.mock import patch

from elg import AtomicNode, Hypothesis
from hypoevolve.archive import Archive
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
                        "reason": "Tightening one atomic condition is a local change that may improve precision.",
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
                        "reason": "Tightening one atomic condition is a local change that may improve precision.",
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
            config.search.mutation_atomic_pool = ["C"]
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
        archive = Archive()
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
                "reason": "Use cached NL.",
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
        archive = Archive()
        parent = Hypothesis(root=AtomicNode("A"))
        archive.add(parent, {"combined_score": 0.1}, metadata={"hypothesis_nl": "A"})
        child = Hypothesis(root=AtomicNode("B"))

        with tempfile.TemporaryDirectory() as tmp, patch(
            "hypoevolve.controller.llm_hypothesis_to_natural_language",
            return_value="B in natural language.",
        ):
            run_dir = Path(tmp)
            (run_dir / "artifacts").mkdir(exist_ok=True)
            controller._reflect_result(
                archive=archive,
                run_dir=run_dir,
                iteration=1,
                parent_hypothesis=parent,
                child_hypothesis=child,
                child_metrics={"combined_score": 0.2},
                metadata={"mutation_summary": "Applied a replace_atomic-style local mutation."},
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
        archive = Archive()
        archive.add(
            Hypothesis(root=AtomicNode("A")),
            {"combined_score": 0.1},
            metadata={"hypothesis_nl": "Cached A."},
        )

        task, _parent = controller._make_worker_task(archive, iteration=1, recent_history=[])

        self.assertEqual(task.parent_hypothesis_nl, "Cached A.")
