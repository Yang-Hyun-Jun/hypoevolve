import tempfile
import unittest
import json
from pathlib import Path
from unittest.mock import patch

from hypoevolve.config import HypoEvolveConfig
from hypoevolve.controller import HypoEvolveController


class TestHypoEvolveController(unittest.TestCase):
    def test_happy_path_run_completes_and_writes_outputs(self):
        class FakeLLM:
            def generate_json(self, system, user, **kwargs):
                if "selected_candidate_index" in system:
                    return {
                        "selected_candidate_index": 0,
                        "reason": "Choose the first legal candidate for a minimal local edit.",
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
                if "selected_candidate_index" in system:
                    return {
                        "selected_candidate_index": 0,
                        "reason": "Choose the first legal candidate for a minimal local edit.",
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
            self.assertEqual(trace[1]["metadata"]["selected_candidate_index"], 0)
