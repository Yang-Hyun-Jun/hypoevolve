import tempfile
import unittest
from pathlib import Path

from hypoevolve.llm import LLMClient
from hypoevolve.config import HypoEvolveConfig
from hypoevolve.controller import HypoEvolveController


class TestHypoEvolveController(unittest.TestCase):
    def test_happy_path_run_completes_and_writes_outputs(self):
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
        with tempfile.TemporaryDirectory() as tmp:
            config = HypoEvolveConfig()
            config.search.iterations = 2
            config.output.base_dir = tmp
            controller = HypoEvolveController(config, llm_client=FakeLLM())
            result = controller.run('if A then B')
            self.assertTrue(result.run_dir.exists())
            self.assertTrue((result.run_dir / 'trace.jsonl').exists())
            self.assertTrue((result.run_dir / 'checkpoint.json').exists())
            self.assertTrue((result.run_dir / 'best.json').exists())
            self.assertIn('combined_score', result.best_metrics)
