import json
import tempfile
import unittest
from pathlib import Path

from hypoevolve.config import HypoEvolveConfig
from hypoevolve.controller import HypoEvolveController


class FakeLLM:
    def generate_json(self, system, user, **kwargs):
        if 'mutation_summary' in system:
            return {
                'child_hypothesis': {
                    'kind': 'relation',
                    'name': 'IMPLIES',
                    'inputs': [
                        {'kind': 'atomic', 'name': 'A2'},
                        {'kind': 'atomic', 'name': 'B'},
                    ],
                },
                'domain_reason': 'Tightening the stress condition is plausible from a crypto downside-regime perspective.',
                'score_reason': 'Tightening one atomic condition is a local change that may improve precision.',
                'operation_score_rankings': {'replace_atomic_feature': 1, 'append_atomic': 2},
                'mutation_summary': 'Applied a replace_atomic-style change in the condition side while keeping the overall relation structure.',
            }
        return {
            'kind': 'relation',
            'name': 'IMPLIES',
            'inputs': [
                {'kind': 'atomic', 'name': 'A'},
                {'kind': 'atomic', 'name': 'B'},
            ],
        }

    def generate_text(self, system, user, **kwargs):
        return 'If A then B.'


class TestPhase0SingleWorkerProvidedFixture(unittest.TestCase):
    def test_single_worker_provided_fixture_matches_stubbed_run(self):
        fixture_root = Path('tests/fixtures/parity/single_worker_provided_hypothesis/normalized_expected')
        with tempfile.TemporaryDirectory() as tmp:
            config = HypoEvolveConfig()
            config.search.iterations = 1
            config.output.base_dir = tmp
            config.evaluator.dataset_schema_path = str(Path(tmp) / 'dataset.yaml')
            Path(config.evaluator.dataset_schema_path).write_text(
                "description: test\nindex:\n  name: close_time\n  dtype: datetime64[us]\nfiles:\n  -\n    entity: BTCUSDT\n    path: /tmp/BTCUSDT.parquet\ncolumns:\n  -\n    name: CLOSE\n",
                encoding='utf-8',
            )
            fake_evaluator = type('FakeEvaluator', (), {'evaluate': lambda self, hypothesis: {'combined_score': 0.5}})()
            controller = HypoEvolveController(config, evaluator=fake_evaluator, llm_client=FakeLLM())
            result = controller.run('if A then B')
            run_dir = result.run_dir
            actual = {
                'best.json': json.loads((run_dir / 'best.json').read_text(encoding='utf-8')),
                'checkpoint.json': json.loads((run_dir / 'checkpoint.json').read_text(encoding='utf-8')),
                'run_summary.json': json.loads((run_dir / 'run_summary.json').read_text(encoding='utf-8')),
                'score_history.json': json.loads((run_dir / 'score_history.json').read_text(encoding='utf-8')),
                'trace.json': [json.loads(line) for line in (run_dir / 'trace.jsonl').read_text(encoding='utf-8').splitlines()],
            }
            actual['run_summary.json']['dataset_schema_path'] = 'dataset.yaml'

        for filename, payload in actual.items():
            expected = json.loads((fixture_root / filename).read_text(encoding='utf-8'))
            self.assertEqual(payload, expected, msg=f'mismatch for {filename}')


if __name__ == '__main__':
    unittest.main()
