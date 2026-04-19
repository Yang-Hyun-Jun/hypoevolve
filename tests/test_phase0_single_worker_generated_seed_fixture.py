import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from hypoevolve.elg import AtomicNode, Hypothesis
from hypoevolve.config import HypoEvolveConfig
from hypoevolve.controller import HypoEvolveController


class TestPhase0SingleWorkerGeneratedSeedFixture(unittest.TestCase):
    def test_single_worker_generated_seed_fixture_matches_stubbed_run(self):
        fixture_root = Path('tests/fixtures/parity/single_worker_generated_seed/normalized_expected')
        seed = Hypothesis(root=AtomicNode('A'))
        fake_tree_result = type('FakeTreeResult', (), {'hypothesis': 'Generated seed hypothesis.'})()
        fake_decision = type(
            'FakeDecision',
            (),
            {
                'child_hypothesis': Hypothesis(root=AtomicNode('B')),
                'domain_reason': 'Generated-seed local mutation.',
                'score_reason': 'Use generated seed path.',
                'operation_score_rankings': {'replace_atomic_feature': 1},
                'mutation_summary': 'Applied a replace_atomic-style local mutation.',
            },
        )()

        with tempfile.TemporaryDirectory() as tmp, patch(
            'hypoevolve.controller.generate_random_tree_pair_hypothesis', return_value=fake_tree_result
        ), patch(
            'hypoevolve.controller.parse_hypothesis_text', return_value=seed
        ), patch(
            'hypoevolve.controller.llm_make_hypothesis_measurable', return_value=seed
        ), patch(
            'hypoevolve.controller.llm_hypothesis_to_natural_language', return_value='A'
        ), patch(
            'hypoevolve.controller.steer_mutation', return_value=fake_decision
        ):
            config = HypoEvolveConfig()
            config.search.iterations = 1
            config.output.base_dir = tmp
            fake_evaluator = type(
                'FakeEvaluator',
                (),
                {
                    'evaluate': lambda self, hypothesis: {'combined_score': 0.5},
                    'last_evaluation_artifacts': {},
                },
            )()
            controller = HypoEvolveController(
                config,
                evaluator=fake_evaluator,
                llm_client=object(),
            )
            result = controller.run(None)
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
