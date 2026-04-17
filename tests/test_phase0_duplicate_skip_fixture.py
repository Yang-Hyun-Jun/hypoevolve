import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from elg import AtomicNode, Hypothesis
from hypoevolve.config import HypoEvolveConfig
from hypoevolve.controller import HypoEvolveController


class TestPhase0DuplicateSkipFixture(unittest.TestCase):
    def test_duplicate_skip_fixture_matches_stubbed_run(self):
        fixture_root = Path('tests/fixtures/parity/duplicate_skip/normalized_expected')
        seed = Hypothesis(root=AtomicNode('A'))
        fake_decision = type(
            'FakeDecision',
            (),
            {
                'child_hypothesis': Hypothesis(root=AtomicNode('A')),
                'domain_reason': 'Duplicate child.',
                'score_reason': 'Should skip repeated eval.',
                'operation_score_rankings': {'replace_atomic_feature': 1},
                'mutation_summary': 'Proposed an already known hypothesis.',
            },
        )()
        config = HypoEvolveConfig()
        config.search.iterations = 1

        def fake_evaluate(hypothesis, evaluator):
            return {'combined_score': 0.5}

        with tempfile.TemporaryDirectory() as tmp, patch(
            'hypoevolve.controller.parse_hypothesis_text', return_value=seed
        ), patch(
            'hypoevolve.controller.llm_make_hypothesis_measurable', return_value=seed
        ), patch(
            'hypoevolve.controller.llm_hypothesis_to_natural_language', return_value='A'
        ), patch(
            'hypoevolve.controller.steer_mutation', return_value=fake_decision
        ):
            config.output.base_dir = tmp
            fake_evaluator = type(
                'FakeEvaluator',
                (),
                {
                    'evaluate': lambda self, hypothesis: fake_evaluate(hypothesis, self),
                    'last_evaluation_artifacts': {},
                },
            )()
            controller = HypoEvolveController(
                config,
                evaluator=fake_evaluator,
                llm_client=object(),
            )
            result = controller.run('if A then B')
            run_dir = result.run_dir

            actual = {
                'best.json': json.loads((run_dir / 'best.json').read_text(encoding='utf-8')),
                'checkpoint.json': json.loads((run_dir / 'checkpoint.json').read_text(encoding='utf-8')),
                'run_summary.json': json.loads((run_dir / 'run_summary.json').read_text(encoding='utf-8')),
                'score_history.json': json.loads((run_dir / 'score_history.json').read_text(encoding='utf-8')),
                'trace.json': [json.loads(line) for line in (run_dir / 'trace.jsonl').read_text(encoding='utf-8').splitlines()],
            }

        for filename, payload in actual.items():
            expected = json.loads((fixture_root / filename).read_text(encoding='utf-8'))
            self.assertEqual(payload, expected, msg=f'mismatch for {filename}')


if __name__ == '__main__':
    unittest.main()
