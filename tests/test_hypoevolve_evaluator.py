import unittest

from elg import AtomicNode, Hypothesis
from hypoevolve.evaluator import PlaceholderEvaluator, evaluate_hypothesis


class TestHypoEvolveEvaluator(unittest.TestCase):
    def test_placeholder_evaluator_returns_expected_shape(self):
        evaluator = PlaceholderEvaluator(seed=1)
        metrics = evaluator.evaluate(Hypothesis(root=AtomicNode('A')))
        self.assertIn('combined_score', metrics)
        self.assertIn('raw_score', metrics)
        self.assertIn('complexity_penalty', metrics)

    def test_helper_calls_evaluator(self):
        evaluator = PlaceholderEvaluator(seed=1)
        metrics = evaluate_hypothesis(Hypothesis(root=AtomicNode('A')), evaluator)
        self.assertIn('combined_score', metrics)
