import unittest

from elg import Hypothesis, RelationNode
from hypoevolve.parser import ParseError, fallback_parse_hypothesis, parse_hypothesis_text


class TestHypoEvolveParser(unittest.TestCase):
    def test_fallback_parser_returns_hypothesis(self):
        hypothesis = fallback_parse_hypothesis('A and B')
        self.assertIsInstance(hypothesis, Hypothesis)

    def test_fallback_parser_can_parse_if_then(self):
        hypothesis = fallback_parse_hypothesis('if funding fee positive then returns positive')
        self.assertIsInstance(hypothesis.root, RelationNode)

    def test_injectable_parser_path(self):
        def custom_parser(text):
            return fallback_parse_hypothesis(text)
        hypothesis = parse_hypothesis_text('custom input', parser=custom_parser, retries=1)
        self.assertIsInstance(hypothesis, Hypothesis)

    def test_retry_then_fail_with_context(self):
        calls = {'n': 0}
        def bad_parser(text):
            calls['n'] += 1
            raise RuntimeError('boom')
        with self.assertRaises(ParseError) as ctx:
            parse_hypothesis_text('x', parser=bad_parser, retries=2)
        self.assertEqual(calls['n'], 3)
        self.assertEqual(len(ctx.exception.errors), 3)
