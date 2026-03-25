import unittest

from elg import Hypothesis, RelationNode
from hypoevolve.prompts import load_prompt
from hypoevolve.parser import (
    ParseError,
    llm_parse_hypothesis,
    parse_hypothesis_text,
)


class TestHypoEvolveParser(unittest.TestCase):
    def test_parse_hypothesis_text_uses_llm_parser_path(self):
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
        hypothesis = parse_hypothesis_text('custom input', llm=FakeLLM(), retries=1)
        self.assertIsInstance(hypothesis, Hypothesis)
        self.assertIsInstance(hypothesis.root, RelationNode)

    def test_retry_then_fail_with_context(self):
        class BadLLM:
            def __init__(self):
                self.calls = 0
            def generate_json(self, system, user, **kwargs):
                self.calls += 1
                raise RuntimeError('boom')
        llm = BadLLM()
        with self.assertRaises(ParseError) as ctx:
            parse_hypothesis_text('x', llm=llm, retries=2)
        self.assertEqual(llm.calls, 3)
        self.assertEqual(len(ctx.exception.errors), 3)

    def test_llm_parse_hypothesis_returns_normalized_hypothesis(self):
        class FakeLLM:
            def generate_json(self, system, user, **kwargs):
                return {
                    "kind": "logical",
                    "op": "AND",
                    "inputs": [
                        {"kind": "atomic", "name": "B", "type": "abstract", "source": "semantic", "params": {}},
                        {"kind": "atomic", "name": "A", "type": "abstract", "source": "semantic", "params": {}},
                        {"kind": "atomic", "name": "A", "type": "abstract", "source": "semantic", "params": {}},
                    ],
                    "params": {},
                }

        hypothesis = llm_parse_hypothesis("A and B", llm=FakeLLM())
        self.assertIsInstance(hypothesis, Hypothesis)
        self.assertEqual(hypothesis.root.inputs[0].name, "A")
        self.assertEqual(hypothesis.root.inputs[1].name, "B")

    def test_llm_parse_hypothesis_retries_and_fails(self):
        class BadLLM:
            def __init__(self):
                self.calls = 0

            def generate_json(self, system, user, **kwargs):
                self.calls += 1
                return {"kind": "mystery"}

        llm = BadLLM()
        with self.assertRaises(ParseError) as ctx:
            llm_parse_hypothesis("bad", llm=llm, retries=2)
        self.assertEqual(llm.calls, 3)
        self.assertEqual(len(ctx.exception.errors), 3)

    def test_llm_parse_hypothesis_rejects_invalid_atomic_payload(self):
        class BadLLM:
            def generate_json(self, system, user, **kwargs):
                return {"kind": "atomic", "name": "   "}

        with self.assertRaises(ParseError):
            llm_parse_hypothesis("bad", llm=BadLLM())
