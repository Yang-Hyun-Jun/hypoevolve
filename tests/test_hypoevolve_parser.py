import unittest

from elg import AtomicNode, Hypothesis, LogicalNode, RelationNode, RelationType
from hypoevolve.prompts import load_prompt
from hypoevolve.parser import (
    ParseError,
    llm_hypothesis_to_natural_language,
    llm_make_hypothesis_measurable,
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

        hypothesis = parse_hypothesis_text("custom input", llm=FakeLLM(), retries=1)
        self.assertIsInstance(hypothesis, Hypothesis)
        self.assertIsInstance(hypothesis.root, RelationNode)

    def test_retry_then_fail_with_context(self):
        class BadLLM:
            def __init__(self):
                self.calls = 0

            def generate_json(self, system, user, **kwargs):
                self.calls += 1
                raise RuntimeError("boom")

        llm = BadLLM()
        with self.assertRaises(ParseError) as ctx:
            parse_hypothesis_text("x", llm=llm, retries=2)
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

    def test_llm_make_hypothesis_measurable_preserves_relation_structure(self):
        class FakeLLM:
            def generate_json(self, system, user, **kwargs):
                return {
                    "kind": "relation",
                    "type": "IMPLIES",
                    "inputs": [
                        {
                            "kind": "logical",
                            "op": "AND",
                            "inputs": [
                                {
                                    "kind": "atomic",
                                    "name": "BTCUSDT_NEW_LOW_SIGNAL_10D == True",
                                    "type": "boolean",
                                    "source": "primitive",
                                    "params": {},
                                },
                                {
                                    "kind": "atomic",
                                    "name": "BTCUSDT_NORMALIZED_CLOSE_MOMENTUM_10D < -2.0",
                                    "type": "boolean",
                                    "source": "primitive",
                                    "params": {},
                                },
                            ],
                            "params": {},
                        },
                        {
                            "kind": "atomic",
                            "name": "ETHUSDT_TRANSFORMED_HIGH_JUMP_10D > 1.5",
                            "type": "boolean",
                            "source": "primitive",
                            "params": {},
                        },
                    ],
                    "params": {},
                }

        original = Hypothesis(
            root=RelationNode(
                "IMPLIES",
                [
                    AtomicNode(
                        "sharp downward accelerations in BTCUSDT price indicated by NewLow signal from normalized 10-day close momentum"
                    ),
                    AtomicNode("significant jumps in transformed high ETHUSDT price series"),
                ],
            )
        )
        measurable = llm_make_hypothesis_measurable(original, llm=FakeLLM())
        self.assertIsInstance(measurable, Hypothesis)
        self.assertEqual(measurable.root.type, RelationType.IMPLIES)
        self.assertIsInstance(measurable.root.inputs[0], LogicalNode)

    def test_llm_make_hypothesis_measurable_retries_and_fails(self):
        class BadLLM:
            def __init__(self):
                self.calls = 0

            def generate_json(self, system, user, **kwargs):
                self.calls += 1
                return {"kind": "logical", "op": "XOR", "inputs": []}

        original = Hypothesis(root=AtomicNode("semantic proposition"))
        llm = BadLLM()
        with self.assertRaises(ParseError) as ctx:
            llm_make_hypothesis_measurable(original, llm=llm, retries=2)
        self.assertEqual(llm.calls, 3)
        self.assertEqual(len(ctx.exception.errors), 3)

    def test_parser_prompts_load_from_files(self):
        system_prompt = load_prompt("parser", "system.md")
        retry_prompt = load_prompt("common", "json-retry.md")
        measurable_prompt = load_prompt("measurable", "system.md")
        self.assertIn("Return JSON only.", system_prompt)
        self.assertIn("invalid", retry_prompt.lower())
        self.assertIn("measurable", measurable_prompt.lower())


    def test_llm_hypothesis_to_natural_language_returns_text(self):
        class FakeLLM:
            def generate_text(self, system, user, **kwargs):
                return "If funding fee is positive and price is above SMA20, then short-term returns are positive."

        hypothesis = Hypothesis(
            root=RelationNode(
                "IMPLIES",
                [
                    LogicalNode("AND", [AtomicNode("funding fee is positive"), AtomicNode("price is above SMA20")]),
                    AtomicNode("short-term returns are positive"),
                ],
            )
        )
        text = llm_hypothesis_to_natural_language(hypothesis, llm=FakeLLM())
        self.assertIn("funding fee", text)
        self.assertIn("then", text.lower())

    def test_llm_hypothesis_to_natural_language_retries_and_fails(self):
        class BadLLM:
            def __init__(self):
                self.calls = 0

            def generate_text(self, system, user, **kwargs):
                self.calls += 1
                return "   "

        llm = BadLLM()
        hypothesis = Hypothesis(root=AtomicNode("A"))
        with self.assertRaises(ParseError) as ctx:
            llm_hypothesis_to_natural_language(hypothesis, llm=llm, retries=2)
        self.assertEqual(llm.calls, 3)
        self.assertEqual(len(ctx.exception.errors), 3)


if __name__ == "__main__":
    unittest.main()
