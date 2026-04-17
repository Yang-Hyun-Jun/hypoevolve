import unittest

from elg import AtomicNode, Hypothesis, LogicalNode, RelationNode, RelationType
from hypoevolve.prompts import load_prompt
from hypoevolve.parser import (
    ParseError,
    llm_hypothesis_to_natural_language,
    llm_make_hypothesis_measurable,
    llm_parse_hypothesis,
    parse_hypothesis_text,
    _validate_parser_payload,
)


class TestHypoEvolveParser(unittest.TestCase):
    def test_parse_hypothesis_text_uses_llm_parser_path(self):
        class FakeLLM:
            def generate_json(self, system, user, **kwargs):
                return {
                    "kind": "relation",
                    "name": "IMPLIES",
                    "inputs": [
                        {"kind": "atomic", "name": "A"},
                        {"kind": "atomic", "name": "B"},
                    ],
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
                    "name": "AND",
                    "inputs": [
                        {"kind": "atomic", "name": "B"},
                        {"kind": "atomic", "name": "A"},
                        {"kind": "atomic", "name": "A"},
                    ],
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
                    "name": "IMPLIES",
                    "inputs": [
                        {
                            "kind": "logical",
                            "name": "AND",
                            "inputs": [
                                {
                                    "kind": "atomic",
                                    "name": "ENTITY_A_LOW_STATE_SIGNAL_W{LOOKBACK_WINDOW}@t == True",
                                },
                                {
                                    "kind": "atomic",
                                    "name": "ENTITY_A_ZSCORE_FEATURE_X_W{LOOKBACK_WINDOW}@t < {NEG_Z_THRESHOLD}",
                                },
                            ],
                        },
                        {
                            "kind": "atomic",
                            "name": "ENTITY_B_ZSCORE_TARGET_Y_W{TARGET_WINDOW}@t+{HORIZON} > {POS_Z_THRESHOLD}",
                        },
                    ],
                }

        original = Hypothesis(
            root=RelationNode(
                "IMPLIES",
                [
                    AtomicNode(
                        "entity A enters an unusually weak regime according to feature X"
                    ),
                    AtomicNode("entity B shows a strong positive move in target Y"),
                ],
            )
        )
        measurable = llm_make_hypothesis_measurable(original, llm=FakeLLM())
        self.assertIsInstance(measurable, Hypothesis)
        self.assertEqual(measurable.root.name, RelationType.IMPLIES)
        self.assertIsInstance(measurable.root.inputs[0], LogicalNode)

    def test_llm_make_hypothesis_measurable_retries_and_fails(self):
        class BadLLM:
            def __init__(self):
                self.calls = 0

            def generate_json(self, system, user, **kwargs):
                self.calls += 1
                return {"kind": "logical", "name": "XOR", "inputs": []}

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
        self.assertIn("minimal ELG schema", system_prompt)
        self.assertIn("measurable", measurable_prompt.lower())


    def test_llm_hypothesis_to_natural_language_returns_text(self):
        class FakeLLM:
            def generate_text(self, system, user, **kwargs):
                return "If signal A is above its baseline and signal B is trending upward, then outcome C becomes more likely."

        hypothesis = Hypothesis(
            root=RelationNode(
                "IMPLIES",
                [
                    LogicalNode(
                        "AND",
                        [
                            AtomicNode("signal A is above its baseline"),
                            AtomicNode("signal B is trending upward"),
                        ],
                    ),
                    AtomicNode("outcome C becomes more likely"),
                ],
            )
        )
        text = llm_hypothesis_to_natural_language(hypothesis, llm=FakeLLM())
        self.assertIn("signal A", text)
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

    def test_hypothesis_to_dict_uses_minimal_elg_schema(self):
        hypothesis = Hypothesis(
            root=RelationNode(
                "IMPLIES",
                [
                    LogicalNode("AND", [AtomicNode("A"), AtomicNode("B")]),
                    AtomicNode("C"),
                ],
            )
        )

        self.assertEqual(
            hypothesis.to_dict()["root"],
            {
                "kind": "relation",
                "name": "IMPLIES",
                "inputs": [
                    {
                        "kind": "logical",
                        "name": "AND",
                        "inputs": [
                            {"kind": "atomic", "name": "A"},
                            {"kind": "atomic", "name": "B"},
                        ],
                    },
                    {"kind": "atomic", "name": "C"},
                ],
            },
        )


    def test_validate_parser_payload_rejects_non_mapping_and_bad_inputs(self):
        with self.assertRaises(ParseError):
            _validate_parser_payload([])
        with self.assertRaises(ParseError):
            _validate_parser_payload({"kind": "logical", "name": "AND", "inputs": [1]})
        with self.assertRaises(ParseError):
            _validate_parser_payload({"kind": "relation", "name": "IMPLIES", "inputs": [{"kind": "atomic", "name": "A"}]})

    def test_parse_hypothesis_text_rejects_empty_input_before_llm(self):
        class NeverCalled:
            def generate_json(self, system, user, **kwargs):
                raise AssertionError("should not be called")

        with self.assertRaises(ParseError):
            parse_hypothesis_text("   ", llm=NeverCalled(), retries=1)


if __name__ == "__main__":
    unittest.main()
