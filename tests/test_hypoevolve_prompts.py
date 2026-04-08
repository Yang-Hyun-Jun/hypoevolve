import tempfile
import unittest
from pathlib import Path

from hypoevolve.prompts import load_and_render_prompt, load_prompt, render_prompt


class TestHypoEvolvePrompts(unittest.TestCase):
    def test_render_prompt_replaces_variables(self):
        template = "Hello {{NAME}}, score={{SCORE}}"
        rendered = render_prompt(template, {"NAME": "Alice", "SCORE": 0.91})
        self.assertEqual(rendered, "Hello Alice, score=0.91")

    def test_render_prompt_leaves_unknown_placeholders(self):
        template = "Hello {{NAME}}, extra={{EXTRA}}"
        rendered = render_prompt(template, {"NAME": "Alice"})
        self.assertEqual(rendered, "Hello Alice, extra={{EXTRA}}")

    def test_load_and_render_prompt_works_with_prompt_files(self):
        rendered = load_and_render_prompt(
            "evaluator",
            "user.md",
            variables={
                "HYPOTHESIS_PRETTY": "If A then B",
                "DATASET_DESCRIPTION": "demo dataset",
                "INDEX_NAME": "close_time",
                "INDEX_DTYPE": "datetime64[us]",
                "ENTITIES": "BTCUSDT, ETHUSDT",
                "COLUMN_SPECS": "- close: close price",
                "DATASET_ACCESSOR_DOC": "accessor.load_dataframe(entity)",
            },
        )
        self.assertIn("If A then B", rendered)
        self.assertIn("close_time", rendered)
        self.assertIn("accessor.load_dataframe(entity)", rendered)

    def test_evaluator_prompts_require_raw_python_only(self):
        system_prompt = load_prompt("evaluator", "system.md")
        user_prompt = load_prompt("evaluator", "user.md")
        self.assertIn("Do not use code fences", system_prompt)
        self.assertIn("Do not call `evaluate_hypothesis(...)` at module scope.", system_prompt)
        self.assertIn("Any required package imports must be done lazily inside `evaluate_hypothesis(...)`", system_prompt)
        self.assertIn("Do not use third-party packages other than pandas.", system_prompt)
        self.assertIn("# Parameter Handling Rules", system_prompt)
        self.assertIn('parameters.get("LOW_PRICE_WINDOW", 20)', system_prompt)
        self.assertIn('parameters["LOW_PRICE_WINDOW"]', system_prompt)
        self.assertIn('Do not use bare `parameters.get("KEY")` without a default.', system_prompt)
        self.assertIn("Treat schema column names as exact, case-sensitive ground truth.", system_prompt)
        self.assertIn("Do not assume transformed, normalized, z-score, rolling, or renamed columns already exist", system_prompt)
        self.assertIn("# JSON Serialization Rules", system_prompt)
        self.assertIn('"support_count": int(support_count)', system_prompt)
        self.assertIn('"combined_score": float(combined_score)', system_prompt)
        self.assertIn("# Import Rules", system_prompt)
        self.assertIn("Use pandas only.", system_prompt)
        self.assertIn("Do not rely on top-level third-party imports.", system_prompt)
        self.assertNotIn("numpy", system_prompt.lower())
        self.assertIn("Return only raw Python source for `candidate.py`.", user_prompt)
        self.assertIn("Do not add explanations, notes, or example usage.", user_prompt)
        self.assertIn("Treat the column names shown in Column Specifications as exact, case-sensitive names.", user_prompt)

    def test_core_prompt_examples_are_domain_neutral(self):
        parser_prompt = load_prompt("parser", "system.md")
        measurable_prompt = load_prompt("measurable", "system.md")
        steering_prompt = load_prompt("steering", "system.md")
        nl_prompt = load_prompt("nl", "system.md")

        for prompt in [parser_prompt, measurable_prompt, steering_prompt, nl_prompt]:
            self.assertNotIn("BTCUSDT", prompt)
            self.assertNotIn("DOGEUSDT", prompt)
            self.assertNotIn("ETHUSDT", prompt)

        self.assertNotIn("funding fee", parser_prompt.lower())
        self.assertIn("ENTITY_A", measurable_prompt)
        self.assertIn("ENTITY_B", steering_prompt)
        self.assertIn("entity names", nl_prompt)


if __name__ == "__main__":
    unittest.main()
