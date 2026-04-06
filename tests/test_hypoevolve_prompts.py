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
        self.assertIn("Return only raw Python source for `candidate.py`.", user_prompt)
        self.assertIn("Do not add explanations, notes, or example usage.", user_prompt)


if __name__ == "__main__":
    unittest.main()
