import unittest
from unittest.mock import Mock, patch

from hypoevolve.hypo.hypothesis import (
    HypothesisGenerationError,
    TreePairHypothesis,
    build_hypothesis_prompt_variables,
    generate_random_tree_pair_hypothesis,
    llm_generate_hypothesis_from_trees,
)


class FakeTree:
    def __init__(self, rendered: str, descriptions: str):
        self.rendered = rendered
        self.descriptions = descriptions

    def render(self, return_str: bool = False):
        if return_str:
            return self.rendered
        return None

    def get_node_descriptions(self) -> str:
        return self.descriptions


class TestHypoTreeHypothesisGeneration(unittest.TestCase):
    def test_build_hypothesis_prompt_variables_concatenates_tree_sections(self):
        tree_a = FakeTree("TREE A", "- A desc")
        tree_b = FakeTree("TREE B", "- B desc")

        variables = build_hypothesis_prompt_variables(tree_a, tree_b)

        self.assertEqual(variables["TREE_A"], "TREE A")
        self.assertEqual(variables["TREE_B"], "TREE B")
        self.assertEqual(variables["NODE_DESCRIPTIONS"], "- A desc\n- B desc")

    def test_llm_generate_hypothesis_from_trees_extracts_tagged_payload(self):
        class FakeLLM:
            def generate_text(self, system, user, **kwargs):
                self.system = system
                self.user = user
                return "<hypothesis>Feature A likely filters when Feature B becomes active.</hypothesis>"

        llm = FakeLLM()
        tree_a = FakeTree("TREE A", "- A desc")
        tree_b = FakeTree("TREE B", "- B desc")

        hypothesis = llm_generate_hypothesis_from_trees(tree_a, tree_b, llm=llm)

        self.assertEqual(
            hypothesis, "Feature A likely filters when Feature B becomes active."
        )
        self.assertIn("TREE A", llm.user)
        self.assertIn("TREE B", llm.user)
        self.assertIn("Role", llm.system)

    def test_llm_generate_hypothesis_from_trees_retries_then_fails(self):
        class BadLLM:
            def __init__(self):
                self.calls = 0

            def generate_text(self, system, user, **kwargs):
                self.calls += 1
                raise RuntimeError("boom")

        llm = BadLLM()
        tree_a = FakeTree("TREE A", "- A desc")
        tree_b = FakeTree("TREE B", "- B desc")

        with self.assertRaises(HypothesisGenerationError) as ctx:
            llm_generate_hypothesis_from_trees(tree_a, tree_b, llm=llm, retries=2)

        self.assertEqual(llm.calls, 3)
        self.assertEqual(len(ctx.exception.errors), 3)

    def test_generate_random_tree_pair_hypothesis_uses_lazy_tree_helpers(self):
        tree_a = FakeTree("TREE A", "- A desc")
        tree_b = FakeTree("TREE B", "- B desc")

        class FakeLLM:
            def generate_text(self, system, user, **kwargs):
                return "<hypothesis>Combined trees imply a regime-dependent interaction.</hypothesis>"

        get_tree_generator = Mock(return_value="GENERATOR")

        with patch(
            "hypoevolve.hypo.hypothesis._load_tree_generation_helpers",
            return_value=(
                get_tree_generator,
                lambda generator, max_depth, num_trees: [tree_a, tree_b],
            ),
        ):
            result = generate_random_tree_pair_hypothesis(
                llm=FakeLLM(),
                max_depth=3,
                dataset_schema_path="custom-dataset.yaml",
            )

        get_tree_generator.assert_called_once_with(
            dataset_schema_path="custom-dataset.yaml"
        )
        self.assertIsInstance(result, TreePairHypothesis)
        self.assertIs(result.tree_a, tree_a)
        self.assertIs(result.tree_b, tree_b)
        self.assertEqual(
            result.hypothesis, "Combined trees imply a regime-dependent interaction."
        )


if __name__ == "__main__":
    unittest.main()
