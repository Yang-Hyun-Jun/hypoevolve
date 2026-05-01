import unittest

from hypoevolve.skills.seed_generation.nodes import nodes
from hypoevolve.skills.seed_generation.tree.base import HypoTree
from hypoevolve.skills.seed_generation.tree.generator import HypoTreeGenerator


class TestHypoTree(unittest.TestCase):
    def test_tree_insert_render_and_roundtrip(self):
        tree = HypoTree("demo")
        tree.insert(nodes.Comparison())
        tree.insert(nodes.DATA(label="A"))
        tree.insert(nodes.DATA(label="B"))

        self.assertTrue(tree.iscompleted)
        self.assertEqual(tree.depth, 1)
        rendered = tree.render(return_str=True)
        self.assertIn("Comparison()", rendered)
        self.assertIn("DATA[A]", rendered)
        self.assertIn("DATA[B]", rendered)
        descriptions = tree.get_node_descriptions()
        self.assertIn("Comparison()", descriptions)
        self.assertIn("DATA[A]", descriptions)

        restored = HypoTree.from_dict(tree.to_dict())
        self.assertEqual(restored.name, "demo")
        self.assertEqual(restored.depth, 1)
        self.assertEqual(restored.render(return_str=True), rendered)

    def test_tree_call_delegates_to_evaluate(self):
        tree = HypoTree("callable")
        tree.insert(nodes.Comparison())
        tree.insert(nodes.DATA(label="A"))
        tree.insert(nodes.DATA(label="B"))

        tree.evaluate = lambda: "ok"
        self.assertEqual(tree(), "ok")


class TestHypoTreeGenerator(unittest.TestCase):
    def test_initial_mask_and_step_transition_follow_io_constraints(self):
        generator = HypoTreeGenerator(
            [
                nodes.DATA(label="A"),
                nodes.SMA(period=2),
                nodes.Comparison(),
            ]
        )

        state = generator.reset(max_depth=2, tree_name="demo")
        self.assertEqual(state["mask"].tolist(), [0, 0, 1])
        next_state = generator.step(2)
        self.assertEqual(next_state["mask"], [0, 1, 0])
        leaf_state = generator.step(1)
        self.assertEqual(leaf_state["mask"], [1, 0, 0])

    def test_generate_returns_tree_when_done(self):
        generator = HypoTreeGenerator([nodes.DATA(label="A")])
        generator.tree = HypoTree("generated")

        import numpy as np
        original_choice = np.random.choice
        np.random.choice = lambda arr, p=None: 0
        try:
            from unittest.mock import Mock
            generator.reset = Mock(return_value={"mask": [1], "done": False})
            generator.step = Mock(side_effect=[{"mask": [1], "done": False}, {"mask": [1], "done": True}])
            tree = generator.generate(max_depth=2, tree_max_iter=5)
        finally:
            np.random.choice = original_choice

        generator.reset.assert_called_once_with(2)
        self.assertIs(tree, generator.tree)
        self.assertEqual(generator.step.call_count, 2)


if __name__ == "__main__":
    unittest.main()
