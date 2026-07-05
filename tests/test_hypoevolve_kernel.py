import unittest

from hypoevolve.elg import (
    AtomicNode,
    Hypothesis,
    LogicalNode,
    LogicalOp,
    RelationNode,
    RelationType,
)
from hypoevolve.elg.kernel import (
    LAMBDA_NEG,
    LAMBDA_WRAP,
    atomic_sim,
    tree_distance,
    tree_kernel,
)


def A(name: str) -> AtomicNode:
    return AtomicNode(name=name)


def AND(*children):
    return LogicalNode(name=LogicalOp.AND, inputs=list(children))


def OR(*children):
    return LogicalNode(name=LogicalOp.OR, inputs=list(children))


def NOT(child):
    return LogicalNode(name=LogicalOp.NOT, inputs=[child])


def IMPLIES(cond, target):
    return RelationNode(name=RelationType.IMPLIES, inputs=[cond, target])


def CONTRADICT(cond, target):
    return RelationNode(name=RelationType.CONTRADICT, inputs=[cond, target])


def CORRELATE(cond, target):
    return RelationNode(name=RelationType.CORRELATE, inputs=[cond, target])


class TestAtomicSim(unittest.TestCase):
    def test_identical_strings_return_one(self):
        self.assertEqual(atomic_sim("VIX above 30", "VIX above 30"), 1.0)

    def test_completely_disjoint_strings_return_zero(self):
        self.assertEqual(atomic_sim("abcdef", "zzzzzz"), 0.0)

    def test_empty_strings_treated_as_identical(self):
        self.assertEqual(atomic_sim("", ""), 1.0)

    def test_partial_overlap_falls_between_zero_and_one(self):
        value = atomic_sim("stock price rises", "stock price falls")
        self.assertGreater(value, 0.3)
        self.assertLess(value, 1.0)


class TestTreeKernelBasics(unittest.TestCase):
    def test_identical_hypothesis_has_kernel_one(self):
        h = IMPLIES(A("earnings beat"), A("price rises"))
        self.assertAlmostEqual(tree_kernel(h, h), 1.0, places=6)
        self.assertAlmostEqual(tree_distance(h, h), 0.0, places=6)

    def test_hypothesis_wrapper_and_bare_root_agree(self):
        node = IMPLIES(A("earnings beat"), A("price rises"))
        wrapped = Hypothesis(root=node)
        self.assertEqual(tree_kernel(node, node), tree_kernel(wrapped, wrapped))

    def test_distance_is_one_minus_kernel(self):
        h1 = IMPLIES(A("a"), A("b"))
        h2 = IMPLIES(A("c"), A("d"))
        self.assertAlmostEqual(
            tree_distance(h1, h2), 1.0 - tree_kernel(h1, h2), places=10
        )


class TestTreeKernelStructuralDefenses(unittest.TestCase):
    def test_relation_type_mismatch_is_zero(self):
        h1 = IMPLIES(A("temp high"), A("sales up"))
        h2 = CONTRADICT(A("temp high"), A("sales up"))
        self.assertEqual(tree_kernel(h1, h2), 0.0)

    def test_correlate_and_contradict_are_zero(self):
        h1 = CORRELATE(A("x"), A("y"))
        h2 = CONTRADICT(A("x"), A("y"))
        self.assertEqual(tree_kernel(h1, h2), 0.0)

    def test_causal_reversal_with_distinct_atoms_gives_low_similarity(self):
        h1 = IMPLIES(A("company earnings beat"), A("stock price rises"))
        h2 = IMPLIES(A("stock price rises"), A("company earnings beat"))
        self.assertLess(tree_kernel(h1, h2), 0.4)

    def test_and_children_are_order_invariant(self):
        h1 = IMPLIES(AND(A("x"), A("y")), A("z"))
        h2 = IMPLIES(AND(A("y"), A("x")), A("z"))
        self.assertAlmostEqual(tree_kernel(h1, h2), 1.0, places=6)

    def test_and_or_operator_mismatch_returns_zero(self):
        h1 = IMPLIES(AND(A("x"), A("y")), A("z"))
        h2 = IMPLIES(OR(A("x"), A("y")), A("z"))
        # Position 1 has kind mismatch fallback of 0; position 2 matches perfectly.
        value = tree_kernel(h1, h2)
        self.assertAlmostEqual(value, 0.5, places=6)

    def test_wrapper_descent_credits_shared_child(self):
        h1 = IMPLIES(A("VIX above 30"), A("SPY drops"))
        h2 = IMPLIES(AND(A("VIX above 30"), A("if weekday")), A("SPY drops"))
        # cond side descends into AND: LAMBDA_WRAP * 1.0 = 0.5, target side = 1.0
        expected = 0.5 * (LAMBDA_WRAP * 1.0 + 1.0)
        self.assertAlmostEqual(tree_kernel(h1, h2), expected, places=6)

    def test_not_wrapper_applies_polarity_penalty(self):
        h1 = IMPLIES(A("momentum"), A("price up"))
        h2 = IMPLIES(A("momentum"), NOT(A("price up")))
        # cond side matches (1.0), target side descends into NOT with LAMBDA_NEG.
        expected = 0.5 * (1.0 + LAMBDA_NEG * 1.0)
        self.assertAlmostEqual(tree_kernel(h1, h2), expected, places=6)


class TestTreeKernelEdgeCases(unittest.TestCase):
    def test_atomic_sim_empty_strings(self):
        self.assertEqual(atomic_sim("", ""), 1.0)
        self.assertEqual(atomic_sim("", "foo"), 0.0)
        self.assertEqual(atomic_sim("foo", ""), 0.0)

    def test_atomic_sim_shorter_than_ngram_size(self):
        # Strings shorter than n=3 fall back to whole-string membership.
        self.assertEqual(atomic_sim("a", "a"), 1.0)
        self.assertEqual(atomic_sim("a", "b"), 0.0)
        self.assertEqual(atomic_sim("ab", "cd"), 0.0)

    def test_self_kernel_is_one_for_every_shape(self):
        shapes = {
            "atomic":     A("foo"),
            "NOT(atomic)": NOT(A("x")),
            "AND(a,b)":   AND(A("a"), A("b")),
            "OR(a,b)":    LogicalNode(name=LogicalOp.OR, inputs=[A("a"), A("b")]),
            "IMPLIES":    IMPLIES(A("a"), A("b")),
            "CONTRADICT": CONTRADICT(A("a"), A("b")),
            "CORRELATE":  CORRELATE(A("a"), A("b")),
            "SUPPORT":    RelationNode(name=RelationType.SUPPORT,
                                       inputs=[A("a"), A("b")]),
        }
        for label, node in shapes.items():
            with self.subTest(shape=label):
                self.assertAlmostEqual(tree_kernel(node, node), 1.0, places=9)

    def test_deep_tree_does_not_recurse_beyond_python_limit(self):
        node = A("leaf")
        for _ in range(100):
            node = NOT(node)
        self.assertAlmostEqual(tree_kernel(node, node), 1.0, places=9)

    def test_atomic_node_rejects_empty_name_at_construction(self):
        # ir.py rejects empty atomic names, so tree_kernel need not guard them.
        with self.assertRaises(ValueError):
            AtomicNode(name="")
        with self.assertRaises(ValueError):
            AtomicNode(name="   ")


class TestSoftJaccardClutter(unittest.TestCase):
    def test_extra_children_lower_similarity(self):
        base = IMPLIES(AND(A("a"), A("b")), A("z"))
        cluttered = IMPLIES(AND(A("a"), A("b"), A("c"), A("d")), A("z"))
        value = tree_kernel(base, cluttered)
        self.assertLess(value, 1.0)
        self.assertGreater(value, 0.5)

    def test_more_clutter_lowers_similarity_further(self):
        base = IMPLIES(AND(A("a"), A("b")), A("z"))
        mildly_cluttered = IMPLIES(AND(A("a"), A("b"), A("c")), A("z"))
        heavily_cluttered = IMPLIES(
            AND(A("a"), A("b"), A("c"), A("d"), A("e"), A("f")),
            A("z"),
        )
        self.assertGreater(
            tree_kernel(base, mildly_cluttered),
            tree_kernel(base, heavily_cluttered),
        )


if __name__ == "__main__":
    unittest.main()
