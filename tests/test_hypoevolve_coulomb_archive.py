import math
import pickle
import random
import unittest

from hypoevolve.elg import (
    AtomicNode,
    Hypothesis,
    LogicalNode,
    LogicalOp,
    RelationNode,
    RelationType,
    fingerprint,
)
from hypoevolve.memory.coulomb_archive import CoulombArchive, CoulombDescriptor


def A(name: str) -> AtomicNode:
    return AtomicNode(name=name)


def AND(*children):
    return LogicalNode(name=LogicalOp.AND, inputs=list(children))


def IMPLIES(cond, target):
    return RelationNode(name=RelationType.IMPLIES, inputs=[cond, target])


def wrap(node) -> Hypothesis:
    return Hypothesis(root=node)


def make_metrics(score: float, coverage: float = 0.2) -> dict:
    return {"combined_score": score, "coverage": coverage}


class TestCoulombArchiveInit(unittest.TestCase):
    def test_defaults_produce_empty_archive(self):
        arc = CoulombArchive()
        self.assertEqual(len(arc), 0)
        self.assertIsNone(arc.best)
        self.assertEqual(arc.entries, [])

    def test_invalid_arguments_raise(self):
        with self.assertRaises(ValueError):
            CoulombArchive(capacity=0)
        with self.assertRaises(ValueError):
            CoulombArchive(gamma=-0.1)
        with self.assertRaises(ValueError):
            CoulombArchive(eps=0.0)


class TestCoulombArchiveAdmission(unittest.TestCase):
    def setUp(self):
        self.arc = CoulombArchive(capacity=3, gamma=0.3)
        self.h_a = wrap(IMPLIES(A("alpha"), A("beta")))
        self.h_b = wrap(IMPLIES(A("gamma"), A("delta")))
        self.h_c = wrap(IMPLIES(A("epsilon"), A("zeta")))

    def test_add_fills_up_to_capacity(self):
        self.arc.add(self.h_a, make_metrics(0.9))
        self.arc.add(self.h_b, make_metrics(0.5))
        self.arc.add(self.h_c, make_metrics(0.7))
        self.assertEqual(len(self.arc), 3)

    def test_add_returns_archive_entry_with_metadata(self):
        entry = self.arc.add(self.h_a, make_metrics(0.9), metadata={"name": "alpha"})
        self.assertEqual(entry.metadata["name"], "alpha")
        self.assertIn("coulomb", entry.metadata)
        self.assertEqual(entry.fingerprint, fingerprint(self.h_a))

    def test_best_reflects_highest_scoring_entry(self):
        self.arc.add(self.h_a, make_metrics(0.5))
        self.arc.add(self.h_b, make_metrics(0.9))
        self.assertEqual(self.arc.best.fingerprint, fingerprint(self.h_b))

    def test_duplicate_fingerprint_keeps_higher_score(self):
        self.arc.add(self.h_a, make_metrics(0.5))
        self.arc.add(self.h_a, make_metrics(0.9))
        self.assertEqual(len(self.arc), 1)
        self.assertAlmostEqual(self.arc.best.score, 0.9)


class TestCoulombArchiveEviction(unittest.TestCase):
    def test_near_duplicate_pair_leaves_only_one_member(self):
        arc = CoulombArchive(capacity=2, gamma=0.3)
        clean = wrap(IMPLIES(A("VIX above 30"), A("SPY drops")))
        cluttered = wrap(
            IMPLIES(
                AND(A("VIX above 30"), A("if weekday"), A("if no news")),
                A("SPY drops"),
            )
        )
        distinct = wrap(IMPLIES(A("unrelated concept"), A("other outcome")))
        arc.add(clean, make_metrics(0.9))
        arc.add(cluttered, make_metrics(0.85))
        arc.add(distinct, make_metrics(0.6))
        fingerprints = {entry.fingerprint for entry in arc.entries}
        self.assertIn(fingerprint(clean), fingerprints)
        self.assertIn(fingerprint(distinct), fingerprints)
        self.assertNotIn(fingerprint(cluttered), fingerprints)

    def test_full_archive_rejects_weaker_offering(self):
        arc = CoulombArchive(capacity=2, gamma=0.5)
        arc.add(wrap(IMPLIES(A("alpha"), A("beta"))), make_metrics(0.9))
        arc.add(wrap(IMPLIES(A("gamma"), A("delta"))), make_metrics(0.8))
        arc.add(wrap(IMPLIES(A("weak"), A("weak"))), make_metrics(0.1))
        self.assertEqual(len(arc), 2)

    def test_high_score_but_clustered_candidate_can_lose_to_lower_score_diverse_incumbent(self):
        """Core eviction rule: Δ = score - γ·U governs replacement, not raw score.

        A strong candidate that lands very close to an existing archive member
        (small distance → huge repulsive potential) can be beaten by a lower
        scoring but well-isolated incumbent when γ is high enough.
        """
        anchor = wrap(IMPLIES(A("VIX above 30"), A("SPY drops")))
        diverse = wrap(IMPLIES(A("unrelated concept"), A("other outcome")))
        near_dup_high_score = wrap(
            IMPLIES(A("VIX above 30"), A("SPY drops next day"))
        )
        arc = CoulombArchive(capacity=2, gamma=8.0, eps=0.01)
        arc.add(anchor, make_metrics(0.6))
        arc.add(diverse, make_metrics(0.3))
        # Even though the newcomer scores 0.85 (higher than the diverse incumbent's 0.3),
        # its distance to anchor is tiny so its Δ is dominated by γ·U.
        arc.add(near_dup_high_score, make_metrics(0.85))
        remaining = {entry.fingerprint for entry in arc.entries}
        self.assertIn(fingerprint(diverse), remaining)
        self.assertNotIn(fingerprint(near_dup_high_score), remaining)


class TestCoulombArchiveDescribeAndSnapshot(unittest.TestCase):
    def test_describe_returns_coulomb_descriptor(self):
        arc = CoulombArchive(capacity=4)
        h = wrap(IMPLIES(A("a"), A("b")))
        descriptor = arc.describe(h, make_metrics(0.7, coverage=0.42))
        self.assertIn("coulomb", descriptor)
        self.assertAlmostEqual(descriptor["coverage"], 0.42)
        coulomb = descriptor["coulomb"]
        self.assertIn("potential", coulomb)
        self.assertIn("quality", coulomb)
        self.assertAlmostEqual(coulomb["quality"], 0.7)
        self.assertGreaterEqual(coulomb["complexity"], 3)

    def test_snapshot_serializes_entries(self):
        arc = CoulombArchive(capacity=2)
        arc.add(wrap(IMPLIES(A("a"), A("b"))), make_metrics(0.9))
        arc.add(wrap(IMPLIES(A("c"), A("d"))), make_metrics(0.6))
        snap = arc.snapshot()
        self.assertEqual(len(snap), 2)
        for row in snap:
            self.assertIn("fingerprint", row)
            self.assertIn("hypothesis", row)


class TestCoulombArchiveSampling(unittest.TestCase):
    def test_sample_parent_returns_entry(self):
        arc = CoulombArchive(capacity=3)
        arc.add(wrap(IMPLIES(A("a"), A("b"))), make_metrics(0.9))
        arc.add(wrap(IMPLIES(A("c"), A("d"))), make_metrics(0.6))
        rng = random.Random(0)
        entry = arc.sample_parent(rng)
        self.assertIn(entry.fingerprint, {row.fingerprint for row in arc.entries})

    def test_sample_parent_raises_on_empty_archive(self):
        arc = CoulombArchive()
        with self.assertRaises(ValueError):
            arc.sample_parent(random.Random(0))

    def test_sampling_favors_higher_scores(self):
        arc = CoulombArchive(capacity=2, gamma=0.0)  # no repulsion → pure quality
        arc.add(wrap(IMPLIES(A("hi"), A("target"))), make_metrics(0.99))
        arc.add(wrap(IMPLIES(A("lo"), A("target"))), make_metrics(0.01))
        counts = {entry.fingerprint: 0 for entry in arc.entries}
        rng = random.Random(0)
        for _ in range(400):
            counts[arc.sample_parent(rng).fingerprint] += 1
        higher = arc.entries[0].fingerprint
        lower = arc.entries[1].fingerprint
        self.assertGreater(counts[higher], counts[lower])

    def test_record_parent_outcome_updates_stats(self):
        arc = CoulombArchive(capacity=2)
        arc.add(wrap(IMPLIES(A("a"), A("b"))), make_metrics(0.9))
        fp = arc.entries[0].fingerprint
        arc.record_parent_outcome(fp, 0.25)
        stats = arc.sampling_stats(fp)
        self.assertEqual(stats["last_reward"], 0.25)


class TestCoulombNumericalStability(unittest.TestCase):
    def test_gamma_zero_sampling_is_score_proportional(self):
        arc = CoulombArchive(capacity=2, gamma=0.0)
        arc.add(wrap(IMPLIES(A("hi"), A("target"))), make_metrics(0.99))
        arc.add(wrap(IMPLIES(A("lo"), A("target"))), make_metrics(0.01))
        counts = {entry.fingerprint: 0 for entry in arc.entries}
        rng = random.Random(0)
        for _ in range(500):
            counts[arc.sample_parent(rng).fingerprint] += 1
        high_fp = arc.entries[0].fingerprint
        self.assertGreater(counts[high_fp], 450)

    def test_gamma_extreme_does_not_return_nan_or_inf(self):
        arc = CoulombArchive(capacity=3, gamma=100.0)
        arc.add(wrap(IMPLIES(A("alpha"), A("beta"))), make_metrics(0.9))
        arc.add(wrap(IMPLIES(A("gamma"), A("delta"))), make_metrics(0.6))
        rng = random.Random(0)
        for _ in range(20):
            entry = arc.sample_parent(rng)
            self.assertIsNotNone(entry)

    def test_all_zero_scores_do_not_crash_sampling(self):
        arc = CoulombArchive(capacity=3, gamma=0.3)
        for name in ("a", "b", "c"):
            arc.add(wrap(IMPLIES(A(name), A(name + "'"))), make_metrics(0.0))
        rng = random.Random(0)
        for _ in range(5):
            entry = arc.sample_parent(rng)
            self.assertIn(entry.fingerprint, {row.fingerprint for row in arc.entries})

    def test_capacity_one_keeps_the_stronger_offer(self):
        arc = CoulombArchive(capacity=1, gamma=0.3)
        arc.add(wrap(IMPLIES(A("first"), A("b"))), make_metrics(0.5))
        arc.add(wrap(IMPLIES(A("second"), A("d"))), make_metrics(0.9))
        arc.add(wrap(IMPLIES(A("third"), A("f"))), make_metrics(0.6))
        self.assertEqual(len(arc), 1)
        self.assertAlmostEqual(arc.best.score, 0.9)

    def test_distance_one_pairs_stay_finite(self):
        # IMPLIES vs CONTRADICT with identical atoms → tree_distance = 1.0.
        arc = CoulombArchive(capacity=3, gamma=0.3, eps=1e-2)
        arc.add(wrap(IMPLIES(A("alpha"), A("beta"))), make_metrics(0.7))
        arc.add(
            wrap(RelationNode(name=RelationType.CONTRADICT,
                              inputs=[A("alpha"), A("beta")])),
            make_metrics(0.6),
        )
        self.assertEqual(len(arc), 2)
        for slot in range(len(arc)):
            self.assertTrue(math.isfinite(arc._potential_at_slot(slot)))

    def test_empty_archive_summary_and_snapshot_are_safe(self):
        arc = CoulombArchive(capacity=5, gamma=0.3)
        self.assertIn("size=0/5", arc.occupancy_summary())
        self.assertEqual(arc.snapshot(), [])
        stats = arc.occupancy_stats()
        self.assertEqual(stats["size"], 0)
        self.assertEqual(stats["capacity"], 5)


class TestCoulombPickleSafety(unittest.TestCase):
    def test_archive_round_trip_preserves_state_and_stays_usable(self):
        arc = CoulombArchive(capacity=4, gamma=0.4, eps=0.02)
        arc.add(wrap(IMPLIES(A("a"), A("b"))), {"combined_score": 0.9, "coverage": 0.2})
        arc.add(wrap(IMPLIES(A("c"), A("d"))), {"combined_score": 0.6, "coverage": 0.15})

        restored = pickle.loads(pickle.dumps(arc))

        self.assertEqual(restored.capacity, arc.capacity)
        self.assertAlmostEqual(restored.gamma, arc.gamma)
        self.assertAlmostEqual(restored.eps, arc.eps)
        self.assertEqual(len(restored), len(arc))
        self.assertAlmostEqual(restored.best.score, arc.best.score)

        restored.add(
            wrap(IMPLIES(A("new"), A("target"))),
            {"combined_score": 0.7, "coverage": 0.1},
        )
        self.assertEqual(len(restored), 3)


class TestCoulombDescriptor(unittest.TestCase):
    def test_descriptor_to_dict_is_serializable(self):
        d = CoulombDescriptor(potential=1.25, quality=0.8, complexity=5)
        payload = d.to_dict()
        self.assertEqual(payload, {"potential": 1.25, "quality": 0.8, "complexity": 5})


class TestOccupancySummary(unittest.TestCase):
    def test_summary_reports_size_and_metrics(self):
        arc = CoulombArchive(capacity=3, gamma=0.3)
        arc.add(wrap(IMPLIES(A("alpha"), A("beta"))), make_metrics(0.9))
        arc.add(wrap(IMPLIES(A("gamma"), A("delta"))), make_metrics(0.6))
        summary = arc.occupancy_summary()
        self.assertIn("size=2/3", summary)
        stats = arc.occupancy_stats()
        self.assertEqual(stats["size"], 2)
        self.assertEqual(stats["capacity"], 3)
        self.assertGreater(stats["mean_pairwise_distance"], 0.0)


if __name__ == "__main__":
    unittest.main()
