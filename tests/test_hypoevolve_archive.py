import random
import unittest

from elg import AtomicNode, Hypothesis
from hypoevolve.archive import Archive


class TestHypoEvolveArchive(unittest.TestCase):
    def test_archive_dedup_and_best_tracking(self):
        archive = Archive(top_k=5)
        hypothesis = Hypothesis(root=AtomicNode('A'))
        archive.add(hypothesis, {'combined_score': 0.2})
        archive.add(hypothesis, {'combined_score': 0.8})
        self.assertEqual(len(archive), 1)
        self.assertEqual(archive.best.score, 0.8)

    def test_archive_is_capped_at_five(self):
        archive = Archive(top_k=5)
        for idx in range(7):
            archive.add(Hypothesis(root=AtomicNode(f'A{idx}')), {'combined_score': float(idx)})
        self.assertEqual(len(archive), 5)
        self.assertEqual(archive.best.score, 6.0)

    def test_archive_add_does_not_crash_when_new_entry_is_immediately_trimmed(self):
        archive = Archive(top_k=2)
        archive.add(Hypothesis(root=AtomicNode('A')), {'combined_score': 2.0})
        archive.add(Hypothesis(root=AtomicNode('B')), {'combined_score': 1.0})
        returned = archive.add(Hypothesis(root=AtomicNode('C')), {'combined_score': -1.0})
        self.assertEqual(len(archive), 2)
        self.assertEqual(returned.hypothesis.root.name, 'C')

    def test_score_weighted_parent_selection_is_seeded(self):
        archive = Archive(top_k=5)
        for idx, score in enumerate([0.1, 0.2, 0.3, 0.4, 0.5]):
            archive.add(Hypothesis(root=AtomicNode(f'A{idx}')), {'combined_score': score})
        first = archive.sample_parent(random.Random(11)).fingerprint
        second = archive.sample_parent(random.Random(11)).fingerprint
        self.assertEqual(first, second)

    def test_higher_score_has_higher_weight(self):
        archive = Archive(top_k=5)
        low = Hypothesis(root=AtomicNode('LOW'))
        high = Hypothesis(root=AtomicNode('HIGH'))
        archive.add(low, {'combined_score': 0.1})
        archive.add(high, {'combined_score': 0.9})
        rng = random.Random(0)
        counts = {'LOW': 0, 'HIGH': 0}
        for _ in range(1000):
            picked = archive.sample_parent(rng)
            counts[picked.hypothesis.root.name] += 1
        self.assertGreater(counts['HIGH'], counts['LOW'])

    def test_explore_prob_can_sample_lower_scored_parent(self):
        archive = Archive(top_k=5)
        low = Hypothesis(root=AtomicNode('LOW'))
        high = Hypothesis(root=AtomicNode('HIGH'))
        archive.add(low, {'combined_score': 0.1})
        archive.add(high, {'combined_score': 0.9})
        rng = random.Random(0)
        counts = {'LOW': 0, 'HIGH': 0}
        for _ in range(1000):
            picked = archive.sample_parent(rng, explore_prob=1.0)
            counts[picked.hypothesis.root.name] += 1
        self.assertGreater(counts['LOW'], 0)
        self.assertGreater(counts['HIGH'], 0)
