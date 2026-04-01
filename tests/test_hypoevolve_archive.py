import random
import unittest

from elg import AtomicNode, Hypothesis, LogicalNode, RelationNode
from hypoevolve.archive import (
    MAPElitesArchive,
    complexity_bin,
    coverage_bin,
)


class TestHypoEvolveArchive(unittest.TestCase):
    def setUp(self):
        self.archive = MAPElitesArchive(
            coverage_bins=[0.05, 0.15, 0.30],
            complexity_bins=[3, 5, 8],
        )

    def test_bin_helpers_match_expected_boundaries(self):
        self.assertEqual(coverage_bin(0.00, self.archive.coverage_bins), 0)
        self.assertEqual(coverage_bin(0.05, self.archive.coverage_bins), 1)
        self.assertEqual(coverage_bin(0.30, self.archive.coverage_bins), 3)
        self.assertEqual(complexity_bin(3, self.archive.complexity_bins), 0)
        self.assertEqual(complexity_bin(4, self.archive.complexity_bins), 1)
        self.assertEqual(complexity_bin(9, self.archive.complexity_bins), 3)

    def test_archive_keeps_best_entry_per_cell(self):
        cell_hypothesis = Hypothesis(root=AtomicNode("A"))
        self.archive.add(cell_hypothesis, {"combined_score": 0.2, "coverage": 0.04})
        self.archive.add(cell_hypothesis, {"combined_score": 0.8, "coverage": 0.04})
        self.assertEqual(len(self.archive), 1)
        self.assertEqual(self.archive.best.score, 0.8)
        self.assertEqual(self.archive.best.cell, (0, 0))

    def test_archive_keeps_distinct_cells_even_when_score_is_lower(self):
        simple = Hypothesis(root=AtomicNode("A"))
        complex_hypothesis = Hypothesis(
            root=RelationNode(
                "IMPLIES",
                [
                    LogicalNode("AND", [AtomicNode("A"), AtomicNode("B")]),
                    AtomicNode("C"),
                ],
            )
        )
        self.archive.add(simple, {"combined_score": 0.9, "coverage": 0.04})
        self.archive.add(complex_hypothesis, {"combined_score": 0.3, "coverage": 0.20})
        self.assertEqual(len(self.archive), 2)
        self.assertEqual({entry.cell for entry in self.archive.entries}, {(0, 0), (2, 1)})

    def test_score_weighted_best_property_still_tracks_global_best(self):
        low = Hypothesis(root=AtomicNode("LOW"))
        high = Hypothesis(root=AtomicNode("HIGH"))
        self.archive.add(low, {"combined_score": 0.1, "coverage": 0.04})
        self.archive.add(high, {"combined_score": 0.9, "coverage": 0.20})
        self.assertEqual(self.archive.best.hypothesis.root.name, "HIGH")

    def test_parent_sampling_is_seeded_over_occupied_cells(self):
        first = Hypothesis(root=AtomicNode("A"))
        second = Hypothesis(
            root=RelationNode(
                "IMPLIES",
                [
                    LogicalNode("AND", [AtomicNode("A"), AtomicNode("B")]),
                    AtomicNode("C"),
                ],
            )
        )
        self.archive.add(first, {"combined_score": 0.1, "coverage": 0.04})
        self.archive.add(second, {"combined_score": 0.9, "coverage": 0.20})
        first_pick = self.archive.sample_parent(random.Random(11)).fingerprint
        second_pick = self.archive.sample_parent(random.Random(11)).fingerprint
        self.assertEqual(first_pick, second_pick)

    def test_uniform_parent_sampling_visits_multiple_cells(self):
        entries = [
            (Hypothesis(root=AtomicNode("LOW")), {"combined_score": 0.1, "coverage": 0.04}),
            (
                Hypothesis(
                    root=RelationNode(
                        "IMPLIES",
                        [
                            LogicalNode("AND", [AtomicNode("A"), AtomicNode("B")]),
                            AtomicNode("C"),
                        ],
                    )
                ),
                {"combined_score": 0.9, "coverage": 0.20},
            ),
        ]
        for hypothesis, metrics in entries:
            self.archive.add(hypothesis, metrics)
        rng = random.Random(0)
        counts = {entry.cell: 0 for entry in self.archive.entries}
        for _ in range(1000):
            picked = self.archive.sample_parent(rng)
            counts[picked.cell] += 1
        self.assertTrue(all(count > 0 for count in counts.values()))

    def test_non_finite_or_missing_coverage_coerces_to_zero_bin(self):
        bad = Hypothesis(root=AtomicNode("BAD"))
        missing = Hypothesis(
            root=RelationNode(
                "IMPLIES",
                [
                    LogicalNode("AND", [AtomicNode("A"), AtomicNode("B")]),
                    AtomicNode("C"),
                ],
            )
        )
        self.archive.add(bad, {"combined_score": 0.1, "coverage": float("nan")})
        self.archive.add(missing, {"combined_score": 0.2})
        self.assertEqual(self.archive.entries[0].cell[0], 0)
        self.assertEqual(self.archive.entries[1].cell[0], 0)

    def test_snapshot_includes_map_elites_metadata(self):
        hypothesis = Hypothesis(root=AtomicNode("A"))
        self.archive.add(hypothesis, {"combined_score": 0.2, "coverage": 0.04})
        snapshot = self.archive.snapshot()
        self.assertEqual(snapshot[0]["cell"], [0, 0])
        self.assertEqual(snapshot[0]["metadata"]["map_elites"]["coverage_bin"], 0)
        self.assertEqual(snapshot[0]["metadata"]["map_elites"]["complexity_bin"], 0)

    def test_occupancy_stats_report_per_axis_counts(self):
        self.archive.add(Hypothesis(root=AtomicNode("A")), {"combined_score": 0.2, "coverage": 0.04})
        self.archive.add(
            Hypothesis(
                root=RelationNode(
                    "IMPLIES",
                    [
                        LogicalNode("AND", [AtomicNode("A"), AtomicNode("B")]),
                        AtomicNode("C"),
                    ],
                )
            ),
            {"combined_score": 0.3, "coverage": 0.20},
        )
        stats = self.archive.occupancy_stats()
        self.assertEqual(stats["occupied_cells"], 2)
        self.assertEqual(stats["coverage_counts"], [1, 0, 1, 0])
        self.assertEqual(stats["complexity_counts"], [1, 1, 0, 0])


if __name__ == "__main__":
    unittest.main()
