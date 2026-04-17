import random
import unittest

from elg import AtomicNode, Hypothesis, LogicalNode, RelationNode
from hypoevolve.archive import (
    ArchiveEntry,
    MAPElitesArchive,
    SamplingStats,
    _coerce_coverage,
    _sampling_stats_dict,
    complexity_bin,
    coverage_bin,
)


class TestHypoEvolveArchive(unittest.TestCase):
    def setUp(self):
        self.archive = MAPElitesArchive(
            coverage_bins=[0.05, 0.15, 0.30],
            complexity_bins=[3, 5, 8],
        )


    def test_archive_entry_score_falls_back_for_non_finite_and_missing_combined_score(self):
        hypothesis = Hypothesis(root=AtomicNode('A'))
        non_finite = ArchiveEntry(hypothesis=hypothesis, metrics={'combined_score': float('nan')}, fingerprint='fp1')
        averaged = ArchiveEntry(hypothesis=hypothesis, metrics={'precision': 0.2, 'coverage': 0.4}, fingerprint='fp2')
        empty = ArchiveEntry(hypothesis=hypothesis, metrics={}, fingerprint='fp3')

        self.assertEqual(non_finite.score, 0.0)
        self.assertAlmostEqual(averaged.score, 0.3)
        self.assertEqual(empty.score, 0.0)

    def test_archive_init_validates_configuration(self):
        with self.assertRaisesRegex(ValueError, 'per_cell_top_k must be >= 1'):
            MAPElitesArchive(per_cell_top_k=0)
        with self.assertRaisesRegex(ValueError, 'parent_sampling_mode'):
            MAPElitesArchive(parent_sampling_mode='epsilon_greedy')

    def test_archive_init_uses_default_bins_when_optional_lists_are_falsy(self):
        archive = MAPElitesArchive(coverage_bins=[], complexity_bins=[])
        self.assertEqual(archive.coverage_bins, [0.05, 0.15, 0.3])
        self.assertEqual(archive.complexity_bins, [3, 5, 8])

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

    def test_archive_keeps_top_k_entries_per_cell(self):
        archive = MAPElitesArchive(per_cell_top_k=3)
        scores = [0.2, 0.8, 0.5, 0.6]
        for score in scores:
            hypothesis = Hypothesis(root=AtomicNode(f"A_{score}"))
            archive.add(hypothesis, {"combined_score": score, "coverage": 0.04})
        self.assertEqual(len(archive), 1)
        self.assertEqual([entry.score for entry in archive.entries], [0.8, 0.6, 0.5])

    def test_archive_deduplicates_by_fingerprint_within_cell(self):
        hypothesis = Hypothesis(root=AtomicNode("A"))
        self.archive.add(hypothesis, {"combined_score": 0.2, "coverage": 0.04})
        self.archive.add(hypothesis, {"combined_score": 0.1, "coverage": 0.04})
        self.archive.add(hypothesis, {"combined_score": 0.7, "coverage": 0.04})
        self.assertEqual(len(self.archive.entries), 1)
        self.assertEqual(self.archive.entries[0].score, 0.7)

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

    def test_ucb_sampling_explores_unpulled_entries_before_revisiting(self):
        archive = MAPElitesArchive(per_cell_top_k=3)
        for score in (0.9, 0.6, 0.3):
            archive.add(
                Hypothesis(root=AtomicNode(f"A_{score}")),
                {"combined_score": score, "coverage": 0.04},
            )
        rng = random.Random(0)
        picked_names = [
            archive.sample_parent(rng).hypothesis.root.name,
            archive.sample_parent(rng).hypothesis.root.name,
            archive.sample_parent(rng).hypothesis.root.name,
        ]
        self.assertEqual(picked_names, ["A_0.9", "A_0.6", "A_0.3"])

    def test_ucb_sampling_uses_recorded_reward_stats(self):
        archive = MAPElitesArchive(per_cell_top_k=3)
        for score in (0.9, 0.6, 0.3):
            archive.add(
                Hypothesis(root=AtomicNode(f"A_{score}")),
                {"combined_score": score, "coverage": 0.04},
            )

        first = archive.sample_parent(random.Random(0))
        archive.record_parent_outcome(first.fingerprint, -0.4)
        second = archive.sample_parent(random.Random(0))
        archive.record_parent_outcome(second.fingerprint, 0.3)
        third = archive.sample_parent(random.Random(0))
        archive.record_parent_outcome(third.fingerprint, -0.1)

        picked = archive.sample_parent(random.Random(0))
        self.assertEqual(picked.hypothesis.root.name, "A_0.6")

    def test_random_parent_sampling_mode_samples_uniformly_over_entries(self):
        archive = MAPElitesArchive(per_cell_top_k=3, parent_sampling_mode="random")
        for score in (0.9, 0.6, 0.3):
            archive.add(
                Hypothesis(root=AtomicNode(f"A_{score}")),
                {"combined_score": score, "coverage": 0.04},
            )

        rng = random.Random(0)
        picked_names = [
            archive.sample_parent(rng).hypothesis.root.name,
            archive.sample_parent(rng).hypothesis.root.name,
            archive.sample_parent(rng).hypothesis.root.name,
        ]

        self.assertEqual(picked_names, ["A_0.6", "A_0.6", "A_0.9"])

    def test_sampling_stats_report_pulls_and_rewards(self):
        entry = self.archive.add(
            Hypothesis(root=AtomicNode("A")),
            {"combined_score": 0.2, "coverage": 0.04},
        )
        self.archive.sample_parent(random.Random(0))
        self.archive.record_parent_outcome(entry.fingerprint, 0.25)
        stats = self.archive.sampling_stats(entry.fingerprint)
        self.assertEqual(stats["pulls"], 1)
        self.assertEqual(stats["total_reward"], 0.25)
        self.assertEqual(stats["mean_reward"], 0.25)
        self.assertEqual(stats["last_reward"], 0.25)

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
        self.archive.add(Hypothesis(root=AtomicNode("A")), {"combined_score": 0.2, "coverage": 0.04})
        self.archive.add(Hypothesis(root=AtomicNode("B")), {"combined_score": 0.1, "coverage": 0.04})
        snapshot = self.archive.snapshot()
        self.assertEqual(snapshot[0]["cell"], [0, 0])
        self.assertEqual(snapshot[0]["metadata"]["map_elites"]["coverage_bin"], 0)
        self.assertEqual(snapshot[0]["metadata"]["map_elites"]["complexity_bin"], 0)
        self.assertEqual(len(snapshot), 2)

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

    def test_snapshot_matches_golden_shape_for_multi_cell_archive(self):
        archive = MAPElitesArchive(per_cell_top_k=2)
        archive.add(
            Hypothesis(root=AtomicNode("A")),
            {"combined_score": 0.2, "coverage": 0.04},
            iteration=1,
            metadata={"source": "seed"},
        )
        archive.add(
            Hypothesis(root=AtomicNode("B")),
            {"combined_score": 0.1, "coverage": 0.04},
            iteration=2,
            metadata={"source": "mut"},
        )
        archive.add(
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
            iteration=3,
            metadata={"source": "mut"},
        )

        self.assertEqual(
            archive.snapshot(),
            [
                {
                    "fingerprint": "d3ffd0fcd85eb1b196fb874682a2617ebc2c25df735b7cd2b2bf35924a963e9f",
                    "iteration": 3,
                    "metrics": {"combined_score": 0.3, "coverage": 0.2},
                    "hypothesis": {
                        "root": {
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
                        }
                    },
                    "metadata": {
                        "source": "mut",
                        "map_elites": {
                            "coverage": 0.2,
                            "complexity": 5,
                            "coverage_bin": 2,
                            "complexity_bin": 1,
                        },
                    },
                    "coverage": 0.2,
                    "complexity": 5,
                    "cell": [2, 1],
                },
                {
                    "fingerprint": "d8240f708d9f80c2a4394f80bf7ec9f27d2ed7d3a6dfd4a272d3e6ff6c6926f7",
                    "iteration": 1,
                    "metrics": {"combined_score": 0.2, "coverage": 0.04},
                    "hypothesis": {"root": {"kind": "atomic", "name": "A"}},
                    "metadata": {
                        "source": "seed",
                        "map_elites": {
                            "coverage": 0.04,
                            "complexity": 1,
                            "coverage_bin": 0,
                            "complexity_bin": 0,
                        },
                    },
                    "coverage": 0.04,
                    "complexity": 1,
                    "cell": [0, 0],
                },
                {
                    "fingerprint": "19ddee656473804aee4c65d5b361ae905904e4f0bfea94458f483c5e46277a3d",
                    "iteration": 2,
                    "metrics": {"combined_score": 0.1, "coverage": 0.04},
                    "hypothesis": {"root": {"kind": "atomic", "name": "B"}},
                    "metadata": {
                        "source": "mut",
                        "map_elites": {
                            "coverage": 0.04,
                            "complexity": 1,
                            "coverage_bin": 0,
                            "complexity_bin": 0,
                        },
                    },
                    "coverage": 0.04,
                    "complexity": 1,
                    "cell": [0, 0],
                },
            ],
        )
        self.assertEqual(archive.occupancy_summary(), "cells=2 cov=[1,0,1,0] cmp=[1,1,0,0]")


    def test_ucb_score_prefers_unpulled_entries_and_then_uses_bonus(self):
        archive = MAPElitesArchive(per_cell_top_k=2)
        high = archive.add(Hypothesis(root=AtomicNode("HIGH")), {"combined_score": 0.9, "coverage": 0.04})
        low = archive.add(Hypothesis(root=AtomicNode("LOW")), {"combined_score": 0.3, "coverage": 0.04})

        self.assertEqual(archive._ucb_score(high, [high, low]), float("inf"))

        archive._stats_for(high.fingerprint).pulls = 4
        archive._stats_for(high.fingerprint).total_reward = 1.0
        archive._stats_for(low.fingerprint).pulls = 2
        archive._stats_for(low.fingerprint).total_reward = 1.0

        self.assertGreater(archive._ucb_score(low, [high, low]), archive._ucb_score(high, [high, low]))

    def test_coerce_coverage_clamps_invalid_and_out_of_range_values(self):
        self.assertEqual(_coerce_coverage(True), 0.0)
        self.assertEqual(_coerce_coverage(None), 0.0)
        self.assertEqual(_coerce_coverage(float('nan')), 0.0)
        self.assertEqual(_coerce_coverage(-1.5), 0.0)
        self.assertEqual(_coerce_coverage(1.5), 1.0)
        self.assertEqual(_coerce_coverage(0.25), 0.25)

    def test_sampling_stats_dict_serializes_sampling_stats_fields(self):
        stats = SamplingStats(pulls=3, total_reward=1.2, last_reward=0.4)
        payload = _sampling_stats_dict(stats)
        self.assertEqual(payload['pulls'], 3)
        self.assertEqual(payload['total_reward'], 1.2)
        self.assertAlmostEqual(payload['mean_reward'], 0.4)
        self.assertEqual(payload['last_reward'], 0.4)


if __name__ == "__main__":
    unittest.main()
