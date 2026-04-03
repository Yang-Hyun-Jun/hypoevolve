"""Archive storage and parent-sampling utilities for HypoEvolve."""

from __future__ import annotations

import math
import random
from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from elg import Hypothesis, count_nodes, fingerprint


Cell = Tuple[int, int]


@dataclass(slots=True)
class ArchiveEntry:
    """Store one hypothesis candidate and its archive metadata."""

    hypothesis: Hypothesis
    metrics: Dict[str, object]
    fingerprint: str
    iteration: int = 0
    metadata: Dict[str, object] = field(default_factory=dict)
    coverage: float = 0.0
    complexity: int = 0
    cell: Optional[Cell] = None

    @property
    def score(self) -> float:
        if "combined_score" in self.metrics and isinstance(
            self.metrics["combined_score"], (int, float)
        ):
            score = float(self.metrics["combined_score"])
            return score if math.isfinite(score) else 0.0
        numeric = [
            v
            for v in self.metrics.values()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        ]
        if not numeric:
            return 0.0
        score = float(sum(numeric) / len(numeric))
        return score if math.isfinite(score) else 0.0


@dataclass(slots=True)
class SamplingStats:
    """Track parent-selection outcomes for UCB-style sampling."""

    pulls: int = 0
    total_reward: float = 0.0
    last_reward: float = 0.0

    @property
    def mean_reward(self) -> float:
        if self.pulls < 1:
            return 0.0
        return self.total_reward / self.pulls


class MAPElitesArchive:
    """Store top-k elites per cell and sample parents from occupied cells."""

    UCB_EXPLORATION_WEIGHT = 0.01

    def __init__(
        self,
        coverage_bins: Optional[List[float]] = None,
        complexity_bins: Optional[List[int]] = None,
        per_cell_top_k: int = 10,
    ):
        coverage_bins = coverage_bins or [0.05, 0.15, 0.30]
        complexity_bins = complexity_bins or [3, 5, 8]

        if not coverage_bins:
            raise ValueError("coverage_bins must not be empty")
        if not complexity_bins:
            raise ValueError("complexity_bins must not be empty")
        if per_cell_top_k < 1:
            raise ValueError("per_cell_top_k must be >= 1")

        self.coverage_bins = [float(value) for value in coverage_bins]
        self.complexity_bins = [int(value) for value in complexity_bins]
        self.per_cell_top_k = int(per_cell_top_k)
        self._cells: Dict[Cell, List[ArchiveEntry]] = {}
        self._sampling_stats: Dict[str, SamplingStats] = {}

    def __len__(self) -> int:
        return len(self._cells)

    @property
    def entries(self) -> List[ArchiveEntry]:
        return sorted(
            (entry for cell_entries in self._cells.values() for entry in cell_entries),
            key=lambda entry: entry.score,
            reverse=True,
        )

    @property
    def best(self) -> Optional[ArchiveEntry]:
        items = self.entries
        return items[0] if items else None

    def add(
        self,
        hypothesis: Hypothesis,
        metrics: Dict[str, object],
        iteration: int = 0,
        metadata: Optional[Dict[str, object]] = None,
    ) -> ArchiveEntry:
        """Insert a candidate into its cell and keep only the top-k entries."""
        descriptor = self.describe(hypothesis, metrics)
        coverage = descriptor["coverage"]
        complexity = descriptor["complexity"]
        cell = descriptor["cell"]
        entry_metadata = dict(metadata or {})
        entry_metadata["map_elites"] = dict(descriptor["map_elites"])
        new_entry = ArchiveEntry(
            hypothesis=hypothesis,
            metrics=dict(metrics),
            fingerprint=fingerprint(hypothesis),
            iteration=iteration,
            metadata=entry_metadata,
            coverage=coverage,
            complexity=complexity,
            cell=cell,
        )

        cell_entries = list(self._cells.get(cell, []))
        duplicate_index = next(
            (
                index
                for index, entry in enumerate(cell_entries)
                if entry.fingerprint == new_entry.fingerprint
            ),
            None,
        )
        
        if duplicate_index is not None:
            if cell_entries[duplicate_index].score >= new_entry.score:
                return cell_entries[duplicate_index]
            cell_entries.pop(duplicate_index)

        cell_entries.append(new_entry)
        cell_entries.sort(key=lambda entry: entry.score, reverse=True)
        self._cells[cell] = cell_entries[: self.per_cell_top_k]
        return next(
            (
                entry
                for entry in self._cells[cell]
                if entry.fingerprint == new_entry.fingerprint
            ),
            self._cells[cell][0],
        )

    def describe(
        self,
        hypothesis: Hypothesis,
        metrics: Dict[str, object],
    ) -> Dict[str, object]:
        """Compute the MAP-Elites descriptor for a hypothesis/metric pair."""
        coverage = _coerce_coverage(metrics.get("coverage"))
        complexity = count_nodes(hypothesis)
        cell = (
            coverage_bin(coverage, self.coverage_bins),
            complexity_bin(complexity, self.complexity_bins),
        )
        
        return {
            "coverage": coverage,
            "complexity": complexity,
            "cell": cell,
            "map_elites": {
                "coverage": coverage,
                "complexity": complexity,
                "coverage_bin": cell[0],
                "complexity_bin": cell[1],
            },
        }

    def sample_parent(
        self,
        rng: Optional[random.Random] = None,
    ) -> ArchiveEntry:
        """Sample one parent by uniform cell choice and within-cell UCB."""
        if not self._cells:
            raise ValueError("Cannot sample from an empty archive")
        chooser = rng or random.Random()
        occupied_cells = sorted(self._cells)
        selected_cell = chooser.choice(occupied_cells)
        candidates = self._cells[selected_cell]
        selected = max(
            candidates,
            key=lambda entry: (
                self._ucb_score(entry, candidates),
                entry.score,
            ),
        )
        self._stats_for(selected.fingerprint).pulls += 1
        return selected

    def record_parent_outcome(self, fingerprint: str, reward: float) -> None:
        """Record the observed reward after sampling an entry as a parent."""
        stats = self._stats_for(fingerprint)
        numeric_reward = float(reward) if math.isfinite(float(reward)) else 0.0
        stats.total_reward += numeric_reward
        stats.last_reward = numeric_reward

    def sampling_stats(self, fingerprint: str | None = None) -> Dict[str, object]:
        """Return aggregated sampling statistics for one entry or all entries."""
        if fingerprint is not None:
            stats = self._stats_for(fingerprint)
            return _sampling_stats_dict(stats)
        return {
            key: _sampling_stats_dict(stats)
            for key, stats in sorted(self._sampling_stats.items())
        }

    def occupancy_stats(self) -> Dict[str, object]:
        coverage_counts = [0] * (len(self.coverage_bins) + 1)
        complexity_counts = [0] * (len(self.complexity_bins) + 1)
        for cell_entries in self._cells.values():
            if not cell_entries:
                continue
            entry = cell_entries[0]
            coverage_counts[entry.cell[0]] += 1
            complexity_counts[entry.cell[1]] += 1
        return {
            "occupied_cells": len(self._cells),
            "coverage_counts": coverage_counts,
            "complexity_counts": complexity_counts,
        }

    def occupancy_summary(self) -> str:
        stats = self.occupancy_stats()
        coverage_text = ",".join(str(value) for value in stats["coverage_counts"])
        complexity_text = ",".join(
            str(value) for value in stats["complexity_counts"]
        )
        return f"cells={stats['occupied_cells']} cov=[{coverage_text}] cmp=[{complexity_text}]"

    def snapshot(self) -> List[Dict[str, object]]:
        return [
            {
                "fingerprint": entry.fingerprint,
                "iteration": entry.iteration,
                "metrics": dict(entry.metrics),
                "hypothesis": entry.hypothesis.to_dict(),
                "metadata": dict(entry.metadata),
                "coverage": entry.coverage,
                "complexity": entry.complexity,
                "cell": list(entry.cell) if entry.cell is not None else None,
            }
            for entry in self.entries
        ]

    def _stats_for(self, fingerprint_value: str) -> SamplingStats:
        return self._sampling_stats.setdefault(fingerprint_value, SamplingStats())

    def _ucb_score(
        self,
        entry: ArchiveEntry,
        candidates: List[ArchiveEntry],
    ) -> float:
        stats = self._stats_for(entry.fingerprint)
        if stats.pulls == 0:
            return float("inf")

        total_pulls = sum(self._stats_for(candidate.fingerprint).pulls for candidate in candidates)
        if total_pulls <= 1:
            return stats.mean_reward

        exploration_bonus = self.UCB_EXPLORATION_WEIGHT * math.sqrt(
            math.log(total_pulls) / stats.pulls
        )
        return stats.mean_reward + exploration_bonus

def coverage_bin(coverage: float, bins: List[float]) -> int:
    """Map a coverage value to its coverage-bin index."""
    return bisect_right(bins, coverage)


def complexity_bin(complexity: int, bins: List[int]) -> int:
    """Map a complexity value to its complexity-bin index."""
    return bisect_left(bins, complexity)


def _coerce_coverage(value: object) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return 0.0
    numeric = float(value)
    if not math.isfinite(numeric):
        return 0.0
    return min(1.0, max(0.0, numeric))


def _sampling_stats_dict(stats: SamplingStats) -> Dict[str, float | int]:
    return {
        "pulls": stats.pulls,
        "total_reward": stats.total_reward,
        "mean_reward": stats.mean_reward,
        "last_reward": stats.last_reward,
    }
