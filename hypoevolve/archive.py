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


class MAPElitesArchive:
    def __init__(
        self,
        coverage_bins: Optional[List[float]] = None,
        complexity_bins: Optional[List[int]] = None,
    ):
        coverage_bins = coverage_bins or [0.05, 0.15, 0.30]
        complexity_bins = complexity_bins or [3, 5, 8]
        if not coverage_bins:
            raise ValueError("coverage_bins must not be empty")
        if not complexity_bins:
            raise ValueError("complexity_bins must not be empty")
        self.coverage_bins = [float(value) for value in coverage_bins]
        self.complexity_bins = [int(value) for value in complexity_bins]
        self._cells: Dict[Cell, ArchiveEntry] = {}

    def __len__(self) -> int:
        return len(self._cells)

    @property
    def entries(self) -> List[ArchiveEntry]:
        return sorted(self._cells.values(), key=lambda entry: entry.score, reverse=True)

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

        existing = self._cells.get(cell)
        if existing is None or new_entry.score > existing.score:
            self._cells[cell] = new_entry
            return new_entry
        return existing

    def describe(
        self,
        hypothesis: Hypothesis,
        metrics: Dict[str, object],
    ) -> Dict[str, object]:
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
        ordered = self.entries
        if not ordered:
            raise ValueError("Cannot sample from an empty archive")
        chooser = rng or random.Random()
        occupied_cells = sorted(self._cells)
        selected_cell = chooser.choice(occupied_cells)
        return self._cells[selected_cell]

    def occupancy_stats(self) -> Dict[str, object]:
        coverage_counts = [0] * (len(self.coverage_bins) + 1)
        complexity_counts = [0] * (len(self.complexity_bins) + 1)
        for entry in self._cells.values():
            if entry.cell is None:
                continue
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

def coverage_bin(coverage: float, bins: List[float]) -> int:
    return bisect_right(bins, coverage)


def complexity_bin(complexity: int, bins: List[int]) -> int:
    return bisect_left(bins, complexity)


def _coerce_coverage(value: object) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return 0.0
    numeric = float(value)
    if not math.isfinite(numeric):
        return 0.0
    return min(1.0, max(0.0, numeric))
