"""Coulomb Archive — repulsive-field archiving and sampling for HypoEvolve.

Each stored hypothesis behaves as a point charge whose intensity is its quality
score. Candidates feel a repulsive potential from every archived member, and
archive maintenance/parent-sampling are derived from that potential.

Two rules define the entire policy:

- **Sampling**: ``P(parent = h) ∝ score(h) · exp(-γ · U(h; A))``
- **Archiving**: ``Δ(h) = score(h) - γ · U(h; A)``; the candidate replaces the
  weakest-``Δ`` member when the archive is at capacity, and is rejected
  otherwise.

Distance defaults to the ELG tree-kernel distance so the archive can be dropped
into HypoEvolve unchanged; a custom callable can be injected for testing or for
alternative similarity metrics.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from hypoevolve.elg import Hypothesis, count_nodes, fingerprint
from hypoevolve.elg.kernel import tree_distance


DistanceFn = Callable[[Hypothesis, Hypothesis], float]


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

    @property
    def score(self) -> float:
        """Return one stable numeric score for archive ranking."""
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
    """Track parent-selection outcomes for repulsive-field sampling."""

    pulls: int = 0
    total_reward: float = 0.0
    last_reward: float = 0.0

    @property
    def mean_reward(self) -> float:
        """Return the average observed reward for one sampled parent."""
        if self.pulls < 1:
            return 0.0
        return self.total_reward / self.pulls


@dataclass(slots=True)
class CoulombDescriptor:
    """One Coulomb-friendly descriptor used for analytics and metadata."""

    potential: float
    quality: float
    complexity: int

    def to_dict(self) -> Dict[str, float]:
        return {
            "potential": float(self.potential),
            "quality": float(self.quality),
            "complexity": int(self.complexity),
        }


class CoulombArchive:
    """Streaming repulsive-potential archive with tree-kernel distance."""

    def __init__(
        self,
        capacity: int = 64,
        gamma: float = 0.3,
        eps: float = 1e-2,
        distance_fn: Optional[DistanceFn] = None,
    ):
        """Initialize a Coulomb archive.

        Args:
            capacity: Maximum number of entries retained in the archive.
            gamma: Repulsion strength dial; balances quality vs diversity.
            eps: Small constant added under the ``d^2`` term to avoid singularities.
            distance_fn: Optional custom distance function on hypotheses;
                defaults to the ELG tree-kernel distance.

        Returns:
            None.
        """
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        if gamma < 0.0:
            raise ValueError("gamma must be non-negative")
        if eps <= 0.0:
            raise ValueError("eps must be strictly positive")

        self.capacity = int(capacity)
        self.gamma = float(gamma)
        self.eps = float(eps)
        self._distance_fn: DistanceFn = distance_fn or tree_distance

        self._entries: List[ArchiveEntry] = []
        self._distances: Dict[tuple[str, str], float] = {}
        self._sampling_stats: Dict[str, SamplingStats] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return the current number of archive entries."""
        return len(self._entries)

    @property
    def entries(self) -> List[ArchiveEntry]:
        """Return all entries sorted by descending score."""
        return sorted(self._entries, key=lambda entry: entry.score, reverse=True)

    @property
    def best(self) -> Optional[ArchiveEntry]:
        """Return the current highest-scoring entry, if any."""
        if not self._entries:
            return None
        return max(self._entries, key=lambda entry: entry.score)

    def describe(
        self,
        hypothesis: Hypothesis,
        metrics: Dict[str, object],
    ) -> Dict[str, object]:
        """Compute a Coulomb-friendly descriptor for a candidate.

        The descriptor reports the potential the candidate would feel against
        the *current* archive contents, together with its quality and node count.
        """
        quality = _score_from_metrics(metrics)
        potential = self._potential_for_hypothesis(
            hypothesis, exclude_fingerprint=fingerprint(hypothesis)
        )
        complexity = count_nodes(hypothesis)
        descriptor = CoulombDescriptor(
            potential=potential, quality=quality, complexity=complexity
        )
        return {
            "coverage": _coverage_from_metrics(metrics),
            "complexity": complexity,
            "coulomb": descriptor.to_dict(),
        }

    def add(
        self,
        hypothesis: Hypothesis,
        metrics: Dict[str, object],
        iteration: int = 0,
        metadata: Optional[Dict[str, object]] = None,
    ) -> ArchiveEntry:
        """Offer a candidate to the archive and return the resulting entry.

        The candidate is admitted directly when there is spare capacity.
        Otherwise the weakest existing member (lowest ``Δ``) is compared against
        the candidate; whichever has the higher ``Δ`` occupies the slot.
        Duplicates (same fingerprint) are merged in place with the higher score.
        """
        descriptor = self.describe(hypothesis, metrics)
        entry_metadata = dict(metadata or {})
        entry_metadata["coulomb"] = dict(descriptor["coulomb"])
        candidate_fp = fingerprint(hypothesis)
        entry = ArchiveEntry(
            hypothesis=hypothesis,
            metrics=dict(metrics),
            fingerprint=candidate_fp,
            iteration=iteration,
            metadata=entry_metadata,
            coverage=float(descriptor["coverage"]),
            complexity=int(descriptor["complexity"]),
        )

        existing_index = self._index_of_fingerprint(candidate_fp)
        if existing_index is not None:
            existing = self._entries[existing_index]
            if entry.score <= existing.score:
                return existing
            self._replace_at(existing_index, entry)
            return entry

        if len(self._entries) < self.capacity:
            self._entries.append(entry)
            return entry

        weakest_slot, weakest_delta = self._weakest_slot()
        candidate_delta = self._delta_if_replacing(entry, weakest_slot)
        if candidate_delta > weakest_delta:
            self._replace_at(weakest_slot, entry)
            return entry
        return entry

    def sample_parent(
        self,
        rng: Optional[random.Random] = None,
    ) -> ArchiveEntry:
        """Sample one parent under ``P(h) ∝ score(h) · exp(-γ · U(h; A))``."""
        if not self._entries:
            raise ValueError("Cannot sample from an empty archive")
        chooser = rng or random.Random()
        logits = [self._log_sampling_weight(index) for index in range(len(self._entries))]
        max_logit = max(logits)
        weights = [math.exp(value - max_logit) for value in logits]
        total = sum(weights)
        if total <= 0.0 or not math.isfinite(total):
            selected = chooser.choice(self._entries)
        else:
            probs = [w / total for w in weights]
            r = chooser.random()
            cumulative = 0.0
            selected = self._entries[-1]
            for entry, prob in zip(self._entries, probs):
                cumulative += prob
                if r <= cumulative:
                    selected = entry
                    break
        self._stats_for(selected.fingerprint).pulls += 1
        return selected

    def record_parent_outcome(self, fingerprint_value: str, reward: float) -> None:
        """Record the observed reward after sampling an entry as a parent."""
        stats = self._stats_for(fingerprint_value)
        numeric = float(reward)
        if not math.isfinite(numeric):
            numeric = 0.0
        stats.total_reward += numeric
        stats.last_reward = numeric

    def sampling_stats(self, fingerprint_value: str | None = None) -> Dict[str, object]:
        """Return sampling stats for one fingerprint or all fingerprints."""
        if fingerprint_value is not None:
            stats = self._stats_for(fingerprint_value)
            return _sampling_stats_dict(stats)
        return {
            key: _sampling_stats_dict(stats)
            for key, stats in sorted(self._sampling_stats.items())
        }

    def occupancy_stats(self) -> Dict[str, object]:
        """Return simple occupancy statistics for logging and analytics."""
        n = len(self._entries)
        if n == 0:
            return {
                "size": 0,
                "capacity": self.capacity,
                "mean_quality": 0.0,
                "mean_pairwise_distance": 0.0,
            }
        scores = [entry.score for entry in self._entries]
        mean_pairwise = self._mean_pairwise_distance()
        return {
            "size": n,
            "capacity": self.capacity,
            "mean_quality": sum(scores) / n,
            "mean_pairwise_distance": mean_pairwise,
        }

    def occupancy_summary(self) -> str:
        """Return a compact one-line human-readable occupancy summary."""
        stats = self.occupancy_stats()
        return (
            f"coulomb size={stats['size']}/{stats['capacity']} "
            f"meanQ={stats['mean_quality']:.3f} "
            f"meanD={stats['mean_pairwise_distance']:.3f}"
        )

    def snapshot(self) -> List[Dict[str, object]]:
        """Return a serializable snapshot of retained archive entries."""
        return [
            {
                "fingerprint": entry.fingerprint,
                "iteration": entry.iteration,
                "metrics": dict(entry.metrics),
                "hypothesis": entry.hypothesis.to_dict(),
                "metadata": dict(entry.metadata),
                "coverage": entry.coverage,
                "complexity": entry.complexity,
            }
            for entry in self.entries
        ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _log_sampling_weight(self, slot: int) -> float:
        entry = self._entries[slot]
        potential = self._potential_at_slot(slot)
        score = entry.score
        base = math.log(max(score, 1e-9))
        return base - self.gamma * potential

    def _weakest_slot(self) -> tuple[int, float]:
        deltas = [self._delta_at_slot(slot) for slot in range(len(self._entries))]
        weakest = min(range(len(deltas)), key=lambda i: deltas[i])
        return weakest, deltas[weakest]

    def _delta_at_slot(self, slot: int) -> float:
        entry = self._entries[slot]
        potential = self._potential_at_slot(slot)
        return entry.score - self.gamma * potential

    def _delta_if_replacing(
        self,
        candidate: ArchiveEntry,
        replaced_slot: int,
    ) -> float:
        potential = 0.0
        for slot, other in enumerate(self._entries):
            if slot == replaced_slot:
                continue
            distance = self._pair_distance(candidate, other)
            potential += other.score / (distance * distance + self.eps)
        return candidate.score - self.gamma * potential

    def _potential_at_slot(self, slot: int) -> float:
        entry = self._entries[slot]
        total = 0.0
        for other_slot, other in enumerate(self._entries):
            if other_slot == slot:
                continue
            distance = self._pair_distance(entry, other)
            total += other.score / (distance * distance + self.eps)
        return total

    def _potential_for_hypothesis(
        self,
        hypothesis: Hypothesis,
        exclude_fingerprint: str | None,
    ) -> float:
        total = 0.0
        for other in self._entries:
            if exclude_fingerprint is not None and other.fingerprint == exclude_fingerprint:
                continue
            distance = self._distance_between(hypothesis, other.hypothesis)
            total += other.score / (distance * distance + self.eps)
        return total

    def _pair_distance(self, a: ArchiveEntry, b: ArchiveEntry) -> float:
        return self._distance_between(a.hypothesis, b.hypothesis, a.fingerprint, b.fingerprint)

    def _distance_between(
        self,
        hypothesis_a: Hypothesis,
        hypothesis_b: Hypothesis,
        fingerprint_a: str | None = None,
        fingerprint_b: str | None = None,
    ) -> float:
        fp_a = fingerprint_a if fingerprint_a is not None else fingerprint(hypothesis_a)
        fp_b = fingerprint_b if fingerprint_b is not None else fingerprint(hypothesis_b)
        if fp_a == fp_b:
            return 0.0
        cache_key = (fp_a, fp_b) if fp_a < fp_b else (fp_b, fp_a)
        cached = self._distances.get(cache_key)
        if cached is not None:
            return cached
        distance = float(self._distance_fn(hypothesis_a, hypothesis_b))
        if not math.isfinite(distance):
            distance = 1.0
        distance = max(0.0, min(1.0, distance))
        self._distances[cache_key] = distance
        return distance

    def _mean_pairwise_distance(self) -> float:
        n = len(self._entries)
        if n < 2:
            return 0.0
        total = 0.0
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                total += self._pair_distance(self._entries[i], self._entries[j])
                count += 1
        return total / count if count else 0.0

    def _replace_at(self, slot: int, entry: ArchiveEntry) -> None:
        old = self._entries[slot]
        self._entries[slot] = entry
        old_fp = old.fingerprint
        stale_keys = [key for key in self._distances if old_fp in key]
        for key in stale_keys:
            self._distances.pop(key, None)
        if old_fp in self._sampling_stats and old_fp != entry.fingerprint:
            self._sampling_stats.pop(old_fp, None)

    def _index_of_fingerprint(self, target: str) -> Optional[int]:
        for index, entry in enumerate(self._entries):
            if entry.fingerprint == target:
                return index
        return None

    def _stats_for(self, fingerprint_value: str) -> SamplingStats:
        return self._sampling_stats.setdefault(fingerprint_value, SamplingStats())


def _score_from_metrics(metrics: Dict[str, object]) -> float:
    value = metrics.get("combined_score")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        numeric = float(value)
        if math.isfinite(numeric):
            return numeric
    return 0.0


def _coverage_from_metrics(metrics: Dict[str, object]) -> float:
    value = metrics.get("coverage")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        numeric = float(value)
        if math.isfinite(numeric):
            return min(1.0, max(0.0, numeric))
    return 0.0


def _sampling_stats_dict(stats: SamplingStats) -> Dict[str, float | int]:
    return {
        "pulls": stats.pulls,
        "total_reward": stats.total_reward,
        "mean_reward": stats.mean_reward,
        "last_reward": stats.last_reward,
    }


__all__ = [
    "ArchiveEntry",
    "SamplingStats",
    "CoulombArchive",
    "CoulombDescriptor",
]
