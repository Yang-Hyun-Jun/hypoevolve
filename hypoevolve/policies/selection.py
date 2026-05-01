"""UCB-based parent selection policy for archive-driven search."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional

from hypoevolve.memory.archive import ArchiveEntry, MAPElitesArchive


UCB_EXPLORATION_WEIGHT = 0.01


@dataclass(slots=True)
class SamplingStats:
    """Track parent-selection outcomes for UCB-style sampling."""

    pulls: int = 0
    total_reward: float = 0.0
    last_reward: float = 0.0

    @property
    def mean_reward(self) -> float:
        """Return the average observed reward for one sampled parent.

        Args:
            None.

        Returns:
            float: The mean reward across recorded pulls.
        """
        if self.pulls < 1:
            return 0.0
        return self.total_reward / self.pulls


class UCBSelectionPolicy:
    """Select parents from an archive using UCB-style exploration."""

    def __init__(self, exploration_weight: float = UCB_EXPLORATION_WEIGHT):
        """Initialize the selection policy.

        Args:
            exploration_weight: The UCB exploration weight.

        Returns:
            None.
        """
        self.exploration_weight = exploration_weight
        self._sampling_stats: Dict[str, SamplingStats] = {}

    def select(
        self,
        archive: MAPElitesArchive,
        rng: Optional[random.Random] = None,
    ) -> ArchiveEntry:
        """Sample one parent from the archive using UCB within MAP-Elites cells.

        Args:
            archive: The MAP-Elites archive to sample from.
            rng: Optional random number generator.

        Returns:
            ArchiveEntry: The selected parent entry.
        """
        if len(archive) == 0:
            raise ValueError("Cannot sample from an empty archive")
        chooser = rng or random.Random()
        occupied_cells = sorted(archive._cells)
        selected_cell = chooser.choice(occupied_cells)
        candidates = archive._cells[selected_cell]
        selected = max(
            candidates,
            key=lambda entry: (
                self._ucb_score(entry, candidates),
                entry.score,
            ),
        )
        self._stats_for(selected.fingerprint).pulls += 1
        return selected

    def _stats_for(self, fingerprint_value: str) -> SamplingStats:
        """Return mutable sampling stats for one fingerprint.

        Args:
            fingerprint_value: The hypothesis fingerprint to track.

        Returns:
            SamplingStats: The mutable stats bucket for that fingerprint.
        """
        return self._sampling_stats.setdefault(fingerprint_value, SamplingStats())

    def _ucb_score(
        self,
        entry: ArchiveEntry,
        candidates: List[ArchiveEntry],
    ) -> float:
        """Compute one UCB-style sampling score within a cell.

        Args:
            entry: The candidate being ranked.
            candidates: The competing candidates in the same occupied cell.

        Returns:
            float: The exploitation-plus-exploration score.
        """
        stats = self._stats_for(entry.fingerprint)
        if stats.pulls == 0:
            return float("inf")

        total_pulls = sum(
            self._stats_for(candidate.fingerprint).pulls for candidate in candidates
        )
        if total_pulls <= 1:
            return stats.mean_reward

        exploration_bonus = self.exploration_weight * math.sqrt(
            math.log(total_pulls) / stats.pulls
        )
        return stats.mean_reward + exploration_bonus
