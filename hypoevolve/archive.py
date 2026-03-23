from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from elg import Hypothesis, fingerprint


@dataclass(slots=True)
class ArchiveEntry:
    hypothesis: Hypothesis
    metrics: Dict[str, float]
    fingerprint: str
    iteration: int = 0
    metadata: Dict[str, object] = field(default_factory=dict)

    @property
    def score(self) -> float:
        if "combined_score" in self.metrics and isinstance(self.metrics["combined_score"], (int, float)):
            return float(self.metrics["combined_score"])
        numeric = [v for v in self.metrics.values() if isinstance(v, (int, float)) and not isinstance(v, bool)]
        return float(sum(numeric) / len(numeric)) if numeric else 0.0


class Archive:
    def __init__(self, top_k: int = 5):
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        self.top_k = top_k
        self._entries: Dict[str, ArchiveEntry] = {}

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def entries(self) -> List[ArchiveEntry]:
        return sorted(self._entries.values(), key=lambda entry: entry.score, reverse=True)

    @property
    def best(self) -> Optional[ArchiveEntry]:
        items = self.entries
        return items[0] if items else None

    def add(
        self,
        hypothesis: Hypothesis,
        metrics: Dict[str, float],
        iteration: int = 0,
        metadata: Optional[Dict[str, object]] = None,
    ) -> ArchiveEntry:
        fp = fingerprint(hypothesis)
        new_entry = ArchiveEntry(
            hypothesis=hypothesis,
            metrics=dict(metrics),
            fingerprint=fp,
            iteration=iteration,
            metadata=dict(metadata or {}),
        )

        existing = self._entries.get(fp)
        if existing is None or new_entry.score > existing.score:
            self._entries[fp] = new_entry

        self._trim()
        return self._entries[fp]

    def sample_parent(self, rng: Optional[random.Random] = None) -> ArchiveEntry:
        ordered = self.entries
        if not ordered:
            raise ValueError("Cannot sample from an empty archive")
        chooser = rng or random.Random()
        weights = [max(entry.score, 1e-6) for entry in ordered]
        return chooser.choices(ordered, weights=weights, k=1)[0]

    def snapshot(self) -> List[Dict[str, object]]:
        return [
            {
                "fingerprint": entry.fingerprint,
                "iteration": entry.iteration,
                "metrics": dict(entry.metrics),
                "hypothesis": entry.hypothesis.to_dict(),
                "metadata": dict(entry.metadata),
            }
            for entry in self.entries
        ]

    def _trim(self) -> None:
        ordered = self.entries
        if len(ordered) <= self.top_k:
            return
        keep = {entry.fingerprint for entry in ordered[: self.top_k]}
        self._entries = {fp: entry for fp, entry in self._entries.items() if fp in keep}
