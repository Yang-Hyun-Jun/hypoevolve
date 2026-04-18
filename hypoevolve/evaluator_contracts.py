"""Evaluator-facing contracts shared across orchestration boundaries."""

from __future__ import annotations

from typing import Protocol

from elg import Hypothesis


REQUIRED_EVALUATION_KEYS = (
    "combined_score",
    "precision",
    "baseline",
    "coverage",
    "uplift",
    "support_count",
    "total_count",
    "rationale",
    "used_parameters",
)


class Evaluator(Protocol):
    """Protocol for objects that can score a hypothesis."""

    def evaluate(self, hypothesis: Hypothesis) -> dict[str, object]:
        """Evaluate one hypothesis and return a normalized payload."""
        ...


__all__ = ["Evaluator", "REQUIRED_EVALUATION_KEYS"]
