"""Iteration-count stopping policy."""
from __future__ import annotations
from typing import Any


class IterationStoppingPolicy:
    """Stop when iteration count reaches the configured maximum."""

    def __init__(self, max_iterations: int) -> None:
        self.max_iterations = max_iterations

    def should_stop(self, state: dict[str, Any]) -> bool:
        """Return True when the iteration count meets or exceeds max_iterations."""
        return state.get("iteration", 0) >= self.max_iterations
