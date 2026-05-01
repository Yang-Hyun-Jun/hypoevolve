"""Selection and stopping policy protocols."""
from __future__ import annotations

from typing import Any, Protocol


class SelectionPolicy(Protocol):
    """Protocol for parent selection from an archive."""

    def select(self, archive: Any, rng: Any | None = None) -> Any: ...


class StoppingPolicy(Protocol):
    """Protocol for iteration stopping decisions."""

    def should_stop(self, state: dict[str, Any]) -> bool: ...
