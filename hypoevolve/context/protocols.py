"""Context provider protocol for prompt variable injection."""
from __future__ import annotations

from typing import Any, Protocol


class ContextProvider(Protocol):
    """Protocol for building prompt context variables."""

    name: str

    def build(self, state: dict[str, Any]) -> dict[str, str]: ...
