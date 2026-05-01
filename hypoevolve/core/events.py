"""Lightweight publish-subscribe event bus for harness lifecycle hooks."""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable


class HookBus:
    """Register handlers and emit named events."""

    def __init__(self) -> None:
        self._handlers: dict[str, list[Callable[..., None]]] = defaultdict(list)

    def register(self, event: str, handler: Callable[..., None]) -> None:
        """Register a handler for a named event."""
        self._handlers[event].append(handler)

    def emit(self, event: str, **kwargs: Any) -> None:
        """Emit a named event, calling all registered handlers."""
        for handler in self._handlers.get(event, []):
            handler(**kwargs)
