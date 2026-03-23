from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

from elg import AtomicNode, Hypothesis, LogicalNode, RelationNode


class ParseError(ValueError):
    def __init__(self, message: str, errors: Optional[List[str]] = None):
        super().__init__(message)
        self.errors = errors or []


ParserCallable = Callable[[str], Hypothesis]


def parse_hypothesis_text(
    text: str,
    parser: Optional[ParserCallable] = None,
    retries: int = 1,
) -> Hypothesis:
    if not text or not text.strip():
        raise ParseError("Hypothesis text must be non-empty")

    if parser is None:
        return fallback_parse_hypothesis(text)

    errors: List[str] = []
    attempts = retries + 1
    for attempt in range(1, attempts + 1):
        try:
            result = parser(text)
            if not isinstance(result, Hypothesis):
                raise TypeError(f"Parser returned {type(result)!r}, expected Hypothesis")
            return result
        except Exception as exc:  # noqa: BLE001
            errors.append(f"attempt {attempt}: {exc}")
    raise ParseError("Failed to convert natural-language hypothesis to ELG", errors=errors)


def fallback_parse_hypothesis(text: str) -> Hypothesis:
    cleaned = " ".join(text.strip().split())
    lowered = cleaned.lower()

    if lowered.startswith("if ") and " then " in lowered:
        then_index = lowered.index(" then ")
        condition = cleaned[3:then_index].strip()
        target = cleaned[then_index + 6 :].strip()
        if condition and target:
            return Hypothesis(root=RelationNode("IMPLIES", [AtomicNode(condition), AtomicNode(target)]))

    splitters = [" and ", " AND ", " 그리고 "]
    for splitter in splitters:
        if splitter in cleaned:
            parts = [part.strip() for part in cleaned.split(splitter) if part.strip()]
            if len(parts) >= 2:
                return Hypothesis(root=LogicalNode("AND", [AtomicNode(part) for part in parts]))

    return Hypothesis(root=AtomicNode(cleaned))
