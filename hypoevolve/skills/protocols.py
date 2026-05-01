"""Role-based Skill protocols for the research plane."""
from __future__ import annotations

from typing import Any, Protocol


class SeedGenerationSkill(Protocol):
    """Protocol for seed hypothesis generation."""

    name: str

    def generate(self, context: dict[str, Any]) -> Any: ...


class CompileSkill(Protocol):
    """Protocol for hypothesis compilation from natural language."""

    name: str

    def compile(self, text: str, **kwargs: Any) -> Any: ...

    def make_measurable(self, hypothesis: Any, **kwargs: Any) -> Any: ...


class MutationSkill(Protocol):
    """Protocol for hypothesis mutation steering."""

    name: str

    def mutate(self, parent: Any, context: dict[str, Any]) -> Any: ...


class EvaluationSkill(Protocol):
    """Protocol for hypothesis evaluation."""

    name: str

    def evaluate(self, hypothesis: Any) -> dict[str, object]: ...


class ReportingSkill(Protocol):
    """Protocol for run report generation."""

    name: str

    def generate_report(self, run_dir: Any) -> Any: ...
