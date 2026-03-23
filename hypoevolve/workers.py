from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from elg import Hypothesis, hypothesis_from_dict, sample_mutation
from hypoevolve.evaluator import PlaceholderEvaluator, evaluate_hypothesis


@dataclass(slots=True)
class WorkerTask:
    parent_hypothesis: Dict[str, Any]
    iteration: int
    parent_score: float
    mutation_atomic_pool: List[str] = field(default_factory=list)
    evaluator_seed: int = 42


@dataclass(slots=True)
class WorkerResult:
    child_hypothesis: Dict[str, Any]
    metrics: Dict[str, float]
    iteration: int
    mutation_operation: str
    mutation_path: List[int]
    mutation_details: Dict[str, Any] = field(default_factory=dict)
    parent_score: float = 0.0


def run_worker_task(task: WorkerTask) -> WorkerResult:
    parent = hypothesis_from_dict(task.parent_hypothesis)
    evaluator = PlaceholderEvaluator(seed=task.evaluator_seed)
    mutation_sample = sample_mutation(
        parent,
        atomic_pool=task.mutation_atomic_pool,
    )
    metrics = evaluate_hypothesis(mutation_sample.result, evaluator)
    return WorkerResult(
        child_hypothesis=mutation_sample.result.to_dict(),
        metrics=metrics,
        iteration=task.iteration,
        mutation_operation=mutation_sample.operation,
        mutation_path=list(mutation_sample.path),
        mutation_details=dict(mutation_sample.details),
        parent_score=task.parent_score,
    )
