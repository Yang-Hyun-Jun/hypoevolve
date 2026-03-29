from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from elg import hypothesis_from_dict, render_pretty
from hypoevolve.archive import ArchiveEntry
from hypoevolve.config import LLMConfig
from hypoevolve.dataset import load_dataset_schema
from hypoevolve.evaluator import LLMEvaluator, evaluate_hypothesis
from hypoevolve.llm import LLMClient
from hypoevolve.logger import logger
from hypoevolve.mutation import steer_mutation
from hypoevolve.parser import ParseError, llm_hypothesis_to_natural_language


@dataclass(slots=True)
class WorkerTask:
    parent_hypothesis: Dict[str, Any]
    parent_metrics: Dict[str, object]
    iteration: int
    parent_score: float
    mutation_atomic_pool: List[str] = field(default_factory=list)
    llm_config: Dict[str, Any] = field(default_factory=dict)
    dataset_schema_path: str = "dataset.yaml"
    evaluator_parameters: Dict[str, object] = field(default_factory=dict)
    parser_retries: int = 1
    steering_retries: int = 2
    recent_history: List[Dict[str, object]] = field(default_factory=list)
    top_hypotheses: List[Dict[str, object]] = field(default_factory=list)


@dataclass(slots=True)
class WorkerResult:
    child_hypothesis: Dict[str, Any]
    metrics: Dict[str, object]
    iteration: int
    mutation_operation: str
    mutation_path: List[int]
    mutation_details: Dict[str, Any] = field(default_factory=dict)
    parent_score: float = 0.0
    selected_candidate_index: int = 0
    steering_reason: str = ""


def run_worker_task(task: WorkerTask) -> WorkerResult:
    logger.info("worker iteration {} started", task.iteration)
    parent = hypothesis_from_dict(task.parent_hypothesis)
    llm = LLMClient(LLMConfig(**task.llm_config))
    schema = load_dataset_schema(task.dataset_schema_path)
    evaluator = LLMEvaluator(
        llm_client=llm,
        dataset_schema=schema,
        dataset_schema_path=task.dataset_schema_path,
        parameters=task.evaluator_parameters or None,
    )
    try:
        parent_nl = llm_hypothesis_to_natural_language(
            parent,
            llm=llm,
            retries=task.parser_retries,
        )
    except ParseError:
        parent_nl = render_pretty(parent)
        logger.error("worker iteration {} fell back to pretty hypothesis text", task.iteration)
    top_hypotheses = [
        ArchiveEntry(
            hypothesis=hypothesis_from_dict(item["hypothesis"]),
            metrics=dict(item["metrics"]),
            fingerprint=str(item["fingerprint"]),
            iteration=int(item.get("iteration", 0)),
            metadata=dict(item.get("metadata", {})),
        )
        for item in task.top_hypotheses
    ]
    decision = steer_mutation(
        parent_hypothesis=parent,
        parent_hypothesis_nl=parent_nl,
        current_metrics=task.parent_metrics,
        llm=llm,
        atomic_pool=task.mutation_atomic_pool,
        recent_history=task.recent_history,
        top_hypotheses=top_hypotheses,
        retries=task.steering_retries,
    )
    logger.info(
        "worker iteration {} selected mutation={} candidate={}",
        task.iteration,
        decision.mutation.operation,
        decision.selected_candidate_index,
    )
    metrics = evaluate_hypothesis(decision.mutation.result, evaluator)
    logger.info("worker iteration {} evaluation completed", task.iteration)
    return WorkerResult(
        child_hypothesis=decision.mutation.result.to_dict(),
        metrics=metrics,
        iteration=task.iteration,
        mutation_operation=decision.mutation.operation,
        mutation_path=list(decision.mutation.path),
        mutation_details=dict(decision.mutation.details),
        parent_score=task.parent_score,
        selected_candidate_index=decision.selected_candidate_index,
        steering_reason=decision.reason,
    )
