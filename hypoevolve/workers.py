"""Worker task payloads and execution helpers for parallel evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from elg import fingerprint, hypothesis_from_dict, render_pretty
from hypoevolve.archive import ArchiveEntry
from hypoevolve.config import LLMConfig
from hypoevolve.dataset import load_dataset_schema
from hypoevolve.evaluator import (
    LLMEvaluator,
    evaluate_hypothesis,
    get_evaluation_artifacts,
)
from hypoevolve.llm import LLMClient
from hypoevolve.logger import logger
from hypoevolve.mutation import steer_mutation
from hypoevolve.parser import ParseError, llm_hypothesis_to_natural_language


@dataclass(slots=True)
class WorkerTask:
    """Serialize the inputs needed for one worker-side mutation step."""

    parent_hypothesis: Dict[str, Any]
    parent_metrics: Dict[str, object]
    iteration: int
    parent_score: float
    parent_hypothesis_nl: str = ""
    use_random_steering: bool = False
    llm_config: Dict[str, Any] = field(default_factory=dict)
    dataset_schema_path: str = "dataset.yaml"
    evaluator_parameters: Dict[str, object] = field(default_factory=dict)
    parser_retries: int = 1
    steering_retries: int = 2
    recent_history: List[Dict[str, object]] = field(default_factory=list)
    top_hypotheses: List[Dict[str, object]] = field(default_factory=list)
    seen_fingerprints: List[str] = field(default_factory=list)


@dataclass(slots=True)
class WorkerResult:
    """Return the outcome of one worker-side mutation and evaluation step."""

    child_hypothesis: Dict[str, Any]
    metrics: Dict[str, object]
    iteration: int
    mutation_summary: str
    parent_score: float = 0.0
    domain_reason: str = ""
    score_reason: str = ""
    operation_score_rankings: Dict[str, int] = field(default_factory=dict)
    random_steering: bool = False
    child_fingerprint: str = ""
    skipped_duplicate: bool = False
    evaluation_artifacts: Dict[str, object] = field(default_factory=dict)


def run_worker_task(task: WorkerTask) -> WorkerResult:
    """Execute one worker task from parent selection through child scoring."""
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
    parent_nl = task.parent_hypothesis_nl.strip()
    if not parent_nl:
        try:
            parent_nl = llm_hypothesis_to_natural_language(
                parent,
                llm=llm,
                retries=task.parser_retries,
            )
        except ParseError:
            parent_nl = render_pretty(parent)
            logger.error(
                "worker iteration {} fell back to pretty hypothesis text",
                task.iteration,
            )
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
        recent_history=task.recent_history,
        top_hypotheses=top_hypotheses,
        use_random_steering=task.use_random_steering,
        retries=task.steering_retries,
    )
    logger.info(
        "worker iteration {} produced mutation_summary={}",
        task.iteration,
        decision.mutation_summary,
    )
    child_fingerprint = fingerprint(decision.child_hypothesis)
    if child_fingerprint in set(task.seen_fingerprints):
        logger.info(
            "worker iteration {} skipped duplicate child_fp={}",
            task.iteration,
            child_fingerprint,
        )
        return WorkerResult(
            child_hypothesis=decision.child_hypothesis.to_dict(),
            metrics={},
            iteration=task.iteration,
            mutation_summary=decision.mutation_summary,
            parent_score=task.parent_score,
            domain_reason=decision.domain_reason,
            score_reason=decision.score_reason,
            operation_score_rankings=dict(
                getattr(decision, "operation_score_rankings", {})
            ),
            random_steering=task.use_random_steering,
            child_fingerprint=child_fingerprint,
            skipped_duplicate=True,
        )
    metrics = evaluate_hypothesis(decision.child_hypothesis, evaluator)
    logger.info("worker iteration {} evaluation completed", task.iteration)
    return WorkerResult(
        child_hypothesis=decision.child_hypothesis.to_dict(),
        metrics=metrics,
        iteration=task.iteration,
        mutation_summary=decision.mutation_summary,
        parent_score=task.parent_score,
        domain_reason=decision.domain_reason,
        score_reason=decision.score_reason,
        operation_score_rankings=dict(
            getattr(decision, "operation_score_rankings", {})
        ),
        random_steering=task.use_random_steering,
        child_fingerprint=child_fingerprint,
        evaluation_artifacts=get_evaluation_artifacts(evaluator),
    )
