"""Worker task payloads and parallel execution flow."""

from __future__ import annotations

from hypoevolve.archive import ArchiveEntry
from hypoevolve.config import LLMConfig
from hypoevolve.dataset import load_dataset_schema
from hypoevolve.elg import fingerprint, hypothesis_from_dict
from hypoevolve.evaluator import LLMEvaluator
from hypoevolve.llm import LLMClient
from hypoevolve.logger import (
    compact_text,
    log_error_event,
    log_info_event,
    summarize_exception,
    summarize_metrics,
)
from hypoevolve.mutation import steer_mutation
from hypoevolve.parser import ParseError
from hypoevolve.worker_contracts import WorkerResult, WorkerTask

__all__ = ["WorkerTask", "WorkerResult", "run_worker_task"]


def run_worker_task(task: WorkerTask) -> WorkerResult:
    """Execute one worker task from parent selection through child scoring."""
    log_info_event(
        "worker.start",
        i=task.iteration,
        random=task.use_random_steering,
    )
    parent = hypothesis_from_dict(task.parent_hypothesis)
    llm = LLMClient(LLMConfig(**task.llm_config))
    schema = load_dataset_schema(task.dataset_schema_path)
    evaluator = LLMEvaluator(
        llm_client=llm,
        dataset_schema=schema,
        dataset_schema_path=task.dataset_schema_path,
        parameters=task.evaluator_parameters or None,
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
    try:
        decision = steer_mutation(
            parent_hypothesis=parent,
            current_metrics=task.parent_metrics,
            llm=llm,
            recent_history=task.recent_history,
            top_hypotheses=top_hypotheses,
            use_random_steering=task.use_random_steering,
            retries=task.steering_retries,
        )
    except ParseError as exc:
        log_error_event(
            "worker.skip_steering_error",
            i=task.iteration,
            **summarize_exception(exc),
        )
        return WorkerResult(
            child_hypothesis={},
            metrics={},
            iteration=task.iteration,
            mutation_summary="steering_failed",
            parent_score=task.parent_score,
            random_steering=task.use_random_steering,
            skipped_steering_error=True,
            steering_error=str(exc),
        )
    log_info_event(
        "worker.mutation",
        i=task.iteration,
        summary=compact_text(decision.mutation_summary, max_len=96),
    )
    child_fingerprint = fingerprint(decision.child_hypothesis)
    if child_fingerprint in set(task.seen_fingerprints):
        log_info_event(
            "worker.skip_duplicate",
            i=task.iteration,
            child_fp=child_fingerprint[:12],
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
    metrics = evaluator.evaluate(decision.child_hypothesis)
    log_info_event(
        "worker.eval",
        i=task.iteration,
        child_fp=child_fingerprint[:12],
        **summarize_metrics(metrics),
    )
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
        evaluation_artifacts=dict(
            getattr(evaluator, "last_evaluation_artifacts", {}) or {}
        ),
    )
