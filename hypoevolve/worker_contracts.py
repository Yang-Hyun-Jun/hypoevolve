"""Worker-side contract payloads shared across orchestration boundaries."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class WorkerTask:
    """Serialize inputs for one worker-side mutation step."""

    parent_hypothesis: dict[str, object]
    parent_metrics: dict[str, object]
    iteration: int
    parent_score: float
    use_random_steering: bool = False
    llm_config: dict[str, object] = field(default_factory=dict)
    dataset_schema_path: str = "dataset.yaml"
    evaluator_parameters: dict[str, object] = field(default_factory=dict)
    steering_retries: int = 2
    recent_history: list[dict[str, object]] = field(default_factory=list)
    top_hypotheses: list[dict[str, object]] = field(default_factory=list)
    seen_fingerprints: list[str] = field(default_factory=list)


@dataclass(slots=True)
class WorkerResult:
    """Capture the outcome of one worker-side mutation and evaluation step."""

    child_hypothesis: dict[str, object]
    metrics: dict[str, object]
    iteration: int
    mutation_summary: str
    parent_score: float = 0.0
    domain_reason: str = ""
    score_reason: str = ""
    operation_score_rankings: dict[str, int] = field(default_factory=dict)
    random_steering: bool = False
    child_fingerprint: str = ""
    skipped_duplicate: bool = False
    skipped_steering_error: bool = False
    steering_error: str = ""
    evaluation_artifacts: dict[str, object] = field(default_factory=dict)


__all__ = ["WorkerTask", "WorkerResult"]
