"""Artifact payload builders shared across recorder and runtime boundaries."""

from __future__ import annotations

from typing import Any

from elg import Hypothesis, fingerprint, render_pretty
from hypoevolve.archive import MAPElitesArchive


def build_checkpoint_payload(
    archive: MAPElitesArchive,
    iteration: int,
) -> dict[str, object]:
    """Build the persisted checkpoint payload for the current archive state."""
    best = archive.best
    return {
        "iteration": iteration,
        "archive_size": len(archive),
        "best_metrics": dict(best.metrics) if best else {},
        "best_hypothesis": best.hypothesis.to_dict() if best else None,
        "archive": archive.snapshot(),
    }


def build_trace_event(
    iteration: int,
    parent: Hypothesis | None,
    child: Hypothesis,
    metrics: dict[str, object],
    metadata: dict[str, object],
) -> dict[str, object]:
    """Build one persisted trace event payload."""
    return {
        "iteration": iteration,
        "parent": parent.to_dict() if parent else None,
        "child": child.to_dict(),
        "metrics": metrics,
        "metadata": metadata,
    }


def build_history_entry(
    *,
    iteration: int,
    hypothesis: Hypothesis,
    metrics: dict[str, object],
    best_score_after: float,
    best_updated: bool,
    status: str,
    metadata: dict[str, object],
    descriptor: dict[str, Any],
    parent_fingerprint: str | None = None,
) -> dict[str, object]:
    """Build one score-history entry payload."""
    hypothesis_nl = str(metadata.get("hypothesis_nl", "")).strip()
    return {
        "iteration": iteration,
        "status": status,
        "fingerprint": fingerprint(hypothesis),
        "parent_fingerprint": parent_fingerprint,
        "score": float(metrics.get("combined_score", 0.0)),
        "precision": metrics.get("precision", 0.0),
        "baseline": metrics.get("baseline", 0.0),
        "coverage": metrics.get("coverage", 0.0),
        "uplift": metrics.get("uplift", 0.0),
        "support_count": metrics.get("support_count", 0),
        "total_count": metrics.get("total_count", 0),
        "best_score_after": best_score_after,
        "best_updated": best_updated,
        "hypothesis_nl": hypothesis_nl or render_pretty(hypothesis),
        "mutation_summary": metadata.get("mutation_summary", ""),
        "score_reason": metadata.get("score_reason", ""),
        "domain_reason": metadata.get("domain_reason", ""),
        "worker_mode": bool(metadata.get("worker_mode", False)),
        "random_steering": bool(metadata.get("random_steering", False)),
        "complexity": descriptor.get("complexity", 0),
        "cell": list(descriptor.get("cell", [])),
    }


__all__ = [
    "build_checkpoint_payload",
    "build_history_entry",
    "build_trace_event",
]
