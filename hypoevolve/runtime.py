"""Runtime artifact writers for traces, checkpoints, and best candidates."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any, Dict

from hypoevolve.elg import Hypothesis, hypothesis_to_json
from hypoevolve.logger import log_info_event


def create_run_dir(
    base_dir: str = ".hypoevolve/runs", run_id: str | None = None
) -> Path:
    """Create and return a run directory with an ``artifacts`` subdirectory."""
    actual_id = run_id or uuid.uuid4().hex[:8]
    run_dir = Path(base_dir) / actual_id
    (run_dir / "artifacts").mkdir(parents=True, exist_ok=True)
    log_info_event("run_dir.create", run=str(actual_id), path=run_dir)
    return run_dir


def write_trace(run_dir: Path, event: Dict[str, Any]) -> Path:
    """Append one JSON event to the run trace file."""
    path = run_dir / "trace.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
    return path


def write_checkpoint(run_dir: Path, state: Dict[str, Any]) -> Path:
    """Write the latest checkpoint snapshot for a run."""
    path = run_dir / "checkpoint.json"
    path.write_text(
        json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return path


def write_best(run_dir: Path, hypothesis: Hypothesis, metrics: Dict[str, Any]) -> Path:
    """Write the current best hypothesis and metrics for a run."""
    path = run_dir / "best.json"
    payload = {
        "hypothesis": json.loads(hypothesis_to_json(hypothesis, indent=None)),
        "metrics": metrics,
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return path


def write_artifact(run_dir: Path, name: str, payload: Dict[str, Any]) -> Path:
    """Write one named JSON artifact under the run's artifact directory."""
    path = run_dir / "artifacts" / f"{name}.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return path


def write_run_summary(run_dir: Path, payload: Dict[str, Any]) -> Path:
    """Write one run-level summary payload."""
    path = run_dir / "run_summary.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return path


def write_score_history(run_dir: Path, payload: list[Dict[str, Any]]) -> Path:
    """Write iteration-level score history for one run."""
    path = run_dir / "score_history.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return path
