from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any, Dict

from elg import Hypothesis, hypothesis_to_json


def create_run_dir(base_dir: str = ".hypoevolve/runs", run_id: str | None = None) -> Path:
    actual_id = run_id or uuid.uuid4().hex[:8]
    run_dir = Path(base_dir) / actual_id
    (run_dir / "artifacts").mkdir(parents=True, exist_ok=True)
    return run_dir


def write_trace(run_dir: Path, event: Dict[str, Any]) -> Path:
    path = run_dir / "trace.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
    return path


def write_checkpoint(run_dir: Path, state: Dict[str, Any]) -> Path:
    path = run_dir / "checkpoint.json"
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return path


def write_best(run_dir: Path, hypothesis: Hypothesis, metrics: Dict[str, Any]) -> Path:
    path = run_dir / "best.json"
    payload = {
        "hypothesis": json.loads(hypothesis_to_json(hypothesis, indent=None)),
        "metrics": metrics,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return path


def write_artifact(run_dir: Path, name: str, payload: Dict[str, Any]) -> Path:
    path = run_dir / "artifacts" / f"{name}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return path
