"""Runs subgroup commands for the HypoEvolve CLI."""

from __future__ import annotations

import json
from pathlib import Path

import click

from hypoevolve.core.config import ConfigError, load_runtime_config
from hypoevolve.skills.reporting import generate_run_report

from .display import (
    _echo_banner,
    _echo_block,
    _echo_error,
    _echo_json,
    _echo_kv_rows,
)

CONFIG_HELP = "Path to the project config file."
DEFAULT_CONFIG_PATH = "hypoevolve.yaml"


@click.group(help="Inspect run directories.")
def runs() -> None:
    """Inspect run directories."""


@runs.command("latest", help="Print the most recently updated run directory.")
@click.option(
    "--config", default=None, help=f"{CONFIG_HELP} Defaults to {DEFAULT_CONFIG_PATH}."
)
@click.option(
    "--json", "as_json", is_flag=True, help="Print machine-readable JSON output."
)
def runs_latest(config: str | None, as_json: bool) -> int:
    """Print the latest run directory."""
    loaded = load_runtime_config(config, default_path=DEFAULT_CONFIG_PATH)
    base_dir = Path(loaded.output.base_dir)
    latest = _latest_run_dir(base_dir)
    if latest is None:
        _echo_error(ConfigError(f"No runs found under: {base_dir}"))
        return 1
    payload = {"run_dir": str(latest)}
    if as_json:
        _echo_json(payload)
        return 0
    click.echo(str(latest))
    return 0


@runs.command("status", help="Show compact run status by run id.")
@click.argument("run_id")
@click.option(
    "--config", default=None, help=f"{CONFIG_HELP} Defaults to {DEFAULT_CONFIG_PATH}."
)
@click.option(
    "--json", "as_json", is_flag=True, help="Print machine-readable JSON output."
)
def runs_status(run_id: str, config: str | None, as_json: bool) -> int:
    """Show run status for one run id."""
    try:
        run_dir = _run_dir_from_id(_resolve_runs_base_dir(config), run_id)
    except ConfigError as exc:
        raise click.ClickException(str(exc)) from exc
    payload = _status_payload(run_dir)
    if as_json:
        _echo_json(payload)
        return 0
    _echo_banner()
    _echo_kv_rows(
        "Run Status",
        [
            ("Run id", run_id),
            ("Run directory", payload["run_dir"]),
            ("Status", payload["status"]),
            (
                "Iteration",
                f"{payload['current_iteration']}/{payload['iterations_requested']}",
            ),
            ("Best score", str(payload["best_score"])),
            ("Archive size", str(payload["archive_size"])),
            ("Duplicate skips", str(payload["duplicate_skips_total"])),
            ("Report", str(payload["report_path"] or "-")),
        ],
    )
    if payload["best_hypothesis_nl"]:
        _echo_block("Best hypothesis summary", str(payload["best_hypothesis_nl"]))
    return 0


@runs.command("report", help="Return or regenerate the markdown report by run id.")
@click.argument("run_id")
@click.option(
    "--config", default=None, help=f"{CONFIG_HELP} Defaults to {DEFAULT_CONFIG_PATH}."
)
@click.option(
    "--json", "as_json", is_flag=True, help="Print machine-readable JSON output."
)
def runs_report(run_id: str, config: str | None, as_json: bool) -> int:
    """Return the report path for one run id."""
    try:
        run_dir = _run_dir_from_id(_resolve_runs_base_dir(config), run_id)
    except ConfigError as exc:
        raise click.ClickException(str(exc)) from exc
    report_path = run_dir / "report" / "report.md"
    if not report_path.exists():
        report_path = generate_run_report(run_dir).markdown_path
    payload = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "report_path": str(report_path),
    }
    if as_json:
        _echo_json(payload)
        return 0
    click.echo(str(report_path))
    return 0


def _latest_run_dir(base_dir: Path) -> Path | None:
    if not base_dir.exists():
        return None
    candidates = [path for path in base_dir.iterdir() if path.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _resolve_runs_base_dir(config: str | None) -> Path:
    loaded = load_runtime_config(config, default_path=DEFAULT_CONFIG_PATH)
    return Path(loaded.output.base_dir)


def _run_dir_from_id(base_dir: Path, run_id: str) -> Path:
    run_dir = base_dir / run_id
    if not run_dir.exists() or not run_dir.is_dir():
        raise ConfigError(f"Run id not found under {base_dir}: {run_id}")
    return run_dir


def _status_payload(run_dir: Path) -> dict[str, object]:
    summary_path = run_dir / "run_summary.json"
    checkpoint_path = run_dir / "checkpoint.json"
    score_history_path = run_dir / "score_history.json"
    report_path = run_dir / "report" / "report.md"

    summary = _read_json_if_exists(summary_path) or {}
    checkpoint = _read_json_if_exists(checkpoint_path) or {}
    score_history = _read_json_if_exists(score_history_path) or []

    current_iteration = int(checkpoint.get("iteration", 0))
    iterations_requested = int(summary.get("iterations_requested", current_iteration))
    best_score = summary.get("best_score")
    if best_score is None:
        best_score = checkpoint.get("best_metrics", {}).get("combined_score", 0.0)
    archive_size = summary.get("archive_size")
    if archive_size is None:
        archive_size = checkpoint.get("archive_size", 0)
    best_hypothesis_nl = summary.get("best_hypothesis_nl", "")
    if not best_hypothesis_nl and score_history:
        best_entries = [item for item in score_history if item.get("best_updated")]
        if best_entries:
            best_hypothesis_nl = best_entries[-1].get("hypothesis_nl", "")
    if summary_path.exists():
        status_text = "completed"
    elif checkpoint_path.exists():
        status_text = "running"
    else:
        status_text = "failed"
    return {
        "run_dir": str(run_dir),
        "status": status_text,
        "current_iteration": current_iteration,
        "iterations_requested": iterations_requested,
        "best_score": best_score,
        "best_hypothesis_nl": best_hypothesis_nl,
        "archive_size": archive_size,
        "duplicate_skips_total": summary.get("duplicate_skips_total", 0),
        "report_path": str(report_path) if report_path.exists() else None,
    }


def _read_json_if_exists(path: Path) -> object | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
