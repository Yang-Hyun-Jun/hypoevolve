"""Click-based command-line interface for the HypoEvolve project."""

from __future__ import annotations

import importlib.metadata as metadata
import json
import platform
import sys
from pathlib import Path

import click

from elg import hypothesis_from_dict, render_pretty, render_tree
from hypoevolve.config import ConfigError, HypoEvolveConfig, load_config
from hypoevolve.controller import HypoEvolveController
from hypoevolve.hypo import HypothesisGenerationError, generate_random_tree_pair_hypothesis
from hypoevolve.llm import LLMClient
from hypoevolve.logger import configure_logger, logger
from hypoevolve.parser import ParseError, parse_hypothesis_text
from hypoevolve.reporting import generate_run_report

CLI_WORDMARK = (
    "    __  __                      ______            __\n"
    "   / / / /_  ______  ____     / ____/   ______  / /   _____\n"
    "  / /_/ / / / / __ \\/ __ \\   / __/ | | / / __ \\/ / | / / _ \\\n"
    " / __  / /_/ / /_/ / /_/ /  / /___ | |/ / /_/ / /| |/ /  __/\n"
    "/_/ /_/\\__, / .___/\\____/  /_____/ |___/\\____/_/ |___/\\___/\n"
    "      /____/_/"
)
CLI_TAGLINE = "LLM-guided ELG hypothesis evolution"
CLI_SUBTITLE = "Evolve measurable ELG hypotheses from natural-language seeds."
CLI_RULE = "─" * 56
CONFIG_HELP = "Path to the project config file."
DEFAULT_CONFIG_PATH = "hypoevolve.yaml"
CLI_EXAMPLES = (
    "Quick start:\n"
    "  hypoevolve --version\n"
    "  hypoevolve run \"if BTC momentum drops then DOGE jumps\"\n"
    "  hypoevolve run\n"
    "  hypoevolve seed\n"
    "  hypoevolve render \"if A then B\" --tree\n"
    "  hypoevolve inspect .hypoevolve/runs/latest/best.json\n"
    "  hypoevolve runs status <run-id> --json\n"
    "  hypoevolve doctor"
)
CLI_TIPS = (
    "Tips:\n"
    "  - Bare `hypoevolve` prints help and exits non-zero to preserve script safety.\n"
    "  - Use `inspect` to review saved run artifacts without rerunning the pipeline."
)


def _detect_version() -> str:
    """Return the installed package version or ``dev`` for local checkouts."""
    try:
        return metadata.version("openresearch")
    except metadata.PackageNotFoundError:
        return "dev"


class StyledGroup(click.Group):
    """Customize help output with the project's banner and grouped sections."""

    def get_help(self, ctx: click.Context) -> str:
        command_rows = []
        for name in self.list_commands(ctx):
            command = self.get_command(ctx, name)
            if command is None:
                continue
            command_rows.append((name, command.get_short_help_str()))

        help_sections = [
            _render_banner(),
            "Usage:\n  hypoevolve [OPTIONS] COMMAND [ARGS]...",
            _render_kv_section("Commands", command_rows),
            "Options:\n  -h, --help     Show this message and exit.\n  --version      Show the installed CLI version and exit.",
            CLI_EXAMPLES,
            CLI_TIPS,
        ]
        return "\n\n".join(help_sections)


@click.group(
    cls=StyledGroup,
    context_settings={
        "help_option_names": ["-h", "--help"],
        "max_content_width": 100,
    },
    invoke_without_command=True,
)
@click.version_option(version=_detect_version(), prog_name="hypoevolve")
@click.pass_context
def app(ctx: click.Context) -> None:
    """Modern CLI for HypoEvolve."""
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())
        ctx.exit(1)


@app.command(help="Run the hypothesis evolution loop from a natural-language prompt.")
@click.argument("hypothesis", required=False)
@click.option("--config", default=None, help=f"{CONFIG_HELP} Defaults to {DEFAULT_CONFIG_PATH}.")
@click.option("--workers", type=int, default=None, help="Override the local worker count for this run.")
def run(hypothesis: str | None, config: str | None, workers: int | None) -> int:
    """Run the hypothesis evolution loop from one natural-language seed."""
    try:
        loaded = _load_runtime_config(config)
        configure_logger(loaded.logging.level)
        if workers is not None:
            loaded.workers.count = workers
            loaded.workers.enabled = workers > 1
        logger.info("cli run command started")
        result = HypoEvolveController(loaded).run(hypothesis)
        seed_generated = bool(getattr(result, "seed_generated", False))
        seed_input_text = str(getattr(result, "seed_input_text", "") or "")
        _echo_banner()
        _echo_kv_rows(
            "Run Summary",
            [
                ("Status", "ok"),
                ("Run directory", str(result.run_dir)),
                ("Report", str(result.report_path)),
                ("Seed source", "generated" if seed_generated else "provided"),
                ("Workers", str(loaded.workers.count)),
            ],
        )
        if seed_generated and seed_input_text:
            _echo_block("Seed hypothesis", seed_input_text)
        _echo_metric_highlights(result.best_metrics)
        _echo_block("Best hypothesis", render_pretty(result.best_hypothesis))
        _echo_block(
            "Best metrics",
            json.dumps(result.best_metrics, ensure_ascii=False, indent=2, sort_keys=True),
        )
        logger.info("cli run command completed")
        return 0
    except (ConfigError, ParseError, HypothesisGenerationError) as exc:
        logger.error("cli run command failed: {}", exc)
        _echo_error(exc)
        return 1


@app.command(help="Generate one random natural-language seed hypothesis from sampled feature trees.")
@click.option("--config", default=None, help=f"{CONFIG_HELP} Defaults to {DEFAULT_CONFIG_PATH}.")
@click.option("--max-depth", type=int, default=3, show_default=True, help="Maximum sampled tree depth.")
def seed(config: str | None, max_depth: int) -> int:
    """Generate one random seed hypothesis without running evolution."""
    try:
        loaded = _load_runtime_config(config)
        configure_logger(loaded.logging.level)
        result = generate_random_tree_pair_hypothesis(
            llm=LLMClient(loaded.llm),
            max_depth=max_depth,
        )
        _echo_banner()
        _echo_block("Feature tree A", result.tree_a.render(return_str=True))
        _echo_block("Feature tree B", result.tree_b.render(return_str=True))
        _echo_block("Generated hypothesis", result.hypothesis)
        return 0
    except (ConfigError, HypothesisGenerationError) as exc:
        logger.error("cli seed command failed: {}", exc)
        _echo_error(exc)
        return 1


@app.command(help="Parse and render a natural-language hypothesis via the LLM parser.")
@click.argument("hypothesis")
@click.option("--config", default=None, help=f"{CONFIG_HELP} Defaults to {DEFAULT_CONFIG_PATH}.")
@click.option("--tree", is_flag=True, help="Render as an ASCII tree instead of the pretty ELG form.")
def render(hypothesis: str, config: str | None, tree: bool) -> int:
    """Parse one hypothesis and render it as ELG text or an ASCII tree."""
    try:
        loaded = _load_runtime_config(config)
        configure_logger(loaded.logging.level)
        logger.info("cli render command started")
        parsed = parse_hypothesis_text(
            hypothesis,
            llm=LLMClient(loaded.llm),
            retries=loaded.parser.retries,
        )
        _echo_banner()
        _echo_block("Rendered hypothesis", render_tree(parsed) if tree else render_pretty(parsed))
        logger.info("cli render command completed")
        return 0
    except (ConfigError, ParseError) as exc:
        logger.error("cli render command failed: {}", exc)
        _echo_error(exc)
        return 1


@app.command(help="Inspect a saved best/checkpoint JSON artifact.")
@click.argument("path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--json", "as_json", is_flag=True, help="Print the raw JSON payload.")
def inspect(path: Path, as_json: bool) -> int:
    """Print a saved best/checkpoint artifact in a human-readable form."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if as_json:
        _echo_json(payload)
        return 0
    _echo_banner()
    _echo_kv_rows("Artifact", [("Path", str(path))])
    if "hypothesis" in payload:
        hypothesis = hypothesis_from_dict(payload["hypothesis"])
        _echo_block("Hypothesis", render_pretty(hypothesis))
        if "metrics" in payload:
            _echo_block(
                "Metrics",
                json.dumps(payload["metrics"], ensure_ascii=False, indent=2, sort_keys=True),
            )
    elif "best_hypothesis" in payload and payload["best_hypothesis"] is not None:
        hypothesis = hypothesis_from_dict(payload["best_hypothesis"])
        _echo_block("Best hypothesis", render_pretty(hypothesis))
        if "best_metrics" in payload:
            _echo_block(
                "Best metrics",
                json.dumps(payload["best_metrics"], ensure_ascii=False, indent=2, sort_keys=True),
            )
    else:
        _echo_block("Payload", json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


@app.command(help="Show environment and config diagnostics.")
@click.option("--config", default=None, help=f"{CONFIG_HELP} Defaults to {DEFAULT_CONFIG_PATH}.")
def doctor(config: str | None) -> int:
    """Report environment and configuration diagnostics for the CLI."""
    _echo_banner()
    config_path = Path(config or DEFAULT_CONFIG_PATH)
    rows = [
        ("Python", platform.python_version()),
        ("Platform", platform.platform()),
        ("Config", str(config_path)),
        ("Config exists", str(config_path.exists()).lower()),
    ]
    if config_path.exists():
        try:
            loaded = load_config(config_path)
            rows.extend(
                [
                    ("Config ok", "true"),
                    ("Archive type", "MAP-Elites"),
                    ("Coverage bins", str(loaded.archive.coverage_bins)),
                    ("Complexity bins", str(loaded.archive.complexity_bins)),
                    ("Iterations", str(loaded.search.iterations)),
                    ("Steering retries", str(loaded.search.steering_retries)),
                    ("Dataset schema", loaded.evaluator.dataset_schema_path),
                    ("Output base dir", loaded.output.base_dir),
                    ("Workers enabled", str(loaded.workers.enabled).lower()),
                    ("Worker count", str(loaded.workers.count)),
                ]
            )
        except Exception as exc:  # noqa: BLE001
            rows.extend([("Config ok", "false"), ("Config error", str(exc))])
            _echo_kv_rows("Diagnostics", rows)
            return 1
    else:
        if config is None:
            rows.append(("Config ok", "using defaults"))
        else:
            rows.extend(
                [
                    ("Config ok", "false"),
                    ("Config error", f"Config file not found: {config_path}"),
                ]
            )
            _echo_kv_rows("Diagnostics", rows)
            return 1
    _echo_kv_rows("Diagnostics", rows)
    return 0


@app.group(help="Inspect run directories.")
def runs() -> None:
    """Inspect run directories."""


@runs.command("latest", help="Print the most recently updated run directory.")
@click.option("--config", default=None, help=f"{CONFIG_HELP} Defaults to {DEFAULT_CONFIG_PATH}.")
@click.option("--json", "as_json", is_flag=True, help="Print machine-readable JSON output.")
def runs_latest(config: str | None, as_json: bool) -> int:
    """Print the latest run directory."""
    loaded = _load_runtime_config(config)
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
@click.option("--config", default=None, help=f"{CONFIG_HELP} Defaults to {DEFAULT_CONFIG_PATH}.")
@click.option("--json", "as_json", is_flag=True, help="Print machine-readable JSON output.")
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
            ("Iteration", f'{payload["current_iteration"]}/{payload["iterations_requested"]}'),
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
@click.option("--config", default=None, help=f"{CONFIG_HELP} Defaults to {DEFAULT_CONFIG_PATH}.")
@click.option("--json", "as_json", is_flag=True, help="Print machine-readable JSON output.")
def runs_report(run_id: str, config: str | None, as_json: bool) -> int:
    """Return the report path for one run id."""
    try:
        run_dir = _run_dir_from_id(_resolve_runs_base_dir(config), run_id)
    except ConfigError as exc:
        raise click.ClickException(str(exc)) from exc
    report_path = run_dir / "report" / "report.md"
    if not report_path.exists():
        report_path = generate_run_report(run_dir).markdown_path
    payload = {"run_id": run_id, "run_dir": str(run_dir), "report_path": str(report_path)}
    if as_json:
        _echo_json(payload)
        return 0
    click.echo(str(report_path))
    return 0


@app.command(help="Show compact run status from persisted artifacts.")
@click.argument("run_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--json", "as_json", is_flag=True, help="Print machine-readable JSON output.")
def status(run_dir: Path, as_json: bool) -> int:
    """Show run status for one run directory."""
    payload = _status_payload(run_dir)
    if as_json:
        _echo_json(payload)
        return 0
    _echo_banner()
    _echo_kv_rows(
        "Run Status",
        [
            ("Run directory", payload["run_dir"]),
            ("Status", payload["status"]),
            ("Iteration", f'{payload["current_iteration"]}/{payload["iterations_requested"]}'),
            ("Best score", str(payload["best_score"])),
            ("Archive size", str(payload["archive_size"])),
            ("Duplicate skips", str(payload["duplicate_skips_total"])),
            ("Report", str(payload["report_path"] or "-")),
        ],
    )
    if payload["best_hypothesis_nl"]:
        _echo_block("Best hypothesis summary", str(payload["best_hypothesis_nl"]))
    return 0


@app.command(help="Return or regenerate the markdown report for one run.")
@click.argument("run_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--json", "as_json", is_flag=True, help="Print machine-readable JSON output.")
def report(run_dir: Path, as_json: bool) -> int:
    """Return the report path for one run directory."""
    report_path = run_dir / "report" / "report.md"
    if not report_path.exists():
        report_path = generate_run_report(run_dir).markdown_path
    payload = {"run_dir": str(run_dir), "report_path": str(report_path)}
    if as_json:
        _echo_json(payload)
        return 0
    click.echo(str(report_path))
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point that returns a shell-friendly exit code."""
    try:
        result = app.main(args=argv, prog_name="hypoevolve", standalone_mode=False)
        return 0 if result is None else int(result)
    except click.exceptions.Exit as exc:
        return exc.exit_code
    except click.ClickException as exc:
        exc.show()
        return exc.exit_code
    except click.Abort:
        click.echo("Aborted.", err=True)
        return 1


def _load_runtime_config(config: str | None) -> HypoEvolveConfig:
    """Load an explicit config or fall back to the default config path."""
    config_path = Path(config or DEFAULT_CONFIG_PATH)
    if config_path.exists():
        return load_config(config_path)
    if config is None:
        return HypoEvolveConfig()
    raise ConfigError(f"Config file not found: {config_path}")


def _render_banner() -> str:
    wordmark = click.style(CLI_WORDMARK, fg="cyan", bold=True)
    name = click.style("HypoEvolve", fg="cyan", bold=True)
    tagline = click.style(CLI_TAGLINE, fg="white", bold=True)
    subtitle = click.style(CLI_SUBTITLE, fg="yellow")
    rule = click.style(CLI_RULE, fg="blue")
    version = click.style(f"v{_detect_version()}", fg="yellow", bold=True)
    return f"{wordmark}\n{name} {version}\n{tagline}\n{subtitle}\n{rule}"


def _echo_banner() -> None:
    click.echo(_render_banner())


def _echo_json(payload: object) -> None:
    click.echo(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def _echo_kv_rows(title: str, rows: list[tuple[str, str]]) -> None:
    click.echo()
    click.secho(f"◆ {title}", fg="cyan", bold=True)
    width = max(len(label) for label, _ in rows)
    for label, value in rows:
        click.echo(f"  {label:<{width}}  {value}")


def _echo_block(title: str, body: str) -> None:
    click.echo()
    click.secho(f"◆ {title}", fg="cyan", bold=True)
    click.echo(body)


def _echo_error(exc: Exception) -> None:
    click.secho(f"Error: {exc}", fg="red", err=True)
    errors = getattr(exc, "errors", None)
    if errors:
        for item in errors:
            click.echo(f"  - {item}")


def _echo_metric_highlights(metrics: dict[str, object]) -> None:
    rows: list[tuple[str, str]] = []
    for label, key in (
        ("Combined score", "combined_score"),
        ("Precision", "precision"),
        ("Coverage", "coverage"),
        ("Uplift", "uplift"),
    ):
        value = metrics.get(key)
        if isinstance(value, (int, float)):
            rows.append((label, f"{value:.4f}"))
    if rows:
        _echo_kv_rows("Metric highlights", rows)


def _render_kv_section(title: str, rows: list[tuple[str, str]]) -> str:
    width = max(len(label) for label, _ in rows)
    lines = [f"{title}:"]
    lines.extend(f"  {label:<{width}}  {value}" for label, value in rows)
    return "\n".join(lines)


def _latest_run_dir(base_dir: Path) -> Path | None:
    if not base_dir.exists():
        return None
    candidates = [path for path in base_dir.iterdir() if path.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _resolve_runs_base_dir(config: str | None) -> Path:
    loaded = _load_runtime_config(config)
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


if __name__ == "__main__":
    sys.exit(main())
