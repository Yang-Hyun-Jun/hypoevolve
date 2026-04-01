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
from hypoevolve.llm import LLMClient
from hypoevolve.logger import configure_logger, logger
from hypoevolve.parser import ParseError, parse_hypothesis_text

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
CLI_EXAMPLES = (
    "Quick start:\n"
    "  hypoevolve --version\n"
    "  hypoevolve run \"if BTC momentum drops then DOGE jumps\"\n"
    "  hypoevolve render \"if A then B\" --tree\n"
    "  hypoevolve inspect .hypoevolve/runs/latest/best.json\n"
    "  hypoevolve doctor"
)
CLI_TIPS = (
    "Tips:\n"
    "  - Bare `hypoevolve` prints help and exits non-zero to preserve script safety.\n"
    "  - Use `inspect` to review saved run artifacts without rerunning the pipeline."
)


def _detect_version() -> str:
    try:
        return metadata.version("openresearch")
    except metadata.PackageNotFoundError:
        return "dev"


class StyledGroup(click.Group):
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
@click.argument("hypothesis")
@click.option("--config", default="hypoevolve.yaml", show_default=True, help=CONFIG_HELP)
@click.option("--workers", type=int, default=None, help="Override the local worker count for this run.")
def run(hypothesis: str, config: str, workers: int | None) -> int:
    try:
        loaded = _load_runtime_config(config)
        configure_logger(loaded.logging.level)
        if workers is not None:
            loaded.workers.count = workers
            loaded.workers.enabled = workers > 1
        logger.info("cli run command started")
        result = HypoEvolveController(loaded).run(hypothesis)
        _echo_banner()
        _echo_kv_rows(
            "Run Summary",
            [
                ("Status", "ok"),
                ("Run directory", str(result.run_dir)),
                ("Workers", str(loaded.workers.count)),
            ],
        )
        _echo_metric_highlights(result.best_metrics)
        _echo_block("Best hypothesis", render_pretty(result.best_hypothesis))
        _echo_block(
            "Best metrics",
            json.dumps(result.best_metrics, ensure_ascii=False, indent=2, sort_keys=True),
        )
        logger.info("cli run command completed")
        return 0
    except (ConfigError, ParseError) as exc:
        logger.error("cli run command failed: {}", exc)
        _echo_error(exc)
        return 1


@app.command(help="Parse and render a natural-language hypothesis via the LLM parser.")
@click.argument("hypothesis")
@click.option("--config", default="hypoevolve.yaml", show_default=True, help=CONFIG_HELP)
@click.option("--tree", is_flag=True, help="Render as an ASCII tree instead of the pretty ELG form.")
def render(hypothesis: str, config: str, tree: bool) -> int:
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
def inspect(path: Path) -> int:
    payload = json.loads(path.read_text(encoding="utf-8"))
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
@click.option("--config", default="hypoevolve.yaml", show_default=True, help=CONFIG_HELP)
def doctor(config: str) -> int:
    _echo_banner()
    config_path = Path(config)
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
        rows.append(("Config ok", "using defaults"))
    _echo_kv_rows("Diagnostics", rows)
    return 0


def main(argv: list[str] | None = None) -> int:
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


def _load_runtime_config(config: str) -> HypoEvolveConfig:
    config_path = Path(config)
    return load_config(config_path) if config_path.exists() else HypoEvolveConfig()


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


if __name__ == "__main__":
    sys.exit(main())
