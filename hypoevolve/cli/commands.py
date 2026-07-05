"""Top-level commands for the HypoEvolve CLI."""

from __future__ import annotations

import json
import platform
from pathlib import Path

import click

from hypoevolve.core.config import (
    ConfigError,
    load_config,
    load_runtime_config,
    resolve_config_path,
)
from hypoevolve.core.orchestrator import HypoEvolveController
from hypoevolve.data.dataset import DatasetSchemaError
from hypoevolve.elg import hypothesis_from_dict, render_pretty, render_tree
from hypoevolve.observability.logger import (
    configure_logger,
    log_error_event,
    log_info_event,
    summarize_exception,
)
from hypoevolve.runtime.llm_client import LLMClient
from hypoevolve.skills.elg_compile import ParseError, parse_hypothesis_text
from hypoevolve.skills.seed_generation import (
    HypothesisGenerationError,
    generate_random_tree_pair_hypothesis,
)

from .display import (
    _echo_banner,
    _echo_block,
    _echo_error,
    _echo_json,
    _echo_kv_rows,
    _echo_metric_highlights,
)

CONFIG_HELP = "Path to the project config file."
DEFAULT_CONFIG_PATH = "hypoevolve.yaml"
CLI_EXAMPLES = (
    "Quick start:\n"
    "  hypoevolve --version\n"
    '  hypoevolve run "if signal A weakens then event B becomes more likely"\n'
    "  hypoevolve run\n"
    "  hypoevolve seed\n"
    '  hypoevolve render "if A then B" --tree\n'
    "  hypoevolve inspect .hypoevolve/runs/latest/best.json\n"
    "  hypoevolve runs status <run-id> --json\n"
    "  hypoevolve doctor"
)
CLI_TIPS = (
    "Tips:\n"
    "  - Bare `hypoevolve` prints help and exits non-zero to preserve script safety.\n"
    "  - Use `inspect` to review saved run artifacts without rerunning the pipeline."
)


@click.command(help="Run the hypothesis evolution loop from a natural-language prompt.")
@click.argument("hypothesis", required=False)
@click.option(
    "--config", default=None, help=f"{CONFIG_HELP} Defaults to {DEFAULT_CONFIG_PATH}."
)
@click.option(
    "--workers",
    type=int,
    default=None,
    help="Override the local worker count for this run.",
)
def run(hypothesis: str | None, config: str | None, workers: int | None) -> int:
    """Run the hypothesis evolution loop from one natural-language seed."""
    try:
        loaded = load_runtime_config(config, default_path=DEFAULT_CONFIG_PATH)
        configure_logger(loaded.logging.level)
        if workers is not None:
            loaded.workers.count = workers
            loaded.workers.enabled = workers > 1
        log_info_event("cli.run.start", workers=loaded.workers.count)
        result = HypoEvolveController(loaded).run(hypothesis)
        seed_generated = bool(getattr(result, "seed_generated", False))
        seed_input_text = str(getattr(result, "seed_input_text", "") or "")
        seed_hypothesis = getattr(result, "seed_hypothesis", None)
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
        if seed_hypothesis is not None:
            _echo_block("Initial hypothesis", render_pretty(seed_hypothesis))
        _echo_block("Best hypothesis", render_pretty(result.best_hypothesis))
        _echo_block(
            "Best metrics",
            json.dumps(
                result.best_metrics, ensure_ascii=False, indent=2, sort_keys=True
            ),
        )
        log_info_event("cli.run.ok", run_dir=result.run_dir)
        return 0
    except (
        ConfigError,
        DatasetSchemaError,
        ParseError,
        HypothesisGenerationError,
    ) as exc:
        log_error_event("cli.run.fail", **summarize_exception(exc))
        _echo_error(exc)
        return 1


@click.command(
    help="Generate one random natural-language seed hypothesis from sampled feature trees."
)
@click.option(
    "--config", default=None, help=f"{CONFIG_HELP} Defaults to {DEFAULT_CONFIG_PATH}."
)
@click.option(
    "--max-depth",
    type=int,
    default=3,
    show_default=True,
    help="Maximum sampled tree depth.",
)
def seed(config: str | None, max_depth: int) -> int:
    """Generate one random seed hypothesis without running evolution."""
    try:
        loaded = load_runtime_config(config, default_path=DEFAULT_CONFIG_PATH)
        configure_logger(loaded.logging.level)
        log_info_event("cli.seed.start", max_depth=max_depth)
        result = generate_random_tree_pair_hypothesis(
            llm=LLMClient(loaded.llm),
            max_depth=max_depth,
            dataset_schema_path=loaded.evaluator.dataset_schema_path,
        )
        _echo_banner()
        _echo_block("Feature tree A", result.tree_a.render(return_str=True))
        _echo_block("Feature tree B", result.tree_b.render(return_str=True))
        _echo_block("Generated hypothesis", result.hypothesis)
        log_info_event("cli.seed.ok", chars=len(result.hypothesis))
        return 0
    except (ConfigError, DatasetSchemaError, HypothesisGenerationError) as exc:
        log_error_event("cli.seed.fail", **summarize_exception(exc))
        _echo_error(exc)
        return 1


@click.command(help="Parse and render a natural-language hypothesis via the LLM parser.")
@click.argument("hypothesis")
@click.option(
    "--config", default=None, help=f"{CONFIG_HELP} Defaults to {DEFAULT_CONFIG_PATH}."
)
@click.option(
    "--tree",
    is_flag=True,
    help="Render as an ASCII tree instead of the pretty ELG form.",
)
def render(hypothesis: str, config: str | None, tree: bool) -> int:
    """Parse one hypothesis and render it as ELG text or an ASCII tree."""
    try:
        loaded = load_runtime_config(config, default_path=DEFAULT_CONFIG_PATH)
        configure_logger(loaded.logging.level)
        log_info_event("cli.render.start", tree=tree)
        parsed = parse_hypothesis_text(
            hypothesis,
            llm=LLMClient(loaded.llm),
            retries=loaded.parser.retries,
        )
        _echo_banner()
        _echo_block(
            "Rendered hypothesis",
            render_tree(parsed) if tree else render_pretty(parsed),
        )
        log_info_event("cli.render.ok")
        return 0
    except (ConfigError, ParseError) as exc:
        log_error_event("cli.render.fail", **summarize_exception(exc))
        _echo_error(exc)
        return 1


@click.command(help="Inspect a saved best/checkpoint JSON artifact.")
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
                json.dumps(
                    payload["metrics"], ensure_ascii=False, indent=2, sort_keys=True
                ),
            )
    elif "best_hypothesis" in payload and payload["best_hypothesis"] is not None:
        hypothesis = hypothesis_from_dict(payload["best_hypothesis"])
        _echo_block("Best hypothesis", render_pretty(hypothesis))
        if "best_metrics" in payload:
            _echo_block(
                "Best metrics",
                json.dumps(
                    payload["best_metrics"],
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                ),
            )
    else:
        _echo_block(
            "Payload", json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
        )
    return 0


@click.command(help="Show environment and config diagnostics.")
@click.option(
    "--config", default=None, help=f"{CONFIG_HELP} Defaults to {DEFAULT_CONFIG_PATH}."
)
def doctor(config: str | None) -> int:
    """Report environment and configuration diagnostics for the CLI."""
    _echo_banner()
    config_path = resolve_config_path(config, DEFAULT_CONFIG_PATH)
    rows = [
        ("Python", platform.python_version()),
        ("Platform", platform.platform()),
        ("Config", str(config_path)),
        ("Config exists", str(config_path.exists()).lower()),
    ]
    if config_path.exists():
        try:
            loaded = load_config(config_path)
            archive_type_label = {
                "map_elites": "MAP-Elites",
                "coulomb": "Coulomb",
            }.get(loaded.archive.kind, loaded.archive.kind)
            rows.extend(
                [
                    ("Config ok", "true"),
                    ("Archive type", archive_type_label),
                    ("LLM model", loaded.llm.model),
                    ("LLM api base", loaded.llm.api_base),
                    ("LLM api key", "configured" if loaded.llm.api_key else "auto"),
                ]
            )
            if loaded.archive.kind == "coulomb":
                rows.extend(
                    [
                        ("Coulomb capacity", str(loaded.archive.coulomb.capacity)),
                        ("Coulomb gamma", str(loaded.archive.coulomb.gamma)),
                        ("Coulomb eps", str(loaded.archive.coulomb.eps)),
                    ]
                )
            else:
                rows.extend(
                    [
                        ("Coverage bins", str(loaded.archive.coverage_bins)),
                        ("Complexity bins", str(loaded.archive.complexity_bins)),
                        ("Parent sampling", loaded.archive.parent_sampling_mode),
                    ]
                )
            rows.extend(
                [
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
