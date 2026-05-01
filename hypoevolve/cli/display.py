"""Display helpers and constants for the HypoEvolve CLI."""

from __future__ import annotations

import importlib.metadata as metadata
import json

import click

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
CLI_RULE = "\u2500" * 56


def _detect_version() -> str:
    """Return the installed package version or ``dev`` for local checkouts."""
    try:
        return metadata.version("openresearch")
    except metadata.PackageNotFoundError:
        return "dev"


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
    click.secho(f"\u25c6 {title}", fg="cyan", bold=True)
    width = max(len(label) for label, _ in rows)
    for label, value in rows:
        click.echo(f"  {label:<{width}}  {value}")


def _echo_block(title: str, body: str) -> None:
    click.echo()
    click.secho(f"\u25c6 {title}", fg="cyan", bold=True)
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
