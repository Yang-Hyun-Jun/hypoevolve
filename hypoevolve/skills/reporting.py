"""Compact post-run report generation for HypoEvolve runs."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any

from hypoevolve.elg import (
    AtomicNode,
    Hypothesis,
    LogicalNode,
    RelationNode,
    count_atomics,
    count_logicals,
    count_nodes,
    count_relations,
    hypothesis_from_dict,
    render_pretty,
    render_tree,
    tree_depth,
)


@dataclass(slots=True)
class ReportArtifacts:
    """Paths to the generated report files."""

    report_dir: Path
    markdown_path: Path
    score_plot_path: Path
    metrics_plot_path: Path
    archive_plot_path: Path


def generate_run_report(run_dir: Path) -> ReportArtifacts:
    """Generate a compact markdown report and SVG assets for one run.

    Args:
        run_dir: The run directory containing persisted artifacts.

    Returns:
        ReportArtifacts: Paths to the generated report outputs.
    """
    best_payload = _read_json(run_dir / "best.json")
    checkpoint_payload = _read_json(run_dir / "checkpoint.json")
    summary_payload = _read_json(run_dir / "run_summary.json")
    history_payload = _read_json(run_dir / "score_history.json")

    best_hypothesis = hypothesis_from_dict(best_payload["hypothesis"])
    best_metrics = dict(best_payload.get("metrics", {}))
    archive_entries = list(checkpoint_payload.get("archive", []))
    score_history = sorted(
        list(history_payload),
        key=lambda item: int(item.get("iteration", 0)),
    )
    evaluated_history = [
        item for item in score_history if item.get("score") is not None
    ]
    seed_entry = next(
        (item for item in evaluated_history if int(item.get("iteration", -1)) == 0),
        {},
    )
    best_entry = next(
        (
            item
            for item in evaluated_history
            if item.get("fingerprint") == summary_payload.get("best_fingerprint")
        ),
        evaluated_history[-1] if evaluated_history else {},
    )
    best_nl = str(
        summary_payload.get("best_hypothesis_nl")
        or best_entry.get("hypothesis_nl")
        or ""
    ).strip()
    if not best_nl:
        best_nl = render_pretty(best_hypothesis)

    report_dir = run_dir / "report"
    assets_dir = report_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    score_plot_path = assets_dir / "score_progression.svg"
    metrics_plot_path = assets_dir / "seed_vs_best_metrics.svg"
    archive_plot_path = assets_dir / "archive_distribution.svg"

    score_plot_path.write_text(
        _build_score_progression_svg(score_history),
        encoding="utf-8",
    )
    metrics_plot_path.write_text(
        _build_seed_vs_best_metrics_svg(seed_entry, best_entry, best_metrics),
        encoding="utf-8",
    )
    archive_plot_path.write_text(
        _build_archive_distribution_svg(archive_entries, summary_payload),
        encoding="utf-8",
    )

    markdown_path = report_dir / "report.md"
    markdown_path.write_text(
        _build_markdown_report(
            run_dir=run_dir,
            summary=summary_payload,
            best_hypothesis=best_hypothesis,
            best_metrics=best_metrics,
            best_nl=best_nl,
            seed_entry=seed_entry,
            archive_entries=archive_entries,
            score_history=score_history,
        ),
        encoding="utf-8",
    )

    return ReportArtifacts(
        report_dir=report_dir,
        markdown_path=markdown_path,
        score_plot_path=score_plot_path,
        metrics_plot_path=metrics_plot_path,
        archive_plot_path=archive_plot_path,
    )


def _build_markdown_report(
    *,
    run_dir: Path,
    summary: dict[str, Any],
    best_hypothesis: Hypothesis,
    best_metrics: dict[str, Any],
    best_nl: str,
    seed_entry: dict[str, Any],
    archive_entries: list[dict[str, Any]],
    score_history: list[dict[str, Any]],
) -> str:
    """Assemble the final markdown report body.

    Args:
        run_dir: The run directory being summarized.
        summary: The persisted run summary payload.
        best_hypothesis: The final best hypothesis.
        best_metrics: The final best metrics payload.
        best_nl: Natural-language text for the final best hypothesis.
        seed_entry: The seed entry from score history.
        archive_entries: The persisted archive snapshot.
        score_history: The persisted score history.

    Returns:
        str: The markdown report body.
    """
    structure = _hypothesis_structure(best_hypothesis)
    best_updates = [item for item in score_history if item.get("best_updated")]
    top_entries = archive_entries[:5]
    evaluated_count = sum(1 for item in score_history if item.get("score") is not None)
    duplicate_count = int(summary.get("duplicate_skips_total", 0))
    seed_score = _as_float(seed_entry.get("score"))
    best_score = _as_float(best_metrics.get("combined_score"))
    improvement = best_score - seed_score
    used_parameters = dict(best_metrics.get("used_parameters") or {})
    score_rationale = str(best_metrics.get("rationale") or "").strip()

    lines: list[str] = [
        "# HypoEvolve Final Report",
        "",
        "## Executive Summary",
        "",
        f"- Run directory: `{run_dir}`",
        f"- Seed hypothesis: {summary.get('seed_input_text', '')}",
        f"- Best ELG natural-language summary: {best_nl}",
        f"- Best combined score: `{best_score:.6f}`",
        f"- Seed → best score improvement: `{improvement:+.6f}`",
        f"- Evaluated candidates: `{evaluated_count}` / requested iterations `{summary.get('iterations_requested', 0)}`",
        f"- Duplicate skips: `{duplicate_count}`",
        "",
        "## Key Metrics",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    for key in ("combined_score", "precision", "baseline", "coverage", "uplift"):
        lines.append(f"| {key} | {_format_metric(best_metrics.get(key))} |")
    lines.append(f"| support_count | `{best_metrics.get('support_count', 0)}` |")
    lines.append(f"| total_count | `{best_metrics.get('total_count', 0)}` |")
    lines.append("")
    if score_rationale:
        lines.append(f"> {score_rationale}")
        lines.append("")
    if used_parameters:
        lines.extend(
            [
                "### Evaluator Parameters",
                "",
                "| Parameter | Value |",
                "| --- | ---: |",
            ]
        )
        for key, value in sorted(used_parameters.items()):
            lines.append(f"| {key} | `{value}` |")
        lines.append("")

    lines.extend(
        [
            "## Best ELG",
            "",
            "### Structure Summary",
            "",
            "| Field | Value |",
            "| --- | --- |",
        ]
    )
    for field, value in structure.items():
        lines.append(f"| {field} | {value} |")
    lines.extend(
        [
            "",
            "### Pretty ELG",
            "",
            "```text",
            render_pretty(best_hypothesis),
            "```",
            "",
            "### ELG Tree",
            "",
            "```text",
            render_tree(best_hypothesis),
            "```",
            "",
            "### Atomic Propositions",
            "",
        ]
    )
    for atomic_name in _collect_atomic_names(best_hypothesis.root):
        lines.append(f"- `{atomic_name}`")
    lines.extend(
        [
            "",
            "## Search Overview",
            "",
            "| Field | Value |",
            "| --- | --- |",
            f"| workers | `{summary.get('worker_count', 1)}` |",
            f"| archive size | `{summary.get('archive_size', 0)}` |",
            f"| occupied cells | `{summary.get('occupied_cells', 0)}` |",
            f"| best iteration | `{summary.get('best_iteration', 0)}` |",
            f"| best fingerprint | `{summary.get('best_fingerprint', '')}` |",
            f"| occupancy summary | `{summary.get('occupancy_summary', '')}` |",
            "",
            "## Visualizations",
            "",
            "### Score Progression",
            "",
            "![Score progression](assets/score_progression.svg)",
            "",
            "### Seed vs Best Metric Profile",
            "",
            "![Seed vs best metrics](assets/seed_vs_best_metrics.svg)",
            "",
            "### Archive Distribution",
            "",
            "![Archive distribution](assets/archive_distribution.svg)",
            "",
        ]
    )

    if best_updates:
        lines.extend(
            [
                "## Best-Score Update Timeline",
                "",
                "| Iteration | Score | Mutation summary |",
                "| ---: | ---: | --- |",
            ]
        )
        for item in best_updates:
            lines.append(
                f"| {item.get('iteration', 0)} | {_format_metric(item.get('score'))} | {_sanitize_pipe_text(item.get('mutation_summary', 'seed initialization'))} |"
            )
        lines.append("")

    if top_entries:
        lines.extend(
            [
                "## Top Archive Candidates",
                "",
                "| Rank | Iteration | Score | Coverage | Complexity | Hypothesis |",
                "| ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for index, entry in enumerate(top_entries, start=1):
            hypothesis_nl = _sanitize_pipe_text(
                str(
                    entry.get("metadata", {}).get("hypothesis_nl")
                    or render_pretty(hypothesis_from_dict(entry["hypothesis"]))
                )
            )
            lines.append(
                "| {rank} | {iteration} | {score} | {coverage} | {complexity} | {hypothesis} |".format(
                    rank=index,
                    iteration=entry.get("iteration", 0),
                    score=_format_metric(
                        entry.get("metrics", {}).get("combined_score")
                    ),
                    coverage=_format_metric(entry.get("coverage", 0.0)),
                    complexity=entry.get("complexity", 0),
                    hypothesis=hypothesis_nl,
                )
            )
        lines.append("")

    lines.extend(["## Artifact Checklist", ""])
    for name in (
        "trace.jsonl",
        "checkpoint.json",
        "best.json",
        "run_summary.json",
        "score_history.json",
    ):
        lines.append(f"- `{name}`")
    lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _build_score_progression_svg(score_history: list[dict[str, Any]]) -> str:
    """Build the score-progression SVG chart.

    Args:
        score_history: The persisted run score history.

    Returns:
        str: The SVG document as text.
    """
    evaluated = [item for item in score_history if item.get("score") is not None]
    width, height = 920, 360
    margin_left, margin_right, margin_top, margin_bottom = 72, 28, 54, 52
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    if not evaluated:
        return _empty_svg(width, height, "No evaluated scores recorded")

    xs = [int(item.get("iteration", 0)) for item in evaluated]
    candidate_scores = [_as_float(item.get("score")) for item in evaluated]
    best_scores = [
        _as_float(item.get("best_score_after"), default=_as_float(item.get("score")))
        for item in evaluated
    ]
    y_values = candidate_scores + best_scores
    y_min, y_max = _expand_numeric_range(min(y_values), max(y_values))
    x_min, x_max = min(xs), max(xs)
    if x_min == x_max:
        x_min -= 1
        x_max += 1

    def x_pos(value: int) -> float:
        return margin_left + ((value - x_min) / (x_max - x_min)) * plot_width

    def y_pos(value: float) -> float:
        return margin_top + (1 - ((value - y_min) / (y_max - y_min))) * plot_height

    candidate_path = " ".join(
        ("M" if index == 0 else "L") + f" {x_pos(iteration):.2f} {y_pos(score):.2f}"
        for index, (iteration, score) in enumerate(zip(xs, candidate_scores))
    )
    best_path = " ".join(
        ("M" if index == 0 else "L") + f" {x_pos(iteration):.2f} {y_pos(score):.2f}"
        for index, (iteration, score) in enumerate(zip(xs, best_scores))
    )
    best_index = max(
        range(len(candidate_scores)), key=lambda idx: candidate_scores[idx]
    )
    grid_lines: list[str] = []
    labels: list[str] = []
    for step in range(5):
        ratio = step / 4
        y_value = y_min + (y_max - y_min) * ratio
        y = y_pos(y_value)
        grid_lines.append(
            f'<line x1="{margin_left}" y1="{y:.2f}" x2="{width - margin_right}" y2="{y:.2f}" stroke="#e5e7eb" stroke-width="1"/>'
        )
        labels.append(
            f'<text x="{margin_left - 10}" y="{y + 4:.2f}" text-anchor="end" font-size="11" fill="#4b5563">{_format_short_number(y_value)}</text>'
        )
    x_labels: list[str] = []
    tick_values = sorted(set([x_min, x_max, *xs]))
    for value in tick_values[:8]:
        x = x_pos(value)
        x_labels.append(
            f'<line x1="{x:.2f}" y1="{height - margin_bottom}" x2="{x:.2f}" y2="{height - margin_bottom + 6}" stroke="#6b7280" stroke-width="1"/>'
        )
        x_labels.append(
            f'<text x="{x:.2f}" y="{height - 14}" text-anchor="middle" font-size="11" fill="#4b5563">{value}</text>'
        )
    circles = [
        f'<circle cx="{x_pos(iteration):.2f}" cy="{y_pos(score):.2f}" r="3.5" fill="#2563eb"/>'
        for iteration, score in zip(xs, candidate_scores)
    ]
    circles.append(
        f'<circle cx="{x_pos(xs[best_index]):.2f}" cy="{y_pos(candidate_scores[best_index]):.2f}" r="5.5" fill="#dc2626" stroke="#ffffff" stroke-width="2"/>'
    )

    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="Score progression">
  <rect width="100%" height="100%" fill="#ffffff"/>
  <text x="{margin_left}" y="28" font-size="20" font-weight="700" fill="#111827">Score progression by iteration</text>
  <text x="{margin_left}" y="46" font-size="12" fill="#4b5563">Blue = evaluated candidate score, Green = best-so-far score, Red = final best point</text>
  {"".join(grid_lines)}
  <line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height - margin_bottom}" stroke="#6b7280" stroke-width="1.25"/>
  <line x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - margin_right}" y2="{height - margin_bottom}" stroke="#6b7280" stroke-width="1.25"/>
  {"".join(labels)}
  {"".join(x_labels)}
  <path d="{candidate_path}" fill="none" stroke="#2563eb" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
  <path d="{best_path}" fill="none" stroke="#059669" stroke-width="2.5" stroke-dasharray="7 5" stroke-linecap="round" stroke-linejoin="round"/>
  {"".join(circles)}
</svg>"""


def _build_seed_vs_best_metrics_svg(
    seed_entry: dict[str, Any],
    best_entry: dict[str, Any],
    best_metrics: dict[str, Any],
) -> str:
    """Build the seed-versus-best metrics SVG chart.

    Args:
        seed_entry: The seed score-history entry.
        best_entry: The best score-history entry.
        best_metrics: The persisted best metrics payload.

    Returns:
        str: The SVG document as text.
    """
    width, height = 920, 360
    margin_left, margin_right, margin_top, margin_bottom = 72, 28, 56, 54
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    metrics = ["precision", "baseline", "coverage", "uplift"]
    seed_values = [_as_float(seed_entry.get(key)) for key in metrics]
    best_values = [
        _as_float(best_entry.get(key), default=_as_float(best_metrics.get(key)))
        for key in metrics
    ]
    all_values = seed_values + best_values + [0.0]
    y_min, y_max = _expand_numeric_range(min(all_values), max(all_values))

    def y_pos(value: float) -> float:
        return margin_top + (1 - ((value - y_min) / (y_max - y_min))) * plot_height

    zero_y = y_pos(0.0)
    group_width = plot_width / max(len(metrics), 1)
    bar_width = min(28.0, group_width * 0.28)
    grid_lines: list[str] = []
    labels: list[str] = []
    for step in range(5):
        ratio = step / 4
        y_value = y_min + (y_max - y_min) * ratio
        y = y_pos(y_value)
        grid_lines.append(
            f'<line x1="{margin_left}" y1="{y:.2f}" x2="{width - margin_right}" y2="{y:.2f}" stroke="#e5e7eb" stroke-width="1"/>'
        )
        labels.append(
            f'<text x="{margin_left - 10}" y="{y + 4:.2f}" text-anchor="end" font-size="11" fill="#4b5563">{_format_short_number(y_value)}</text>'
        )

    bars: list[str] = []
    x_labels: list[str] = []
    for index, metric in enumerate(metrics):
        center = margin_left + group_width * index + group_width / 2
        seed_x = center - bar_width - 4
        best_x = center + 4
        for x, value, color in (
            (seed_x, seed_values[index], "#2563eb"),
            (best_x, best_values[index], "#059669"),
        ):
            top = min(y_pos(value), zero_y)
            rect_height = max(abs(zero_y - y_pos(value)), 1.5)
            bars.append(
                f'<rect x="{x:.2f}" y="{top:.2f}" width="{bar_width:.2f}" height="{rect_height:.2f}" rx="3" fill="{color}" opacity="0.9"/>'
            )
        x_labels.append(
            f'<text x="{center:.2f}" y="{height - 18}" text-anchor="middle" font-size="12" fill="#374151">{metric}</text>'
        )

    combined_seed = _as_float(seed_entry.get("score"))
    combined_best = _as_float(best_metrics.get("combined_score"))
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="Seed versus best metrics">
  <rect width="100%" height="100%" fill="#ffffff"/>
  <text x="{margin_left}" y="28" font-size="20" font-weight="700" fill="#111827">Seed vs best metric profile</text>
  <text x="{margin_left}" y="46" font-size="12" fill="#4b5563">Combined score improved from {_format_short_number(combined_seed)} to {_format_short_number(combined_best)}</text>
  {"".join(grid_lines)}
  <line x1="{margin_left}" y1="{zero_y:.2f}" x2="{width - margin_right}" y2="{zero_y:.2f}" stroke="#6b7280" stroke-width="1.25"/>
  <line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height - margin_bottom}" stroke="#6b7280" stroke-width="1.25"/>
  {"".join(labels)}
  {"".join(bars)}
  {"".join(x_labels)}
  <rect x="{width - 220}" y="20" width="12" height="12" rx="2" fill="#2563eb"/><text x="{width - 202}" y="30" font-size="12" fill="#374151">Seed</text>
  <rect x="{width - 140}" y="20" width="12" height="12" rx="2" fill="#059669"/><text x="{width - 122}" y="30" font-size="12" fill="#374151">Best</text>
</svg>"""


def _build_archive_distribution_svg(
    archive_entries: list[dict[str, Any]],
    summary: dict[str, Any],
) -> str:
    """Build the archive-distribution SVG scatter plot.

    Args:
        archive_entries: The persisted archive snapshot.
        summary: The persisted run summary.

    Returns:
        str: The SVG document as text.
    """
    width, height = 920, 360
    margin_left, margin_right, margin_top, margin_bottom = 72, 28, 54, 52
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    if not archive_entries:
        return _empty_svg(width, height, "Archive snapshot is empty")

    coverages = [_as_float(entry.get("coverage")) for entry in archive_entries]
    scores = [
        _as_float(entry.get("metrics", {}).get("combined_score"))
        for entry in archive_entries
    ]
    y_min, y_max = _expand_numeric_range(min(scores), max(scores))
    x_min, x_max = 0.0, max(max(coverages), 0.01)
    if math.isclose(x_min, x_max):
        x_max = x_min + 1.0

    def x_pos(value: float) -> float:
        return margin_left + ((value - x_min) / (x_max - x_min)) * plot_width

    def y_pos(value: float) -> float:
        return margin_top + (1 - ((value - y_min) / (y_max - y_min))) * plot_height

    grid_lines: list[str] = []
    labels: list[str] = []
    for step in range(5):
        ratio = step / 4
        y_value = y_min + (y_max - y_min) * ratio
        y = y_pos(y_value)
        grid_lines.append(
            f'<line x1="{margin_left}" y1="{y:.2f}" x2="{width - margin_right}" y2="{y:.2f}" stroke="#e5e7eb" stroke-width="1"/>'
        )
        labels.append(
            f'<text x="{margin_left - 10}" y="{y + 4:.2f}" text-anchor="end" font-size="11" fill="#4b5563">{_format_short_number(y_value)}</text>'
        )
    x_labels: list[str] = []
    for step in range(5):
        ratio = step / 4
        x_value = x_min + (x_max - x_min) * ratio
        x = x_pos(x_value)
        x_labels.append(
            f'<line x1="{x:.2f}" y1="{height - margin_bottom}" x2="{x:.2f}" y2="{height - margin_bottom + 6}" stroke="#6b7280" stroke-width="1"/>'
        )
        x_labels.append(
            f'<text x="{x:.2f}" y="{height - 14}" text-anchor="middle" font-size="11" fill="#4b5563">{_format_short_number(x_value)}</text>'
        )

    best_fp = str(summary.get("best_fingerprint") or "")
    points: list[str] = []
    for entry in archive_entries:
        coverage = _as_float(entry.get("coverage"))
        score = _as_float(entry.get("metrics", {}).get("combined_score"))
        complexity = int(entry.get("complexity", 0))
        is_best = entry.get("fingerprint") == best_fp
        fill = "#dc2626" if is_best else "#2563eb"
        radius = 6 if is_best else 4.5
        px = x_pos(coverage)
        py = y_pos(score)
        points.append(
            f'<circle cx="{px:.2f}" cy="{py:.2f}" r="{radius}" fill="{fill}" opacity="0.88"/>'
        )
        label = "best" if is_best else f"c{complexity}"
        points.append(
            f'<text x="{px + 7:.2f}" y="{py - 7:.2f}" font-size="11" fill="#374151">{escape(label)}</text>'
        )

    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="Archive distribution">
  <rect width="100%" height="100%" fill="#ffffff"/>
  <text x="{margin_left}" y="28" font-size="20" font-weight="700" fill="#111827">Archive distribution</text>
  <text x="{margin_left}" y="46" font-size="12" fill="#4b5563">Coverage on x-axis, combined score on y-axis, non-best labels show complexity</text>
  {"".join(grid_lines)}
  <line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height - margin_bottom}" stroke="#6b7280" stroke-width="1.25"/>
  <line x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - margin_right}" y2="{height - margin_bottom}" stroke="#6b7280" stroke-width="1.25"/>
  {"".join(labels)}
  {"".join(x_labels)}
  {"".join(points)}
</svg>"""


def _hypothesis_structure(hypothesis: Hypothesis) -> dict[str, str]:
    root = hypothesis.root
    relation = (
        root.name.value if isinstance(root, RelationNode) else type(root).__name__
    )
    condition = render_pretty(root.condition) if isinstance(root, RelationNode) else "-"
    target = render_pretty(root.target) if isinstance(root, RelationNode) else "-"
    return {
        "relation": relation,
        "node_count": f"`{count_nodes(hypothesis)}`",
        "atomic_count": f"`{count_atomics(hypothesis)}`",
        "logical_count": f"`{count_logicals(hypothesis)}`",
        "relation_count": f"`{count_relations(hypothesis)}`",
        "tree_depth": f"`{tree_depth(hypothesis)}`",
        "condition": f"`{condition}`",
        "target": f"`{target}`",
    }


def _collect_atomic_names(node: AtomicNode | LogicalNode | RelationNode) -> list[str]:
    if isinstance(node, AtomicNode):
        return [node.name]
    names: list[str] = []
    for child in getattr(node, "inputs", []):
        names.extend(_collect_atomic_names(child))
    return names


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _as_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else default
    return default


def _expand_numeric_range(low: float, high: float) -> tuple[float, float]:
    if math.isclose(low, high):
        padding = 1.0 if math.isclose(low, 0.0) else abs(low) * 0.25
        return low - padding, high + padding
    span = high - low
    padding = span * 0.15
    return low - padding, high + padding


def _format_metric(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return f"`{value}`"
    if isinstance(value, int):
        return f"`{value}`"
    if isinstance(value, float):
        return f"`{value:.6f}`"
    return f"`{value}`"


def _format_short_number(value: float) -> str:
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    if abs(value) >= 1:
        return f"{value:.3f}"
    return f"{value:.4f}"


def _sanitize_pipe_text(text: Any) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ").strip()


def _empty_svg(width: int, height: int, message: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-label="Empty chart">'
        '<rect width="100%" height="100%" fill="#ffffff"/>'
        f'<text x="50%" y="50%" text-anchor="middle" font-size="18" fill="#6b7280">{escape(message)}</text>'
        "</svg>"
    )
