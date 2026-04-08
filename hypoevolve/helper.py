"""Prompt-variable builders shared by parser, evaluator, and steering flows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Mapping, Sequence

from elg import Hypothesis, render_pretty
from hypoevolve.archive import ArchiveEntry
from hypoevolve.dataset import DatasetAccessor, DatasetSchema


def build_evaluator_prompt_variables(
    hypothesis: Hypothesis,
    schema: DatasetSchema,
    accessor: DatasetAccessor,
) -> Dict[str, str]:
    """Build the string variables used in evaluator prompt templates."""
    column_specs = "\n".join(
        f"- {column.name}: {column.description or ''}".rstrip()
        for column in schema.columns
    )

    accessor_doc = (
        "DatasetAccessor methods:\n"
        "- accessor.entities() -> list[str]\n"
        "- accessor.column_names() -> list[str]\n"
        "- accessor.column_descriptions() -> dict[str, str]\n"
        "- accessor.load_dataframe(entity) -> pandas.DataFrame\n"
        "- accessor.load_all_dataframes() -> dict[str, pandas.DataFrame]\n"
        "- accessor.head(entity, n=5) -> pandas.DataFrame\n"
        "- accessor.summary() -> dict"
    )

    return {
        "HYPOTHESIS_PRETTY": render_pretty(hypothesis),
        "DATASET_DESCRIPTION": schema.description or "",
        "INDEX_NAME": schema.index.name or "",
        "INDEX_DTYPE": schema.index.dtype or "",
        "ENTITIES": ", ".join(accessor.entities()),
        "COLUMN_SPECS": column_specs,
        "DATASET_ACCESSOR_DOC": accessor_doc,
    }


def build_evaluator_runtime_wrapper(
    dataset_schema_path: str | Path,
    parameters: dict[str, object] | None = None,
) -> str:
    """Build the wrapper script that executes generated evaluator code."""
    parameters_literal = "None"

    if parameters is not None:
        parameters_literal = json.dumps(parameters, ensure_ascii=False)
    schema_path_literal = json.dumps(
        str(Path(dataset_schema_path).resolve()), ensure_ascii=False
    )
    project_root_literal = json.dumps(
        str(Path(__file__).resolve().parent.parent), ensure_ascii=False
    )
    return (
        "import io\n"
        "import json\n"
        "import sys\n"
        "from contextlib import redirect_stdout\n"
        f"sys.path.insert(0, {project_root_literal})\n"
        "from hypoevolve.dataset import DatasetAccessor, load_dataset_schema\n"
        "from candidate import evaluate_hypothesis\n\n"
        f"schema = load_dataset_schema({schema_path_literal})\n"
        "accessor = DatasetAccessor(schema)\n"
        f"parameters = {parameters_literal}\n"
        "buffer = io.StringIO()\n"
        "with redirect_stdout(buffer):\n"
        "    result = evaluate_hypothesis(accessor, parameters=parameters)\n"
        "print(json.dumps(result, ensure_ascii=False))\n"
    )


def build_steering_prompt_variables(
    parent_hypothesis: Hypothesis,
    current_metrics: Mapping[str, object],
    recent_history: Sequence[Mapping[str, object]] | None = None,
    top_hypotheses: Sequence[ArchiveEntry] | None = None,
) -> Dict[str, str]:
    """Build the string variables used in mutation-steering prompts."""
    metric_definitions = (
        "- precision = P(target | condition)\n"
        "- baseline = P(target)\n"
        "- coverage = P(condition)\n"
        "- uplift = precision - baseline\n"
        "- combined_score = uplift * coverage"
    )

    history_payload = list(recent_history or [])
    top_payload = [
        {
            "hypothesis": render_pretty(entry.hypothesis),
            "score": entry.score,
            "metrics": dict(entry.metrics),
            "iteration": entry.iteration,
            "metadata": dict(entry.metadata),
        }
        for entry in (top_hypotheses or [])
    ]
    return {
        "PARENT_HYPOTHESIS_MEASURABLE": render_pretty(parent_hypothesis),
        "CURRENT_METRICS": json.dumps(
            dict(current_metrics), ensure_ascii=False, indent=2
        ),
        "METRIC_DEFINITIONS": metric_definitions,
        "RECENT_HISTORY": json.dumps(history_payload, ensure_ascii=False, indent=2),
        "TOP_HYPOTHESES": json.dumps(top_payload, ensure_ascii=False, indent=2),
    }
