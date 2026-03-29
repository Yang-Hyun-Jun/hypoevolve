from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Mapping, Sequence

from elg import Hypothesis, MutationSample, render_pretty
from hypoevolve.archive import ArchiveEntry
from hypoevolve.dataset import DatasetAccessor, DatasetSchema

_MUTATION_OPERATION_GUIDE = {
    "wrap_not": (
        "Wrap the selected node with NOT(...).",
        "Example: A -> NOT(A)",
    ),
    "unwrap_not": (
        "Remove an existing NOT(...) wrapper.",
        "Example: NOT(A) -> A",
    ),
    "replace_atomic": (
        "Replace one atomic proposition with another atomic proposition.",
        "Example: X < -2.0 -> X < -1.5",
    ),
    "change_logical_operator": (
        "Change a logical operator such as AND <-> OR.",
        "Example: AND(A, B) -> OR(A, B)",
    ),
    "append_child": (
        "Add one child proposition to an AND/OR node.",
        "Example: AND(A, B) -> AND(A, B, C)",
    ),
    "remove_child": (
        "Remove one child proposition from an AND/OR node.",
        "Example: AND(A, B, C) -> AND(A, B)",
    ),
    "change_relation_type": (
        "Change the relation type.",
        "Example: IMPLIES(A, B) -> SUPPORT(A, B)",
    ),
}


def build_evaluator_prompt_variables(
    hypothesis: Hypothesis,
    schema: DatasetSchema,
    accessor: DatasetAccessor,
) -> Dict[str, str]:
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
    parent_hypothesis_nl: str,
    current_metrics: Mapping[str, object],
    mutation_candidates: Sequence[MutationSample],
    recent_history: Sequence[Mapping[str, object]] | None = None,
    top_hypotheses: Sequence[ArchiveEntry] | None = None,
) -> Dict[str, str]:
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
    candidate_lines = []
    path_guide = (
        "Path notation guide:\n"
        "- [] means the root node\n"
        "- [0] means the first child of the root\n"
        "- [1] means the second child of the root\n"
        "- [0, 1] means the second child of the first child of the root"
    )
    for index, candidate in enumerate(mutation_candidates):
        meaning, example = _MUTATION_OPERATION_GUIDE.get(
            candidate.operation,
            ("Apply the named mutation operation.", "Example: see result hypothesis below"),
        )
        details = f"\n  details: {json.dumps(dict(candidate.details), ensure_ascii=False)}" if candidate.details else ""
        candidate_lines.append(
            "\n".join(
                [
                    f"[{index}] {candidate.operation}",
                    f"  meaning: {meaning}",
                    f"  example: {example}",
                    f"  path: {list(candidate.path)}{details}",
                    f"  result: {render_pretty(candidate.result).replace(chr(10), ' ')}",
                ]
            )
        )

    return {
        "PARENT_HYPOTHESIS_MEASURABLE": render_pretty(parent_hypothesis),
        "PARENT_HYPOTHESIS_NL": parent_hypothesis_nl.strip(),
        "CURRENT_METRICS": json.dumps(
            dict(current_metrics), ensure_ascii=False, indent=2
        ),
        "METRIC_DEFINITIONS": metric_definitions,
        "RECENT_HISTORY": json.dumps(history_payload, ensure_ascii=False, indent=2),
        "TOP_HYPOTHESES": json.dumps(top_payload, ensure_ascii=False, indent=2),
        "MUTATION_CANDIDATES": f"{path_guide}\n\n" + "\n\n".join(candidate_lines),
    }
