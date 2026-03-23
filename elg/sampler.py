from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

from .ir import AtomicNode, Hypothesis, LogicalNode, LogicalOp, RelationNode, RelationType
from .mutate import (
    Path,
    get_node_at_path,
    iter_paths,
    mutate_append_child,
    mutate_logical_operator,
    mutate_relation_type,
    mutate_remove_child,
    mutate_replace_subtree,
    mutate_unwrap_not,
    mutate_wrap_not,
)


@dataclass(slots=True)
class MutationSample:
    operation: str
    path: Path
    result: Hypothesis
    details: Dict[str, Any] = field(default_factory=dict)


def generate_mutation_candidates(
    hypothesis: Hypothesis,
    atomic_pool: Sequence[AtomicNode | str] | None = None,
) -> List[MutationSample]:
    candidates: List[MutationSample] = []
    normalized_pool = _normalize_atomic_pool(atomic_pool)

    for path in iter_paths(hypothesis):
        node = get_node_at_path(hypothesis, path)

        candidates.append(
            MutationSample(
                operation="wrap_not",
                path=path,
                result=mutate_wrap_not(hypothesis, path),
            )
        )

        if isinstance(node, AtomicNode):
            candidates.extend(_atomic_replacement_candidates(hypothesis, path, node, normalized_pool))
            continue

        if isinstance(node, LogicalNode):
            candidates.extend(_logical_mutation_candidates(hypothesis, path, node, normalized_pool))
            continue

        if isinstance(node, RelationNode):
            candidates.extend(_relation_mutation_candidates(hypothesis, path, node))

    return candidates


def sample_mutation(
    hypothesis: Hypothesis,
    rng: random.Random | None = None,
    atomic_pool: Sequence[AtomicNode | str] | None = None,
) -> MutationSample:
    candidates = generate_mutation_candidates(hypothesis, atomic_pool=atomic_pool)
    if not candidates:
        raise ValueError("No legal mutation candidates available")
    chooser = rng or random.Random()
    return chooser.choice(candidates)


def _atomic_replacement_candidates(
    hypothesis: Hypothesis,
    path: Path,
    node: AtomicNode,
    atomic_pool: Sequence[AtomicNode],
) -> List[MutationSample]:
    candidates: List[MutationSample] = []
    for replacement in atomic_pool:
        if _same_atomic(node, replacement):
            continue
        candidates.append(
            MutationSample(
                operation="replace_atomic",
                path=path,
                result=mutate_replace_subtree(hypothesis, path, replacement),
                details={"replacement": replacement.to_dict()},
            )
        )
    return candidates


def _logical_mutation_candidates(
    hypothesis: Hypothesis,
    path: Path,
    node: LogicalNode,
    atomic_pool: Sequence[AtomicNode],
) -> List[MutationSample]:
    candidates: List[MutationSample] = []

    for op in LogicalOp:
        if op == node.op:
            continue
        try:
            result = mutate_logical_operator(hypothesis, path, op)
        except ValueError:
            continue
        candidates.append(
            MutationSample(
                operation="change_logical_operator",
                path=path,
                result=result,
                details={"new_op": op.value},
            )
        )

    if node.op is LogicalOp.NOT:
        candidates.append(
            MutationSample(
                operation="unwrap_not",
                path=path,
                result=mutate_unwrap_not(hypothesis, path),
            )
        )

    if node.op in (LogicalOp.AND, LogicalOp.OR):
        for candidate in atomic_pool:
            if any(_same_atomic(child, candidate) for child in node.inputs if isinstance(child, AtomicNode)):
                continue
            candidates.append(
                MutationSample(
                    operation="append_child",
                    path=path,
                    result=mutate_append_child(hypothesis, path, candidate),
                    details={"child": candidate.to_dict()},
                )
            )

        if len(node.inputs) > 2:
            for child_index in range(len(node.inputs)):
                candidates.append(
                    MutationSample(
                        operation="remove_child",
                        path=path,
                        result=mutate_remove_child(hypothesis, path, child_index),
                        details={"child_index": child_index},
                    )
                )

    return candidates


def _relation_mutation_candidates(
    hypothesis: Hypothesis,
    path: Path,
    node: RelationNode,
) -> List[MutationSample]:
    candidates: List[MutationSample] = []
    for relation_type in RelationType:
        if relation_type == node.type:
            continue
        candidates.append(
            MutationSample(
                operation="change_relation_type",
                path=path,
                result=mutate_relation_type(hypothesis, path, relation_type),
                details={"new_type": relation_type.value},
            )
        )
    return candidates


def _normalize_atomic_pool(atomic_pool: Sequence[AtomicNode | str] | None) -> List[AtomicNode]:
    if not atomic_pool:
        return []

    normalized: List[AtomicNode] = []
    for candidate in atomic_pool:
        if isinstance(candidate, AtomicNode):
            normalized.append(
                AtomicNode(
                    name=candidate.name,
                    type=candidate.type,
                    source=candidate.source,
                    params=dict(candidate.params),
                )
            )
        else:
            normalized.append(AtomicNode(str(candidate)))
    return normalized


def _same_atomic(left: AtomicNode, right: AtomicNode) -> bool:
    return (
        left.name == right.name
        and left.type == right.type
        and left.source == right.source
        and left.params == right.params
    )
