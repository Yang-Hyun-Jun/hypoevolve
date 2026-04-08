"""Normalization helpers for canonicalizing equivalent ELG structures."""

from __future__ import annotations

from typing import List

from .codec import hypothesis_to_json
from .ir import AtomicNode, Hypothesis, LogicalNode, LogicalOp, Node, RelationNode


def normalize_hypothesis(hypothesis: Hypothesis) -> Hypothesis:
    """Normalize a hypothesis into a canonical structural form."""
    return Hypothesis(root=normalize_node(hypothesis.root))


def normalize_node(node: Node) -> Node:
    """Normalize one ELG node recursively."""
    if isinstance(node, AtomicNode):
        return AtomicNode(name=node.name)

    if isinstance(node, RelationNode):
        return RelationNode(name=node.name, inputs=[normalize_node(child) for child in node.inputs])

    normalized_children = [normalize_node(child) for child in node.inputs]

    if node.name is LogicalOp.NOT:
        child = normalized_children[0]
        if isinstance(child, LogicalNode) and child.name is LogicalOp.NOT:
            return child.inputs[0]
        return LogicalNode(node.name, [child])

    flattened: List[Node] = []
    for child in normalized_children:
        if isinstance(child, LogicalNode) and child.name is node.name:
            flattened.extend(child.inputs)
        else:
            flattened.append(child)

    deduped = _dedupe_and_sort(flattened)
    if len(deduped) == 1:
        return deduped[0]

    return LogicalNode(node.name, deduped)


def _dedupe_and_sort(nodes: List[Node]) -> List[Node]:
    deduped = {}
    for node in nodes:
        deduped.setdefault(_node_sort_key(node), node)
    return [deduped[key] for key in sorted(deduped)]


def _node_sort_key(node: Node) -> str:
    return hypothesis_to_json(Hypothesis(root=node), indent=None)
