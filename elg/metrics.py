"""Structural metrics and stable fingerprints for ELG hypotheses."""

from __future__ import annotations

import hashlib

from .codec import hypothesis_to_json
from .ir import AtomicNode, Hypothesis, LogicalNode, Node, RelationNode
from .normalize import normalize_hypothesis


def count_nodes(hypothesis: Hypothesis) -> int:
    """Count every node in a hypothesis tree."""
    return _count_nodes(hypothesis.root)


def tree_depth(hypothesis: Hypothesis) -> int:
    """Return the maximum root-to-leaf depth of a hypothesis."""
    return _tree_depth(hypothesis.root)


def count_atomics(hypothesis: Hypothesis) -> int:
    """Count atomic nodes in a hypothesis."""
    return _count_nodes_by_type(hypothesis.root, AtomicNode)


def count_logicals(hypothesis: Hypothesis) -> int:
    """Count logical nodes in a hypothesis."""
    return _count_nodes_by_type(hypothesis.root, LogicalNode)


def count_relations(hypothesis: Hypothesis) -> int:
    """Count relation nodes in a hypothesis."""
    return _count_nodes_by_type(hypothesis.root, RelationNode)


def fingerprint(hypothesis: Hypothesis) -> str:
    """Return a stable hash for a normalized hypothesis structure."""
    normalized = normalize_hypothesis(hypothesis)
    payload = hypothesis_to_json(normalized, indent=None)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _count_nodes(node: Node) -> int:
    if isinstance(node, AtomicNode):
        return 1
    return 1 + sum(_count_nodes(child) for child in node.inputs)


def _tree_depth(node: Node) -> int:
    if isinstance(node, AtomicNode):
        return 1
    return 1 + max(_tree_depth(child) for child in node.inputs)


def _count_nodes_by_type(node: Node, expected_type: type) -> int:
    count = int(isinstance(node, expected_type))
    if isinstance(node, AtomicNode):
        return count
    return count + sum(_count_nodes_by_type(child, expected_type) for child in node.inputs)
