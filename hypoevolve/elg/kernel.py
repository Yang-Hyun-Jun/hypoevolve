"""Tree kernel and distance utilities for ELG hypotheses.

The kernel returns similarity in [0, 1] directly, so no external normalization is
required. Three ELG-specific extensions are folded into the classical Collins-Duffy
recursion:

- **Soft Jaccard for AND/OR children** — extra unmatched children raise the union
  and lower the similarity, giving explicit clutter detection.
- **Wrapper Descent for kind mismatch** — atomic vs AND/OR is not collapsed to 0;
  the atomic is compared to the wrapper's best child with a decay factor.
- **Position-aware relation nodes** — IMPLIES/CONTRADICT/CORRELATE/SUPPORT children
  are matched by position (condition, target), so causal reversal is detected.
"""

from __future__ import annotations

from typing import Sequence

from hypoevolve.elg.ir import (
    AtomicNode,
    Hypothesis,
    LogicalNode,
    LogicalOp,
    Node,
    RelationNode,
)


LAMBDA_WRAP: float = 0.5
"""Decay applied when descending from an atomic through an AND/OR wrapper."""

LAMBDA_NEG: float = 0.3
"""Decay applied when comparing across an unmatched NOT wrapper."""

_ATOMIC_NGRAM_SIZE: int = 3


def atomic_sim(s1: str, s2: str) -> float:
    """Return character n-gram Jaccard similarity for two atomic labels."""
    if s1 == s2:
        return 1.0
    a = _ngrams(s1, _ATOMIC_NGRAM_SIZE)
    b = _ngrams(s2, _ATOMIC_NGRAM_SIZE)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    intersection = len(a & b)
    union = len(a | b)
    return intersection / union if union else 0.0


def tree_kernel(t1: Node | Hypothesis, t2: Node | Hypothesis) -> float:
    """Return the ELG tree-kernel similarity in [0, 1] for two hypotheses/nodes."""
    return _kernel(_root(t1), _root(t2))


def tree_distance(t1: Node | Hypothesis, t2: Node | Hypothesis) -> float:
    """Return the tree-kernel distance ``1 - tree_kernel(t1, t2)`` in [0, 1]."""
    return 1.0 - tree_kernel(t1, t2)


def _root(t: Node | Hypothesis) -> Node:
    if isinstance(t, Hypothesis):
        return t.root
    return t


def _kernel(n1: Node, n2: Node) -> float:
    if n1.kind != n2.kind:
        return _wrapper_descent(n1, n2)
    if isinstance(n1, AtomicNode) and isinstance(n2, AtomicNode):
        return atomic_sim(n1.name, n2.name)
    if isinstance(n1, LogicalNode) and isinstance(n2, LogicalNode):
        if n1.name != n2.name:
            return 0.0
        if n1.name is LogicalOp.NOT:
            return _kernel(n1.inputs[0], n2.inputs[0])
        return _soft_jaccard(n1.inputs, n2.inputs)
    if isinstance(n1, RelationNode) and isinstance(n2, RelationNode):
        if n1.name != n2.name:
            return 0.0
        cond = _kernel(n1.inputs[0], n2.inputs[0])
        target = _kernel(n1.inputs[1], n2.inputs[1])
        return 0.5 * (cond + target)
    return 0.0


def _wrapper_descent(n1: Node, n2: Node) -> float:
    """Handle kind mismatches by descending through common wrappers with decay."""
    # NOT wrapper vs anything: recurse into the NOT's child with a polarity penalty.
    if isinstance(n1, LogicalNode) and n1.name is LogicalOp.NOT:
        return LAMBDA_NEG * _kernel(n1.inputs[0], n2)
    if isinstance(n2, LogicalNode) and n2.name is LogicalOp.NOT:
        return LAMBDA_NEG * _kernel(n1, n2.inputs[0])
    # atomic vs AND/OR wrapper: best atomic-to-child match, scaled by LAMBDA_WRAP.
    if isinstance(n1, AtomicNode) and isinstance(n2, LogicalNode) and n2.name in (
        LogicalOp.AND,
        LogicalOp.OR,
    ):
        best = max((_kernel(n1, child) for child in n2.inputs), default=0.0)
        return LAMBDA_WRAP * best
    if isinstance(n2, AtomicNode) and isinstance(n1, LogicalNode) and n1.name in (
        LogicalOp.AND,
        LogicalOp.OR,
    ):
        best = max((_kernel(child, n2) for child in n1.inputs), default=0.0)
        return LAMBDA_WRAP * best
    return 0.0


def _soft_jaccard(children1: Sequence[Node], children2: Sequence[Node]) -> float:
    """Return the soft Jaccard similarity between two child multisets.

    The soft intersection is the greedy best-match sum of pairwise child kernels;
    the union is ``|children1| + |children2| - intersection`` so that additional
    unmatched children lower the similarity.
    """
    if not children1 and not children2:
        return 1.0
    if not children1 or not children2:
        return 0.0
    pair_scores = [
        (_kernel(c1, c2), i, j)
        for i, c1 in enumerate(children1)
        for j, c2 in enumerate(children2)
    ]
    pair_scores.sort(key=lambda item: -item[0])
    intersection = 0.0
    used_left: set[int] = set()
    used_right: set[int] = set()
    for value, i, j in pair_scores:
        if i in used_left or j in used_right:
            continue
        intersection += value
        used_left.add(i)
        used_right.add(j)
    union = len(children1) + len(children2) - intersection
    return intersection / union if union > 1e-12 else 0.0


def _ngrams(text: str, size: int) -> set[str]:
    lowered = text.lower()
    if len(lowered) < size:
        return {lowered} if lowered else set()
    return {lowered[i : i + size] for i in range(len(lowered) - size + 1)}


__all__ = [
    "LAMBDA_NEG",
    "LAMBDA_WRAP",
    "atomic_sim",
    "tree_distance",
    "tree_kernel",
]
