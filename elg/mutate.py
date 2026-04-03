"""Immutable tree-edit helpers for local ELG mutations."""

from __future__ import annotations

from typing import List, Tuple

from .ir import Hypothesis, LogicalNode, LogicalOp, Node, RelationNode, RelationType

Path = Tuple[int, ...]


def get_node_at_path(hypothesis: Hypothesis, path: Path) -> Node:
    """Return the node found at a positional path within a hypothesis."""
    node = hypothesis.root
    for index in path:
        children = _children_of(node)
        if index < 0 or index >= len(children):
            raise IndexError(f"path {path} is out of range")
        node = children[index]
    return node


def iter_paths(hypothesis: Hypothesis) -> List[Path]:
    """Enumerate every valid node path in a hypothesis tree."""
    paths: List[Path] = []
    _collect_paths(hypothesis.root, (), paths)
    return paths


def replace_at_path(hypothesis: Hypothesis, path: Path, new_node: Node) -> Hypothesis:
    """Return a new hypothesis with one subtree replaced at ``path``."""
    if not path:
        return Hypothesis(root=new_node, params=dict(hypothesis.params))
    return Hypothesis(
        root=_replace_in_node(hypothesis.root, path, new_node),
        params=dict(hypothesis.params),
    )


def mutate_replace_subtree(hypothesis: Hypothesis, path: Path, new_node: Node) -> Hypothesis:
    """Replace the subtree at ``path`` with ``new_node``."""
    return replace_at_path(hypothesis, path, new_node)


def mutate_replace_child(
    hypothesis: Hypothesis, path: Path, child_index: int, new_child: Node
) -> Hypothesis:
    """Replace one direct child beneath the node located at ``path``."""
    parent = get_node_at_path(hypothesis, path)
    if isinstance(parent, LogicalNode):
        updated_inputs = list(parent.inputs)
        if child_index < 0 or child_index >= len(updated_inputs):
            raise IndexError("child_index is out of range")
        updated_inputs[child_index] = new_child
        return replace_at_path(
            hypothesis, path, LogicalNode(parent.op, updated_inputs, params=dict(parent.params))
        )
    if isinstance(parent, RelationNode):
        updated_inputs = list(parent.inputs)
        if child_index < 0 or child_index >= len(updated_inputs):
            raise IndexError("child_index is out of range")
        updated_inputs[child_index] = new_child
        return replace_at_path(
            hypothesis, path, RelationNode(parent.type, updated_inputs, params=dict(parent.params))
        )
    raise TypeError("Only logical or relation nodes can replace children")


def mutate_logical_operator(
    hypothesis: Hypothesis, path: Path, new_op: LogicalOp | str
) -> Hypothesis:
    """Change the logical operator at ``path`` while preserving inputs."""
    node = get_node_at_path(hypothesis, path)
    if not isinstance(node, LogicalNode):
        raise TypeError("Target node is not a LogicalNode")
    return replace_at_path(
        hypothesis,
        path,
        LogicalNode(new_op, list(node.inputs), params=dict(node.params)),
    )


def mutate_relation_type(
    hypothesis: Hypothesis, path: Path, new_type: RelationType | str
) -> Hypothesis:
    """Change the relation type at ``path`` while preserving both sides."""
    node = get_node_at_path(hypothesis, path)
    if not isinstance(node, RelationNode):
        raise TypeError("Target node is not a RelationNode")
    return replace_at_path(
        hypothesis,
        path,
        RelationNode(new_type, list(node.inputs), params=dict(node.params)),
    )


def mutate_wrap_not(hypothesis: Hypothesis, path: Path) -> Hypothesis:
    """Wrap the node at ``path`` in a ``NOT`` logical node."""
    node = get_node_at_path(hypothesis, path)
    return replace_at_path(
        hypothesis,
        path,
        LogicalNode(LogicalOp.NOT, [node]),
    )


def mutate_unwrap_not(hypothesis: Hypothesis, path: Path) -> Hypothesis:
    """Remove a ``NOT`` node at ``path`` and return its only child."""
    node = get_node_at_path(hypothesis, path)
    if not isinstance(node, LogicalNode) or node.op is not LogicalOp.NOT:
        raise TypeError("Target node is not a NOT logical node")
    return replace_at_path(hypothesis, path, node.inputs[0])


def mutate_append_child(hypothesis: Hypothesis, path: Path, new_child: Node) -> Hypothesis:
    """Append a child to an ``AND`` or ``OR`` node at ``path``."""
    node = get_node_at_path(hypothesis, path)
    if not isinstance(node, LogicalNode) or node.op not in (LogicalOp.AND, LogicalOp.OR):
        raise TypeError("Children can only be appended to AND/OR logical nodes")
    return replace_at_path(
        hypothesis,
        path,
        LogicalNode(node.op, list(node.inputs) + [new_child], params=dict(node.params)),
    )


def mutate_remove_child(hypothesis: Hypothesis, path: Path, child_index: int) -> Hypothesis:
    """Remove one child from an ``AND`` or ``OR`` node at ``path``."""
    node = get_node_at_path(hypothesis, path)
    if not isinstance(node, LogicalNode) or node.op not in (LogicalOp.AND, LogicalOp.OR):
        raise TypeError("Children can only be removed from AND/OR logical nodes")
    if child_index < 0 or child_index >= len(node.inputs):
        raise IndexError("child_index is out of range")
    remaining_inputs = [child for i, child in enumerate(node.inputs) if i != child_index]
    return replace_at_path(
        hypothesis,
        path,
        LogicalNode(node.op, remaining_inputs, params=dict(node.params)),
    )


def _collect_paths(node: Node, path: Path, paths: List[Path]) -> None:
    paths.append(path)
    for index, child in enumerate(_children_of(node)):
        _collect_paths(child, path + (index,), paths)


def _replace_in_node(node: Node, path: Path, new_node: Node) -> Node:
    children = _children_of(node)
    index = path[0]
    if index < 0 or index >= len(children):
        raise IndexError(f"path {path} is out of range")

    updated_children = list(children)
    if len(path) == 1:
        updated_children[index] = new_node
    else:
        updated_children[index] = _replace_in_node(updated_children[index], path[1:], new_node)

    return _clone_with_children(node, updated_children)


def _children_of(node: Node) -> List[Node]:
    if isinstance(node, LogicalNode):
        return list(node.inputs)
    if isinstance(node, RelationNode):
        return list(node.inputs)
    return []


def _clone_with_children(node: Node, children: List[Node]) -> Node:
    if isinstance(node, LogicalNode):
        return LogicalNode(node.op, children, params=dict(node.params))
    if isinstance(node, RelationNode):
        return RelationNode(node.type, children, params=dict(node.params))
    raise TypeError("Atomic nodes do not have children")
