"""Human-readable rendering helpers for ELG nodes and hypotheses."""

from __future__ import annotations

from typing import List, Union

from .ir import AtomicNode, Hypothesis, LogicalNode, RelationNode

Renderable = Union[AtomicNode, LogicalNode, RelationNode, Hypothesis]


def label_of(node: Renderable) -> str:
    """Return the display label for one renderable ELG object."""
    if isinstance(node, Hypothesis):
        return "Hypothesis"
    if isinstance(node, AtomicNode):
        return node.name
    if isinstance(node, LogicalNode):
        return node.name.value
    if isinstance(node, RelationNode):
        return node.name.value
    raise TypeError(f"Unsupported node type: {type(node)!r}")


def children_of(node: Renderable) -> List[Renderable]:
    """Return the renderable children of one ELG object."""
    if isinstance(node, Hypothesis):
        return [node.root]
    if isinstance(node, AtomicNode):
        return []
    if isinstance(node, (LogicalNode, RelationNode)):
        return list(node.inputs)
    raise TypeError(f"Unsupported node type: {type(node)!r}")


def render_pretty(node: Renderable, indent: int = 0) -> str:
    """Render an ELG object as nested function-style text."""
    if isinstance(node, Hypothesis):
        return render_pretty(node.root, indent=indent)
    if isinstance(node, AtomicNode):
        return node.name

    pad = " " * indent
    child_indent = indent + 2
    rendered_children = [render_pretty(child, indent=child_indent) for child in children_of(node)]
    joined = ",\n".join((" " * child_indent) + child for child in rendered_children)
    return f"{label_of(node)}(\n{joined}\n{pad})"


def render_tree(node: Renderable) -> str:
    """Render an ELG object as an ASCII tree."""
    lines = [label_of(node)]
    for index, child in enumerate(children_of(node)):
        is_last = index == len(children_of(node)) - 1
        lines.extend(_render_tree_lines(child, prefix="", is_last=is_last))
    return "\n".join(lines)


def _render_tree_lines(node: Renderable, prefix: str, is_last: bool) -> List[str]:
    connector = "└── " if is_last else "├── "
    lines = [f"{prefix}{connector}{label_of(node)}"]
    child_prefix = prefix + ("    " if is_last else "│   ")
    node_children = children_of(node)
    for index, child in enumerate(node_children):
        child_is_last = index == len(node_children) - 1
        lines.extend(_render_tree_lines(child, child_prefix, child_is_last))
    return lines
