"""Core immutable data structures for Executable Logic Graphs."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Union


class AtomicType(str, Enum):
    """Supported semantic types for atomic propositions."""

    BOOLEAN = "boolean"
    NUMERIC = "numeric"
    ABSTRACT = "abstract"


class AtomicSource(str, Enum):
    """Origin categories for atomic propositions."""

    PRIMITIVE = "primitive"
    SEMANTIC = "semantic"


class LogicalOp(str, Enum):
    """Logical operators supported by ELG logical nodes."""

    AND = "AND"
    OR = "OR"
    NOT = "NOT"


class RelationType(str, Enum):
    """Relation operators supported by ELG relation nodes."""

    IMPLIES = "IMPLIES"
    SUPPORT = "SUPPORT"
    CONTRADICT = "CONTRADICT"
    CORRELATE = "CORRELATE"


@dataclass(slots=True)
class AtomicNode:
    """Represent one measurable or semantic atomic proposition."""

    name: str
    type: AtomicType | str = AtomicType.ABSTRACT
    source: AtomicSource | str = AtomicSource.SEMANTIC
    params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("AtomicNode.name must be a non-empty string")
        self.type = AtomicType(self.type)
        self.source = AtomicSource(self.source)

    @property
    def kind(self) -> str:
        return "atomic"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "name": self.name,
            "type": self.type.value,
            "source": self.source.value,
            "params": dict(self.params),
        }


Node = Union["AtomicNode", "LogicalNode", "RelationNode"]


@dataclass(slots=True)
class LogicalNode:
    """Represent a logical composition of one or more child nodes."""

    op: LogicalOp | str
    inputs: List[Node]
    params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.op = LogicalOp(self.op)
        if not self.inputs:
            raise ValueError("LogicalNode.inputs must not be empty")
        if self.op is LogicalOp.NOT and len(self.inputs) != 1:
            raise ValueError("LogicalOp.NOT requires exactly one input")
        if self.op in (LogicalOp.AND, LogicalOp.OR) and len(self.inputs) < 2:
            raise ValueError(f"LogicalOp.{self.op.value} requires at least two inputs")

    @property
    def kind(self) -> str:
        return "logical"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "op": self.op.value,
            "inputs": [node_to_dict(node) for node in self.inputs],
            "params": dict(self.params),
        }


@dataclass(slots=True)
class RelationNode:
    """Represent a two-sided relation between a condition and a target."""

    type: RelationType | str
    inputs: List[Node]
    params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.type = RelationType(self.type)
        if len(self.inputs) != 2:
            raise ValueError("RelationNode.inputs must contain exactly two nodes")

    @property
    def kind(self) -> str:
        return "relation"

    @property
    def condition(self) -> Node:
        return self.inputs[0]

    @property
    def target(self) -> Node:
        return self.inputs[1]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "type": self.type.value,
            "inputs": [node_to_dict(node) for node in self.inputs],
            "params": dict(self.params),
        }


@dataclass(slots=True)
class Hypothesis:
    """Wrap one ELG root node as a hypothesis object."""

    root: Node
    params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"root": node_to_dict(self.root), "params": dict(self.params)}


def node_to_dict(node: Node) -> Dict[str, Any]:
    """Convert any ELG node to its JSON-serializable dictionary form."""
    return node.to_dict()
