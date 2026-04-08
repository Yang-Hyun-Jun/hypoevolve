"""Core immutable data structures for Executable Logic Graphs."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Union


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

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("AtomicNode.name must be a non-empty string")

    @property
    def kind(self) -> str:
        return "atomic"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "name": self.name,
        }


Node = Union["AtomicNode", "LogicalNode", "RelationNode"]


@dataclass(slots=True)
class LogicalNode:
    """Represent a logical composition of one or more child nodes."""

    name: LogicalOp | str
    inputs: List[Node]

    def __post_init__(self) -> None:
        self.name = LogicalOp(self.name)
        if not self.inputs:
            raise ValueError("LogicalNode.inputs must not be empty")
        if self.name is LogicalOp.NOT and len(self.inputs) != 1:
            raise ValueError("LogicalOp.NOT requires exactly one input")
        if self.name in (LogicalOp.AND, LogicalOp.OR) and len(self.inputs) < 2:
            raise ValueError(
                f"LogicalOp.{self.name.value} requires at least two inputs"
            )

    @property
    def kind(self) -> str:
        return "logical"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "name": self.name.value,
            "inputs": [node_to_dict(node) for node in self.inputs],
        }


@dataclass(slots=True)
class RelationNode:
    """Represent a two-sided relation between a condition and a target."""

    name: RelationType | str
    inputs: List[Node]

    def __post_init__(self) -> None:
        self.name = RelationType(self.name)
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
            "name": self.name.value,
            "inputs": [node_to_dict(node) for node in self.inputs],
        }


@dataclass(slots=True)
class Hypothesis:
    """Wrap one ELG root node as a hypothesis object."""

    root: Node

    def to_dict(self) -> Dict[str, Any]:
        return {"root": node_to_dict(self.root)}


def node_to_dict(node: Node) -> Dict[str, Any]:
    """Convert any ELG node to its JSON-serializable dictionary form."""
    return node.to_dict()
