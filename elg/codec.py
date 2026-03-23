from __future__ import annotations

import json
from typing import Any, Dict, Optional

from .ir import (
    AtomicNode,
    Hypothesis,
    LogicalNode,
    Node,
    RelationNode,
)


def node_from_dict(data: Dict[str, Any]) -> Node:
    kind = data.get("kind")
    if kind == "atomic":
        return AtomicNode(
            name=data["name"],
            type=data.get("type", "abstract"),
            source=data.get("source", "semantic"),
            params=dict(data.get("params", {})),
        )
    if kind == "logical":
        return LogicalNode(
            op=data["op"],
            inputs=[node_from_dict(child) for child in data.get("inputs", [])],
            params=dict(data.get("params", {})),
        )
    if kind == "relation":
        return RelationNode(
            type=data["type"],
            inputs=[node_from_dict(child) for child in data.get("inputs", [])],
            params=dict(data.get("params", {})),
        )
    raise ValueError(f"Unsupported node kind: {kind!r}")


def hypothesis_from_dict(data: Dict[str, Any]) -> Hypothesis:
    if "root" not in data:
        raise ValueError("Hypothesis payload must include 'root'")
    return Hypothesis(
        root=node_from_dict(data["root"]),
        params=dict(data.get("params", {})),
    )


def hypothesis_to_json(hypothesis: Hypothesis, *, indent: Optional[int] = 2) -> str:
    return json.dumps(hypothesis.to_dict(), ensure_ascii=False, indent=indent, sort_keys=True)


def hypothesis_from_json(payload: str) -> Hypothesis:
    return hypothesis_from_dict(json.loads(payload))
