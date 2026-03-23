from elg import (
    AtomicNode,
    Hypothesis,
    LogicalNode,
    RelationNode,
    render_pretty,
    render_tree,
)

hypothesis = Hypothesis(
    root=RelationNode(
        "IMPLIES",
        [
            LogicalNode(
                "AND",
                [
                    AtomicNode("FUNDING_FEE > 0", type="boolean", source="primitive"),
                    AtomicNode("CLOSE > SMA_20", type="boolean", source="primitive"),
                ],
            ),
            AtomicNode("RETURN_5M > 0", type="boolean", source="primitive"),
        ],
    )
)

print(render_pretty(hypothesis))
print()
print(render_tree(hypothesis))
