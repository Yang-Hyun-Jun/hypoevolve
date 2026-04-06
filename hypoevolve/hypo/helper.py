from hypoevolve.hypo.nodes import nodes
from hypoevolve.hypo.tree.base import HypoTree
from hypoevolve.hypo.tree.generator import HypoTreeGenerator

LABELS = [
    "OPEN",
    "HIGH",
    "LOW",
    "CLOSE",
    "VOLUME",
    "PREMIUM_INDEX_CLOSE",
    "PREMIUM_INDEX_OPEN",
    "PREMIUM_INDEX_HIGH",
    "PREMIUM_INDEX_LOW",
    "TAKER_BUY_VOLUME",
    "TAKER_SELL_VOLUME",
    "FUNDING_SCORE",
    "ORDER_FLOW_IMBALANCE",
]


def generate_trees(
    generator: HypoTreeGenerator,
    max_depth: int = None,
    num_trees: int = None,
) -> list[HypoTree]:

    trees = []
    attempts = 0
    max_attempts = num_trees * 20

    while len(trees) < num_trees:
        attempts += 1
        if attempts > max_attempts:
            raise (
                "Failed to generate enough completed trees; "
                f"generated {len(trees)}/{num_trees} after {max_attempts} attempts"
            )

        tree = generator.generate(max_depth=max_depth)

        if tree.iscompleted and tree.depth == max_depth:
            trees.append(tree)

    return trees


def get_tree_generator():

    nodes = get_nodes()
    generator = HypoTreeGenerator(nodes)
    return generator


def get_nodes() -> list:

    P = "PERIOD"
    DATA_KWARGS = {}

    NODES = [
        # Basic math operation nodes
        nodes.ADD(),
        nodes.DIV(),
        nodes.SUB(),
        nodes.ABS(),
        nodes.SMA(period=P),
        nodes.SHIFT(period=P),
        nodes.DIFF(period=P),
        nodes.PctChange(period=P),
        nodes.STD(period=P),
        nodes.NewHigh(period=P),
        nodes.NewLow(period=P),
        nodes.MAX(period=P),
        nodes.MIN(period=P),
        nodes.ZSCORE(period=P),
        nodes.SKEW(period=P),
        nodes.KURT(period=P),
        nodes.ZEXP(period=P),
        nodes.ZSigmoid(period=P),
        # Root nodes
        nodes.CrossUp(),
        nodes.CrossDown(),
        nodes.Comparison(),
        nodes.ZBetween(period=P, lo=-1.5, hi=1.5),
        nodes.EqualApprox(tol=1e-2),
        nodes.UpStreak(period=P),
        nodes.DownStreak(period=P),
        nodes.MeanRevertKick(period=P, z_th=1.0, dmax=P, eps=0.01),
        nodes.PullbackWithinBand(period=P, k=0.5),
        nodes.DrawdownExceed(pct=0.1, lookback=P),
        nodes.JumpDetect(period=P, q_tail=0.1),
    ]

    NODES.extend(
        nodes.DATA(
            label=label,
            **DATA_KWARGS,
        )
        for label in LABELS
    )

    return NODES
