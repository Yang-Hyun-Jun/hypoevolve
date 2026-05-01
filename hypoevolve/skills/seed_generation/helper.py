from functools import lru_cache
from pathlib import Path

from hypoevolve.data.dataset import load_dataset_schema
from .nodes import nodes
from .tree.base import HypoTree
from .tree.generator import HypoTreeGenerator

DEFAULT_DATASET_SCHEMA_PATH = "dataset.yaml"


@lru_cache(maxsize=None)
def _load_dataset_labels(dataset_schema_path: str) -> tuple[str, ...]:
    schema = load_dataset_schema(dataset_schema_path)
    labels = tuple(column.name for column in schema.columns)
    if not labels:
        raise ValueError(
            "Dataset schema must declare at least one column for DATA nodes"
        )
    return labels


def get_labels(
    dataset_schema_path: str | Path = DEFAULT_DATASET_SCHEMA_PATH,
) -> list[str]:
    resolved_path = str(Path(dataset_schema_path).resolve())
    return list(_load_dataset_labels(resolved_path))


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


def get_tree_generator(
    dataset_schema_path: str | Path = DEFAULT_DATASET_SCHEMA_PATH,
):

    nodes = get_nodes(dataset_schema_path=dataset_schema_path)
    generator = HypoTreeGenerator(nodes)
    return generator


def get_nodes(
    dataset_schema_path: str | Path = DEFAULT_DATASET_SCHEMA_PATH,
) -> list:

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
        for label in get_labels(dataset_schema_path)
    )

    return NODES
