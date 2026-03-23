from __future__ import annotations

import random
from typing import Dict, Protocol

from elg import Hypothesis, count_nodes, fingerprint, tree_depth


class Evaluator(Protocol):
    def evaluate(self, hypothesis: Hypothesis) -> Dict[str, float]:
        ...


class PlaceholderEvaluator:
    def __init__(self, seed: int = 42):
        self.seed = seed

    def evaluate(self, hypothesis: Hypothesis) -> Dict[str, float]:
        key = fingerprint(hypothesis)
        rng = random.Random(f"{self.seed}:{key}")
        base_score = rng.random()
        complexity_penalty = min(0.25, count_nodes(hypothesis) * 0.01)
        combined = max(0.0, min(1.0, base_score - complexity_penalty))
        return {
            "combined_score": combined,
            "raw_score": base_score,
            "complexity_penalty": complexity_penalty,
            "node_count": float(count_nodes(hypothesis)),
            "depth": float(tree_depth(hypothesis)),
        }


def evaluate_hypothesis(hypothesis: Hypothesis, evaluator: Evaluator) -> Dict[str, float]:
    return evaluator.evaluate(hypothesis)
