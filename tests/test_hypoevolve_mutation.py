import unittest

from elg import AtomicNode, Hypothesis, LogicalNode, RelationNode
from hypoevolve.archive import ArchiveEntry
from hypoevolve.config import LLMConfig
from hypoevolve.mutation import MutationDecision, steer_mutation


class FakeLLM:
    def __init__(self, payloads, retries: int = 1):
        self.payloads = list(payloads)
        self.config = LLMConfig(retries=retries)
        self.calls = []

    def generate_json(self, system: str, user: str, **kwargs):
        self.calls.append({"system": system, "user": user, "kwargs": kwargs})
        if not self.payloads:
            raise RuntimeError("No fake payloads remaining")
        payload = self.payloads.pop(0)
        if isinstance(payload, Exception):
            raise payload
        return payload


class TestHypoEvolveMutation(unittest.TestCase):
    def setUp(self):
        self.hypothesis = Hypothesis(
            root=RelationNode(
                "IMPLIES",
                [
                    LogicalNode("AND", [AtomicNode("A"), AtomicNode("B")]),
                    AtomicNode("C"),
                ],
            )
        )

    def test_steer_mutation_returns_selected_candidate(self):
        llm = FakeLLM([
            {
                "selected_candidate_index": 0,
                "reason": "The first legal candidate is the most direct local change for improving coverage without increasing structural complexity.",
            }
        ])
        decision = steer_mutation(
            parent_hypothesis=self.hypothesis,
            parent_hypothesis_nl="If A and B then C.",
            current_metrics={"combined_score": 0.1, "precision": 0.2, "baseline": 0.3, "coverage": 0.05},
            llm=llm,
            atomic_pool=[AtomicNode("D")],
            recent_history=[{"operation": "wrap_not", "score_delta": -0.1}],
            top_hypotheses=[
                ArchiveEntry(
                    hypothesis=Hypothesis(root=AtomicNode("BEST")),
                    metrics={"combined_score": 0.9},
                    fingerprint="best-fp",
                )
            ],
        )
        self.assertIsInstance(decision, MutationDecision)
        self.assertEqual(decision.selected_candidate_index, 0)
        self.assertTrue(decision.reason)
        self.assertEqual(decision.mutation, decision.mutation)

    def test_steer_mutation_retries_on_invalid_index(self):
        llm = FakeLLM([
            {"selected_candidate_index": 999, "reason": "bad"},
            {
                "selected_candidate_index": 0,
                "reason": "A valid local mutation is better than an invalid selection and is most likely to improve the score.",
            },
        ], retries=1)
        decision = steer_mutation(
            parent_hypothesis=self.hypothesis,
            parent_hypothesis_nl="If A and B then C.",
            current_metrics={"combined_score": 0.1},
            llm=llm,
            atomic_pool=[AtomicNode("D")],
        )
        self.assertEqual(decision.selected_candidate_index, 0)
        self.assertEqual(len(llm.calls), 2)
        self.assertIn("Previous Attempt Failed", llm.calls[1]["user"])


if __name__ == '__main__':
    unittest.main()
