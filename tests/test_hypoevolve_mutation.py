import unittest

from hypoevolve.elg import AtomicNode, Hypothesis, LogicalNode, RelationNode
from hypoevolve.memory.archive import ArchiveEntry
from hypoevolve.core.config import LLMConfig
from hypoevolve.skills.mutation import (
    MutationDecision,
    STEERING_RANDOM_SYSTEM_PROMPT,
    STEERING_SYSTEM_PROMPT,
    steer_mutation,
)


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
                "child_hypothesis": {
                    "kind": "relation",
                    "name": "IMPLIES",
                    "inputs": [
                        {"kind": "logical", "name": "AND", "inputs": [
                            {"kind": "atomic", "name": "A"},
                            {"kind": "atomic", "name": "D"},
                        ]},
                        {"kind": "atomic", "name": "C"},
                    ],
                },
                "domain_reason": "The added condition-side atomic makes the stress regime more coherent from a market-structure perspective.",
                "score_reason": "Adding a related condition-side atomic may improve precision without fully changing the structure.",
                "operation_score_rankings": {"append_atomic": 1, "replace_atomic_feature": 2},
                "mutation_summary": "Applied an append_child-style local mutation in the condition subtree.",
            }
        ])
        decision = steer_mutation(
            parent_hypothesis=self.hypothesis,
            current_metrics={"combined_score": 0.1, "precision": 0.2, "baseline": 0.3, "coverage": 0.05},
            llm=llm,
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
        self.assertEqual(decision.child_hypothesis.root.name.value, "IMPLIES")
        self.assertTrue(decision.domain_reason)
        self.assertTrue(decision.score_reason)
        self.assertEqual(decision.operation_score_rankings["append_atomic"], 1)
        self.assertIn("append_child-style", decision.mutation_summary)

    def test_steer_mutation_retries_on_invalid_payload(self):
        llm = FakeLLM([
            {"child_hypothesis": "bad", "domain_reason": "bad", "score_reason": "", "operation_score_rankings": {}, "mutation_summary": ""},
            {
                "child_hypothesis": {
                    "kind": "relation",
                    "name": "IMPLIES",
                    "inputs": [
                        {"kind": "logical", "name": "AND", "inputs": [
                            {"kind": "atomic", "name": "A"},
                            {"kind": "atomic", "name": "D"},
                        ]},
                        {"kind": "atomic", "name": "C"},
                    ],
                },
                "domain_reason": "The revised child remains plausible as a coherent market hypothesis.",
                "score_reason": "A valid local mutation is better than an invalid payload and is most likely to improve the score.",
                "operation_score_rankings": {"append_atomic": 1, "change_relation_type": 2},
                "mutation_summary": "Applied an append_child-style local mutation in the condition subtree.",
            },
        ], retries=1)
        decision = steer_mutation(
            parent_hypothesis=self.hypothesis,
            current_metrics={"combined_score": 0.1},
            llm=llm,
        )
        self.assertIn("append_child-style", decision.mutation_summary)
        self.assertEqual(len(llm.calls), 2)
        self.assertIn("Previous Attempt Failed", llm.calls[1]["user"])

    def test_random_steer_mutation_allows_missing_reason(self):
        llm = FakeLLM([
            {
                "child_hypothesis": {
                    "kind": "relation",
                    "name": "IMPLIES",
                    "inputs": [
                        {"kind": "logical", "name": "AND", "inputs": [
                            {"kind": "atomic", "name": "A"},
                            {"kind": "atomic", "name": "D"},
                        ]},
                        {"kind": "atomic", "name": "C"},
                    ],
                },
                "mutation_summary": "Applied three exploratory local mutations in the condition subtree.",
            }
        ])
        decision = steer_mutation(
            parent_hypothesis=self.hypothesis,
            current_metrics={"combined_score": 0.1},
            llm=llm,
            use_random_steering=True,
        )
        self.assertEqual(decision.domain_reason, "")
        self.assertEqual(decision.score_reason, "")
        self.assertIn("exploratory", decision.mutation_summary)

    def test_guided_steering_prompt_includes_conclusion_mutation_constraints(self):
        self.assertIn("# Mutation Constraint", STEERING_SYSTEM_PROMPT)
        self.assertIn(
            "`remove_atomic` cannot be applied to the conclusion / target side.",
            STEERING_SYSTEM_PROMPT,
        )
        self.assertIn(
            "Do not use a conclusion / target atomic proposition that is identical to any atomic proposition already used on the condition side.",
            STEERING_SYSTEM_PROMPT,
        )

    def test_random_steering_prompt_includes_conclusion_mutation_constraints(self):
        self.assertIn("# Mutation Constraint", STEERING_RANDOM_SYSTEM_PROMPT)
        self.assertIn(
            "`remove_atomic` cannot be applied to the conclusion / target side.",
            STEERING_RANDOM_SYSTEM_PROMPT,
        )
        self.assertIn(
            "Do not use a conclusion / target atomic proposition that is identical to any atomic proposition already used on the condition side.",
            STEERING_RANDOM_SYSTEM_PROMPT,
        )

    def test_guided_steer_mutation_normalizes_ranking_values_to_ints(self):
        llm = FakeLLM([
            {
                "child_hypothesis": {
                    "kind": "relation",
                    "name": "IMPLIES",
                    "inputs": [
                        {"kind": "logical", "name": "AND", "inputs": [
                            {"kind": "atomic", "name": "A"},
                            {"kind": "atomic", "name": "D"},
                        ]},
                        {"kind": "atomic", "name": "C"},
                    ],
                },
                "domain_reason": "valid",
                "score_reason": "valid",
                "operation_score_rankings": {"append_atomic": "1", "replace_atomic_feature": 2.0},
                "mutation_summary": "Applied an append_child-style local mutation.",
            }
        ])
        decision = steer_mutation(
            parent_hypothesis=self.hypothesis,
            current_metrics={"combined_score": 0.1},
            llm=llm,
        )
        self.assertEqual(decision.operation_score_rankings, {"append_atomic": 1, "replace_atomic_feature": 2})

    def test_guided_steer_mutation_rejects_non_positive_rank_and_reports_failure(self):
        llm = FakeLLM([
            {
                "child_hypothesis": {
                    "kind": "relation",
                    "name": "IMPLIES",
                    "inputs": [
                        {"kind": "atomic", "name": "A"},
                        {"kind": "atomic", "name": "C"},
                    ],
                },
                "domain_reason": "valid",
                "score_reason": "valid",
                "operation_score_rankings": {"append_atomic": 0},
                "mutation_summary": "Applied a mutation.",
            }
        ], retries=0)
        with self.assertRaises(Exception):
            steer_mutation(parent_hypothesis=self.hypothesis, current_metrics={"combined_score": 0.1}, llm=llm, retries=0)

    def test_random_steer_retry_prompt_uses_random_specific_correction_text(self):
        llm = FakeLLM([
            {"child_hypothesis": "bad", "mutation_summary": ""},
            {
                "child_hypothesis": {
                    "kind": "relation",
                    "name": "IMPLIES",
                    "inputs": [
                        {"kind": "atomic", "name": "A"},
                        {"kind": "atomic", "name": "C"},
                    ],
                },
                "mutation_summary": "Applied random mutation.",
            },
        ], retries=1)
        decision = steer_mutation(
            parent_hypothesis=self.hypothesis,
            current_metrics={"combined_score": 0.1},
            llm=llm,
            use_random_steering=True,
            retries=1,
        )
        self.assertIn('child_hypothesis', llm.calls[1]['user'])
        self.assertNotIn('domain_reason', llm.calls[1]['user'])
        self.assertEqual(decision.mutation_summary, 'Applied random mutation.')


if __name__ == '__main__':
    unittest.main()
