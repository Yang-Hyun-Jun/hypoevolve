import hashlib
import unittest

from hypoevolve.archive import ArchiveEntry
from hypoevolve.helper import build_steering_prompt_variables
from hypoevolve.prompts import load_and_render_prompt, load_prompt
from elg import AtomicNode, Hypothesis, LogicalNode, RelationNode


class TestHypoEvolvePromptSnapshots(unittest.TestCase):
    def test_raw_prompt_files_match_snapshots(self):
        expected_hashes = {
            "parser/system.md": "481612b0115d3fc1e53bfa4a00e4183390ccea1b6060a991651ae98a9b74d4ce",
            "measurable/system.md": "1506f70d8ec791e12f8225b06df58fdfdec17de065e9eae62c0a5b1f89b3919b",
            "nl/system.md": "04f4baba75010b536d3bce8ac47a83d69011d43847b6ca8a2dba5e571e81e033",
            "steering/system.md": "f27b57c16d5dc39c228ed1c17e5b6f147950eef02752b555647ae2390781fd5c",
            "steering-random/system.md": "21172a30b67e1cff8500b72ba89c060fbbbd8a74a3cc5ce83e87bf0cfc48d5f1",
            "evaluator/system.md": "77ff769b47adc813475bd5618dcda08ab3719bb3a9d099454ac944c224090d50",
            "evaluator/user.md": "6e59c4cda8aa686c1f6ae2c4aad9992a00b49553f202c83a07ca6ad947271e70",
        }

        for relative_path, expected_hash in expected_hashes.items():
            parts = relative_path.split("/")
            prompt = load_prompt(*parts)
            actual_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            self.assertEqual(
                actual_hash,
                expected_hash,
                msg=f"Prompt snapshot drifted: {relative_path}",
            )

    def test_rendered_evaluator_prompt_matches_snapshot(self):
        rendered = load_and_render_prompt(
            "evaluator",
            "user.md",
            variables={
                "HYPOTHESIS_PRETTY": "If A then B",
                "DATASET_DESCRIPTION": "demo dataset",
                "INDEX_NAME": "close_time",
                "INDEX_DTYPE": "datetime64[us]",
                "ENTITIES": "BTCUSDT, ETHUSDT",
                "COLUMN_SPECS": "- close: close price",
                "DATASET_ACCESSOR_DOC": "accessor.load_dataframe(entity)",
            },
        )
        self.assertEqual(
            hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
            "d9c4b61392c1bfe2c3813345ca5ca93e391c15d38bbc6fba1043d011eb27aa45",
        )

    def test_rendered_steering_prompt_matches_snapshot(self):
        hypothesis = Hypothesis(
            root=RelationNode(
                "IMPLIES",
                [
                    LogicalNode("AND", [AtomicNode("A"), AtomicNode("B")]),
                    AtomicNode("C"),
                ],
            )
        )
        variables = build_steering_prompt_variables(
            parent_hypothesis=hypothesis,
            current_metrics={"combined_score": 0.1, "precision": 0.2},
            recent_history=[
                {
                    "mutation_summary": "Applied a wrap_not-style local mutation.",
                    "score_delta": -0.1,
                }
            ],
            top_hypotheses=[
                ArchiveEntry(
                    hypothesis=Hypothesis(root=AtomicNode("BEST")),
                    metrics={"combined_score": 0.9},
                    fingerprint="best-fp",
                    iteration=3,
                    metadata={"source": "test"},
                )
            ],
        )
        rendered = load_and_render_prompt("steering", "user.md", variables=variables)
        self.assertEqual(
            hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
            "f3de668f74a9f69435451228cc8a59d13df32a104b2aae7abaf0848b928935d3",
        )


if __name__ == "__main__":
    unittest.main()
