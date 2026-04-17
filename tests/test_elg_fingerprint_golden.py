import unittest

from elg import AtomicNode, Hypothesis, LogicalNode, RelationNode, fingerprint, hypothesis_to_json, normalize_hypothesis


class TestELGFingerprintGolden(unittest.TestCase):
    def test_fingerprint_matches_golden_normalized_payloads(self):
        cases = {
            "atomic_A": (
                Hypothesis(root=AtomicNode("A")),
                '{"root": {"kind": "atomic", "name": "A"}}',
                "d8240f708d9f80c2a4394f80bf7ec9f27d2ed7d3a6dfd4a272d3e6ff6c6926f7",
            ),
            "and_unsorted_duped": (
                Hypothesis(root=LogicalNode("AND", [AtomicNode("B"), AtomicNode("A"), AtomicNode("B")])),
                '{"root": {"inputs": [{"kind": "atomic", "name": "A"}, {"kind": "atomic", "name": "B"}], "kind": "logical", "name": "AND"}}',
                "a6f00313d893ca9b8443c20bd173c8a1584a5df3c006697a5f74e583cd855ddd",
            ),
            "and_nested": (
                Hypothesis(root=LogicalNode("AND", [AtomicNode("C"), LogicalNode("AND", [AtomicNode("B"), AtomicNode("A")])])),
                '{"root": {"inputs": [{"kind": "atomic", "name": "A"}, {"kind": "atomic", "name": "B"}, {"kind": "atomic", "name": "C"}], "kind": "logical", "name": "AND"}}',
                "7dd12a1b3458308d7a8f85d20dba5d8969b0906426d9a1d4fcf02c0a97e518b2",
            ),
            "double_not": (
                Hypothesis(root=LogicalNode("NOT", [LogicalNode("NOT", [AtomicNode("A")])])),
                '{"root": {"kind": "atomic", "name": "A"}}',
                "d8240f708d9f80c2a4394f80bf7ec9f27d2ed7d3a6dfd4a272d3e6ff6c6926f7",
            ),
            "relation_and": (
                Hypothesis(root=RelationNode("IMPLIES", [LogicalNode("AND", [AtomicNode("B"), AtomicNode("A")]), AtomicNode("C")])),
                '{"root": {"inputs": [{"inputs": [{"kind": "atomic", "name": "A"}, {"kind": "atomic", "name": "B"}], "kind": "logical", "name": "AND"}, {"kind": "atomic", "name": "C"}], "kind": "relation", "name": "IMPLIES"}}',
                "d3ffd0fcd85eb1b196fb874682a2617ebc2c25df735b7cd2b2bf35924a963e9f",
            ),
        }

        for name, (hypothesis, expected_payload, expected_fingerprint) in cases.items():
            with self.subTest(name=name):
                normalized_payload = hypothesis_to_json(normalize_hypothesis(hypothesis), indent=None)
                self.assertEqual(normalized_payload, expected_payload)
                self.assertEqual(fingerprint(hypothesis), expected_fingerprint)


if __name__ == "__main__":
    unittest.main()
