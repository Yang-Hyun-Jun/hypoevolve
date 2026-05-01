import unittest

from hypoevolve.data.yaml_parser import SimpleYAMLError, ensure_mapping, parse_simple_yaml


class TestHypoEvolveSimpleYaml(unittest.TestCase):
    def test_parse_simple_yaml_supports_scalars_lists_and_comments(self):
        payload = parse_simple_yaml(
            "# comment\n"
            "name: demo\n"
            "enabled: true\n"
            "threshold: 0.5\n"
            "count: 3\n"
            "items:\n"
            "  - alpha\n"
            "  - beta\n"
            "nested:\n"
            "  key: value\n"
        )
        self.assertEqual(payload["name"], "demo")
        self.assertTrue(payload["enabled"])
        self.assertEqual(payload["threshold"], 0.5)
        self.assertEqual(payload["count"], 3)
        self.assertEqual(payload["items"], ["alpha", "beta"])
        self.assertEqual(payload["nested"], {"key": "value"})

    def test_parse_simple_yaml_rejects_top_level_list_and_bad_indentation(self):
        with self.assertRaises(SimpleYAMLError):
            parse_simple_yaml("- item")
        with self.assertRaises(SimpleYAMLError):
            parse_simple_yaml("root:\n      overindented: true")

    def test_ensure_mapping_rejects_non_mapping(self):
        ensure_mapping({}, "config")
        with self.assertRaises(SimpleYAMLError):
            ensure_mapping([], "config")

    def test_parse_simple_yaml_parses_quoted_and_null_scalars(self):
        payload = parse_simple_yaml(
            "title: 'demo title'\n"
            'subtitle: "quoted"\n'
            "missing: null\n"
            "other_missing: none\n"
        )
        self.assertEqual(payload["title"], "demo title")
        self.assertEqual(payload["subtitle"], "quoted")
        self.assertIsNone(payload["missing"])
        self.assertIsNone(payload["other_missing"])

    def test_parse_simple_yaml_rejects_mixed_list_and_mapping_in_same_block(self):
        with self.assertRaises(SimpleYAMLError):
            parse_simple_yaml(
                "root:\n"
                "  - item\n"
                "  key: value\n"
            )


if __name__ == "__main__":
    unittest.main()
