import unittest

from hypoevolve.hypo.nodes.base import Node, NodeIOTypes
from hypoevolve.hypo.nodes.nodes import DATA


class _LeafNode(Node):
    """Leaf node used for base-node behavior tests."""

    def __init__(self, value):
        super().__init__(input_types=[], output_type=NodeIOTypes.FLOAT, max_childs=0)
        self.value = value

    @property
    def params(self) -> dict:
        return {"value": self.value}

    def activate(self):
        return self.value


class _BinaryNode(Node):
    """Binary node used for propagate/call tests."""

    def __init__(self):
        super().__init__(input_types=[NodeIOTypes.FLOAT], output_type=NodeIOTypes.FLOAT, max_childs=2)

    @property
    def params(self) -> dict:
        return {}

    def activate(self, left, right):
        return left + right


class _FakeProvider:
    def __init__(self, rows):
        self.rows = rows

    def has(self, ticker):
        return ticker in self.rows

    def get(self, ticker):
        return self.rows[ticker]


class TestHypoNodes(unittest.TestCase):
    def test_base_node_repr_description_full_call_and_propagate(self):
        left = _LeafNode(2)
        right = _LeafNode(3)
        parent = _BinaryNode()

        self.assertIn("Leaf node used", left.description)
        self.assertEqual(repr(parent), "_BinaryNode()")
        self.assertEqual(parent.full, False)
        self.assertEqual(parent(4, 5), 9)
        with self.assertRaises(ValueError):
            parent.propagate()

        parent.add_child(left)
        parent.add_child(right)
        self.assertTrue(parent.full)
        self.assertEqual(parent.propagate(), 5)
        self.assertIs(left.parent, parent)
        self.assertIs(right.parent, parent)

    def test_node_to_dict_and_from_dict_roundtrip_for_data_node(self):
        node = DATA(label="ALPHA", ticker="BTCUSDT", start_date="2024-01-01", end_date="2024-01-31")
        payload = node.to_dict()
        restored = Node.from_dict(payload)

        self.assertEqual(payload["class"], "DATA")
        self.assertEqual(payload["params"]["label"], "ALPHA")
        self.assertIsInstance(restored, DATA)
        self.assertEqual(restored.label, "ALPHA")
        self.assertEqual(restored.ticker, "BTCUSDT")
        self.assertEqual(restored.start_date, "2024-01-01")
        self.assertEqual(restored.end_date, "2024-01-31")

    def test_data_node_name_params_and_activate_with_provider(self):
        node = DATA(label="CLOSE", ticker="BTCUSDT", provider=_FakeProvider({"BTCUSDT": {"CLOSE": [1, 2, 3]}}))

        self.assertEqual(node.name, "DATA[CLOSE]")
        self.assertEqual(
            node.params,
            {
                "label": "CLOSE",
                "ticker": "BTCUSDT",
                "start_date": None,
                "end_date": None,
            },
        )
        self.assertEqual(node.activate(), [1, 2, 3])

    def test_data_node_activate_rejects_missing_provider_or_ticker(self):
        with self.assertRaises(ValueError):
            DATA(label="CLOSE", ticker="BTCUSDT", provider=None).activate()

        provider = _FakeProvider({"ETHUSDT": {"CLOSE": [1]}})
        with self.assertRaises(ValueError):
            DATA(label="CLOSE", ticker="BTCUSDT", provider=provider).activate()


if __name__ == "__main__":
    unittest.main()
