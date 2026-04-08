import tempfile
import unittest
from pathlib import Path

from elg import AtomicNode, Hypothesis, RelationNode
from hypoevolve.logger import (
    configure_logger,
    event_message,
    logger,
    summarize_hypothesis,
)


class TestHypoEvolveLogger(unittest.TestCase):
    def test_configure_logger_writes_file_sink(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / 'hypoevolve.log'
            configured = configure_logger('INFO', log_path)
            configured.info('hello logger')
            self.assertTrue(log_path.exists())
            self.assertIn('hello logger', log_path.read_text(encoding='utf-8'))
            self.assertIs(configured, logger)

    def test_event_message_renders_compact_logfmt_style(self):
        rendered = event_message(
            "iter.eval",
            i=3,
            score=0.125,
            note="compact summary",
        )
        self.assertEqual(
            rendered,
            'event=iter.eval i=3 score=0.125 note="compact summary"',
        )

    def test_summarize_hypothesis_returns_compact_structural_fields(self):
        hypothesis = Hypothesis(
            root=RelationNode(
                "IMPLIES",
                [AtomicNode("A"), AtomicNode("B")],
            )
        )
        summary = summarize_hypothesis(hypothesis)
        self.assertEqual(summary["root"], "IMPLIES")
        self.assertEqual(summary["nodes"], 3)
        self.assertEqual(summary["depth"], 2)
        self.assertEqual(summary["relations"], 1)
        self.assertEqual(summary["atomics"], 2)
        self.assertEqual(len(summary["fp"]), 12)


if __name__ == '__main__':
    unittest.main()
