import tempfile
import unittest
from pathlib import Path

from hypoevolve.elg import AtomicNode, Hypothesis, RelationNode
from hypoevolve.logger import (
    compact_text,
    configure_logger,
    event_message,
    log_debug_event,
    log_info_event,
    logger,
    _format_log_value,
    _quote_if_needed,
    summarize_exception,
    summarize_hypothesis,
    summarize_metrics,
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

    def test_debug_events_are_suppressed_by_info_level_file_sink(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / 'hypoevolve.log'
            configure_logger('INFO', log_path)

            log_debug_event('parser.attempt', attempt=1)
            log_info_event('run.start', run='demo')

            log_text = log_path.read_text(encoding='utf-8')

        self.assertNotIn('event=parser.attempt', log_text)
        self.assertIn('event=run.start run=demo', log_text)

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


    def test_compact_text_collapses_whitespace_and_truncates(self):
        self.assertEqual(compact_text("A   B\nC"), "A B C")
        self.assertTrue(compact_text("x" * 200, max_len=20).endswith("…"))

    def test_summarize_exception_extracts_keyerror_and_exit_code(self):
        summary = summarize_exception(RuntimeError("Generated evaluator code failed with exit_code=3: KeyError: 'MISSING_COL'"))
        self.assertEqual(summary["err_type"], "RuntimeError")
        self.assertEqual(summary["missing_col"], "MISSING_COL")
        self.assertEqual(summary["exit_code"], 3)

    def test_summarize_metrics_returns_compact_scalar_fields(self):
        summary = summarize_metrics({
            "combined_score": 0.7,
            "precision": 0.6,
            "baseline": 0.1,
            "coverage": 0.3,
            "uplift": 0.5,
        })
        self.assertEqual(summary, {"score": 0.7, "prec": 0.6, "base": 0.1, "cov": 0.3, "up": 0.5})


    def test_event_message_skips_none_and_quotes_structured_values(self):
        rendered = event_message(
            "iter.meta",
            i=1,
            note=None,
            flags=["a", "b"],
            path=Path("/tmp/demo path"),
        )
        self.assertIn('event=iter.meta', rendered)
        self.assertIn('i=1', rendered)
        self.assertNotIn('note=', rendered)
        self.assertIn('flags=', rendered)
        self.assertIn('a', rendered)
        self.assertIn('b', rendered)
        self.assertIn('path=/tmp/demo path', rendered)

    def test_format_log_value_and_quote_helpers_cover_scalar_edge_cases(self):
        self.assertEqual(_format_log_value(True), 'true')
        self.assertEqual(_format_log_value(False), 'false')
        self.assertEqual(_format_log_value(3), '3')
        self.assertEqual(_format_log_value(0.125), '0.125')
        self.assertEqual(_format_log_value(float('nan')), 'nan')
        self.assertEqual(_format_log_value(Path('/tmp/file')), '/tmp/file')
        self.assertEqual(_quote_if_needed('hello world'), '"hello world"')
        self.assertEqual(_quote_if_needed(''), '""')
        self.assertEqual(_quote_if_needed('alpha'), 'alpha')


if __name__ == '__main__':
    unittest.main()
