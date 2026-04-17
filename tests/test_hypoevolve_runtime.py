import json
import tempfile
import unittest
from unittest.mock import patch

from elg import AtomicNode, Hypothesis
from hypoevolve.runtime import (
    create_run_dir,
    write_artifact,
    write_best,
    write_checkpoint,
    write_run_summary,
    write_score_history,
    write_trace,
)


class TestHypoEvolveRuntime(unittest.TestCase):
    def test_runtime_writes_expected_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = create_run_dir(tmp, run_id='run1')
            write_trace(run_dir, {'iteration': 0})
            write_checkpoint(run_dir, {'iteration': 0})
            write_best(run_dir, Hypothesis(root=AtomicNode('A')), {'combined_score': 0.5})
            write_artifact(run_dir, 'seed', {'hello': 'world'})
            write_run_summary(run_dir, {'iterations_requested': 1})
            write_score_history(run_dir, [{'iteration': 0, 'score': 0.5}])
            self.assertTrue((run_dir / 'trace.jsonl').exists())
            self.assertTrue((run_dir / 'checkpoint.json').exists())
            self.assertTrue((run_dir / 'best.json').exists())
            self.assertTrue((run_dir / 'run_summary.json').exists())
            self.assertTrue((run_dir / 'score_history.json').exists())
            self.assertTrue((run_dir / 'artifacts' / 'seed.json').exists())
            checkpoint = json.loads((run_dir / "checkpoint.json").read_text(encoding="utf-8"))
            best = json.loads((run_dir / "best.json").read_text(encoding="utf-8"))
            summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
            history = json.loads((run_dir / "score_history.json").read_text(encoding="utf-8"))
            self.assertEqual(checkpoint["iteration"], 0)
            self.assertEqual(sorted(best), ["hypothesis", "metrics"])
            self.assertEqual(summary["iterations_requested"], 1)
            self.assertEqual(history[0]["iteration"], 0)


    def test_create_run_dir_generates_id_and_logs_event(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake_uuid = type('FakeUUID', (), {'hex': 'abcdef1234567890'})()
            with patch('hypoevolve.runtime.uuid.uuid4', return_value=fake_uuid),                  patch('hypoevolve.runtime.log_info_event') as log_info_event:
                run_dir = create_run_dir(tmp)

            self.assertEqual(run_dir.name, 'abcdef12')
            self.assertTrue((run_dir / 'artifacts').exists())
            log_info_event.assert_called_once_with('run_dir.create', run='abcdef12', path=run_dir)

    def test_write_trace_appends_sorted_json_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = create_run_dir(tmp, run_id='run-trace')
            write_trace(run_dir, {'b': 2, 'a': 1})
            write_trace(run_dir, {'iteration': 2})

            lines = (run_dir / 'trace.jsonl').read_text(encoding='utf-8').splitlines()

        self.assertEqual(lines[0], '{"a": 1, "b": 2}')
        self.assertEqual(json.loads(lines[1]), {'iteration': 2})

    def test_write_best_serializes_hypothesis_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = create_run_dir(tmp, run_id='run-best')
            write_best(run_dir, Hypothesis(root=AtomicNode('BEST')), {'combined_score': 0.9})
            payload = json.loads((run_dir / 'best.json').read_text(encoding='utf-8'))

        self.assertEqual(payload['hypothesis']['root']['name'], 'BEST')
        self.assertEqual(payload['metrics']['combined_score'], 0.9)

    def test_write_artifact_writes_named_json_under_artifacts_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = create_run_dir(tmp, run_id='run-artifact')
            path = write_artifact(run_dir, 'iteration_0001', {'hello': 'world'})
            payload = json.loads(path.read_text(encoding='utf-8'))

        self.assertEqual(path.name, 'iteration_0001.json')
        self.assertEqual(payload, {'hello': 'world'})


    def test_write_checkpoint_persists_sorted_json_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = create_run_dir(tmp, run_id='run-checkpoint')
            path = write_checkpoint(run_dir, {'b': 2, 'a': 1})
            self.assertEqual(path.name, 'checkpoint.json')
            self.assertEqual(json.loads(path.read_text(encoding='utf-8')), {'a': 1, 'b': 2})

    def test_write_run_summary_persists_summary_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = create_run_dir(tmp, run_id='run-summary')
            path = write_run_summary(run_dir, {'iterations_requested': 3, 'best_score': 0.9})
            self.assertEqual(path.name, 'run_summary.json')
            self.assertEqual(
                json.loads(path.read_text(encoding='utf-8')),
                {'iterations_requested': 3, 'best_score': 0.9},
            )

    def test_write_score_history_persists_iteration_list(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = create_run_dir(tmp, run_id='run-history')
            path = write_score_history(run_dir, [{'iteration': 0, 'score': 0.1}, {'iteration': 1, 'score': 0.2}])
            self.assertEqual(path.name, 'score_history.json')
            self.assertEqual(
                json.loads(path.read_text(encoding='utf-8')),
                [{'iteration': 0, 'score': 0.1}, {'iteration': 1, 'score': 0.2}],
            )
