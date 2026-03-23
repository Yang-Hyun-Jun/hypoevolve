import json
import tempfile
import unittest
from pathlib import Path

from elg import AtomicNode, Hypothesis
from hypoevolve.runtime import create_run_dir, write_artifact, write_best, write_checkpoint, write_trace


class TestHypoEvolveRuntime(unittest.TestCase):
    def test_runtime_writes_expected_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = create_run_dir(tmp, run_id='run1')
            write_trace(run_dir, {'iteration': 0})
            write_checkpoint(run_dir, {'iteration': 0})
            write_best(run_dir, Hypothesis(root=AtomicNode('A')), {'combined_score': 0.5})
            write_artifact(run_dir, 'seed', {'hello': 'world'})
            self.assertTrue((run_dir / 'trace.jsonl').exists())
            self.assertTrue((run_dir / 'checkpoint.json').exists())
            self.assertTrue((run_dir / 'best.json').exists())
            self.assertTrue((run_dir / 'artifacts' / 'seed.json').exists())
