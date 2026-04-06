import json
import tempfile
import unittest

from elg import AtomicNode, Hypothesis
from hypoevolve.archive import MAPElitesArchive
from hypoevolve.artifacts import RunArtifactRecorder
from hypoevolve.runtime import create_run_dir


class TestRunArtifactRecorder(unittest.TestCase):
    def test_recorder_persists_run_artifacts_and_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = create_run_dir(tmp, run_id="run1")
            recorder = RunArtifactRecorder(
                run_dir=run_dir,
                seed_input_text="if A then B",
                worker_count=1,
                workers_enabled=False,
                dataset_schema_path="dataset.yaml",
                top_k_code_artifacts=1,
            )
            archive = MAPElitesArchive()

            seed = Hypothesis(root=AtomicNode("A"))
            seed_metrics = {"combined_score": 0.1, "coverage": 0.2}
            seed_metadata = {"source": "seed", "hypothesis_nl": "Seed A."}
            seed_descriptor = archive.describe(seed, seed_metrics)
            archive.add(seed, seed_metrics, iteration=0, metadata=seed_metadata)
            recorder.record_seed(
                archive=archive,
                hypothesis=seed,
                metrics=seed_metrics,
                metadata=seed_metadata,
                descriptor=seed_descriptor,
                evaluation_artifacts={
                    "candidate_code": "def evaluate_hypothesis():\n    return {'combined_score': 0.1}\n",
                    "wrapper_code": "print('seed wrapper')\n",
                    "work_dir": "/tmp/seed",
                },
            )

            child = Hypothesis(root=AtomicNode("B"))
            child_metrics = {"combined_score": 0.3, "coverage": 0.4}
            child_metadata = {
                "mutation_summary": "replace A with B",
                "hypothesis_nl": "Child B.",
            }
            child_descriptor = archive.describe(child, child_metrics)
            archive.add(child, child_metrics, iteration=1, metadata=child_metadata)
            recorder.record_iteration_result(
                archive=archive,
                iteration=1,
                parent_hypothesis=seed,
                parent_fingerprint=archive.entries[-1].fingerprint,
                child_hypothesis=child,
                child_metrics=child_metrics,
                metadata=child_metadata,
                descriptor=child_descriptor,
                best_updated=True,
                evaluation_artifacts={
                    "candidate_code": "def evaluate_hypothesis():\n    return {'combined_score': 0.3}\n",
                    "wrapper_code": "print('child wrapper')\n",
                    "work_dir": "/tmp/child",
                },
            )

            recorder.record_duplicate_skip(
                iteration=2,
                parent_fingerprint=archive.best.fingerprint,
                child_fingerprint=archive.best.fingerprint,
                parent_score=archive.best.score,
                best_score_after=archive.best.score,
                worker_mode=False,
            )

            report_path = recorder.finalize(
                archive=archive,
                iterations_requested=2,
                known_fingerprint_count=2,
                best_hypothesis_nl="Child B.",
            )

            self.assertTrue((run_dir / "best.json").exists())
            self.assertTrue((run_dir / "checkpoint.json").exists())
            self.assertTrue((run_dir / "trace.jsonl").exists())
            self.assertTrue((run_dir / "run_summary.json").exists())
            self.assertTrue((run_dir / "score_history.json").exists())
            self.assertFalse((run_dir / "artifacts" / "seed_candidate.py").exists())
            self.assertFalse(
                (run_dir / "artifacts" / "iteration_0001_candidate.py").exists()
            )
            self.assertTrue((run_dir / "artifacts" / "top_evaluators.json").exists())
            self.assertTrue(report_path.exists())
            self.assertTrue((run_dir / "report" / "assets" / "score_progression.svg").exists())
            score_history = json.loads(
                (run_dir / "score_history.json").read_text(encoding="utf-8")
            )
            iteration_payload = json.loads(
                (run_dir / "artifacts" / "iteration_0001.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(len(score_history), 3)
            self.assertEqual(recorder.duplicate_skips_solo, 1)
            self.assertNotIn("evaluation_artifacts", iteration_payload)
            top_manifest = json.loads(
                (run_dir / "artifacts" / "top_evaluators.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(top_manifest), 1)
            self.assertIn("candidate_code", top_manifest[0])
            self.assertTrue((run_dir / top_manifest[0]["candidate_code"]).exists())
            self.assertIn(
                "Child B.",
                (run_dir / "report" / "report.md").read_text(encoding="utf-8"),
            )
