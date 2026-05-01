import json
import tempfile
from pathlib import Path
import unittest

from hypoevolve.elg import AtomicNode, Hypothesis
from hypoevolve.memory.archive import MAPElitesArchive
from hypoevolve.memory.artifacts import (
    build_checkpoint_payload,
    build_history_entry,
    build_run_summary_payload,
    build_trace_event,
)
from hypoevolve.memory.artifacts import RunArtifactRecorder
from hypoevolve.runtime.checkpoint import create_run_dir


class TestRunArtifactRecorder(unittest.TestCase):
    def test_recorder_creates_artifact_directory_on_init(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run-no-layout"
            recorder = RunArtifactRecorder(
                run_dir=run_dir,
                seed_input_text="if A then B",
                worker_count=1,
                workers_enabled=False,
                dataset_schema_path="dataset.yaml",
            )

            self.assertEqual(recorder.run_dir, run_dir)
            self.assertTrue(run_dir.exists())
            self.assertTrue((run_dir / "artifacts").exists())

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
            best_payload = json.loads(
                (run_dir / "best.json").read_text(encoding="utf-8")
            )
            checkpoint_payload = json.loads(
                (run_dir / "checkpoint.json").read_text(encoding="utf-8")
            )
            run_summary = json.loads(
                (run_dir / "run_summary.json").read_text(encoding="utf-8")
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
            self.assertEqual(sorted(best_payload), ["hypothesis", "metrics"])
            self.assertIn("iteration", checkpoint_payload)
            self.assertIn("archive", checkpoint_payload)
            self.assertIn("best_hypothesis", checkpoint_payload)
            self.assertIn("best_metrics", checkpoint_payload)
            self.assertEqual(
                run_summary,
                build_run_summary_payload(
                    archive=archive,
                    seed_input_text="if A then B",
                    iterations_requested=2,
                    worker_count=1,
                    workers_enabled=False,
                    dataset_schema_path="dataset.yaml",
                    duplicate_skips_solo=1,
                    duplicate_skips_worker=0,
                    known_fingerprint_count=2,
                    best_hypothesis_nl="Child B.",
                ),
            )
            self.assertEqual(run_summary["iterations_requested"], 2)
            self.assertEqual(run_summary["archive_size"], 2)
            self.assertEqual(run_summary["duplicate_skips_total"], 1)
            self.assertEqual(run_summary["best_score"], 0.3)
            self.assertEqual(len(top_manifest), 1)
            self.assertEqual(
                sorted(top_manifest[0].keys()),
                [
                    "candidate_code",
                    "fingerprint",
                    "hypothesis",
                    "iteration",
                    "metadata",
                    "metrics",
                    "rank",
                    "score",
                    "wrapper_code",
                ],
            )
            self.assertIn("candidate_code", top_manifest[0])
            self.assertTrue((run_dir / top_manifest[0]["candidate_code"]).exists())
            self.assertEqual(score_history[0]["status"], "seed")
            self.assertEqual(score_history[1]["status"], "evaluated")
            self.assertEqual(score_history[2]["status"], "skipped_duplicate")
            self.assertNotIn("score", score_history[2])
            self.assertIn(
                "Child B.",
                (run_dir / "report" / "report.md").read_text(encoding="utf-8"),
            )
            report_text = (run_dir / "report" / "report.md").read_text(encoding="utf-8")
            for required_section in (
                "# HypoEvolve Final Report",
                "## Executive Summary",
                "## Key Metrics",
                "## Best ELG",
                "## Search Overview",
            ):
                self.assertIn(required_section, report_text)


    def test_cache_evaluation_artifacts_skips_empty_payloads_and_keeps_non_empty(self):
        recorder = RunArtifactRecorder(
            run_dir=Path('/tmp/non-persistent-run'),
            seed_input_text='if A then B',
            worker_count=1,
            workers_enabled=False,
            dataset_schema_path='dataset.yaml',
        )
        hypothesis = Hypothesis(root=AtomicNode('A'))

        recorder._cache_evaluation_artifacts(hypothesis, {})
        self.assertEqual(recorder.evaluation_artifact_cache, {})

        recorder._cache_evaluation_artifacts(hypothesis, {'candidate_code': 'print(1)'})
        self.assertEqual(
            recorder.evaluation_artifact_cache[next(iter(recorder.evaluation_artifact_cache))],
            {'candidate_code': 'print(1)'},
        )

    def test_checkpoint_trace_and_history_helpers_return_expected_payloads(self):
        recorder = RunArtifactRecorder(
            run_dir=Path('/tmp/non-persistent-run'),
            seed_input_text='if A then B',
            worker_count=1,
            workers_enabled=False,
            dataset_schema_path='dataset.yaml',
        )
        archive = MAPElitesArchive()
        parent = Hypothesis(root=AtomicNode('A'))
        child = Hypothesis(root=AtomicNode('B'))
        archive.add(child, {'combined_score': 0.4}, iteration=2, metadata={'mutation_summary': 'replace A with B'})
        descriptor = archive.describe(child, {'combined_score': 0.4})

        checkpoint = recorder._checkpoint_payload(archive, 2)
        trace_event = recorder._trace_event(2, parent, child, {'combined_score': 0.4}, {'mutation_summary': 'replace A with B'})
        history_entry = recorder._history_entry(
            iteration=2,
            hypothesis=child,
            metrics={'combined_score': 0.4, 'precision': 0.7},
            best_score_after=0.4,
            best_updated=True,
            status='evaluated',
            metadata={'mutation_summary': 'replace A with B', 'worker_mode': True},
            descriptor=descriptor,
            parent_fingerprint='parent-fp',
        )

        self.assertEqual(checkpoint['iteration'], 2)
        self.assertEqual(checkpoint['archive_size'], 1)
        self.assertEqual(checkpoint['best_hypothesis']['root']['name'], 'B')
        self.assertEqual(trace_event['parent']['root']['name'], 'A')
        self.assertEqual(trace_event['child']['root']['name'], 'B')
        self.assertEqual(trace_event['metadata']['mutation_summary'], 'replace A with B')
        self.assertEqual(history_entry['status'], 'evaluated')
        self.assertEqual(history_entry['parent_fingerprint'], 'parent-fp')
        self.assertEqual(history_entry['hypothesis_nl'], 'B')
        self.assertTrue(history_entry['worker_mode'])
        self.assertEqual(history_entry['cell'], list(descriptor['cell']))
        self.assertEqual(checkpoint, build_checkpoint_payload(archive, 2))
        self.assertEqual(
            trace_event,
            build_trace_event(
                2,
                parent,
                child,
                {'combined_score': 0.4},
                {'mutation_summary': 'replace A with B'},
            ),
        )
        self.assertEqual(
            history_entry,
            build_history_entry(
                iteration=2,
                hypothesis=child,
                metrics={'combined_score': 0.4, 'precision': 0.7},
                best_score_after=0.4,
                best_updated=True,
                status='evaluated',
                metadata={'mutation_summary': 'replace A with B', 'worker_mode': True},
                descriptor=descriptor,
                parent_fingerprint='parent-fp',
            ),
        )

    def test_materialize_top_k_evaluator_artifacts_writes_only_ranked_entries(self):
        with tempfile.TemporaryDirectory() as tmp:
            recorder = RunArtifactRecorder(
                run_dir=Path(tmp),
                seed_input_text='if A then B',
                worker_count=1,
                workers_enabled=False,
                dataset_schema_path='dataset.yaml',
                top_k_code_artifacts=1,
            )
            archive = MAPElitesArchive()
            best = Hypothesis(root=AtomicNode('BEST'))
            other = Hypothesis(root=AtomicNode('OTHER'))
            archive.add(best, {'combined_score': 0.9}, iteration=1)
            archive.add(other, {'combined_score': 0.2}, iteration=2)
            recorder._cache_evaluation_artifacts(best, {'candidate_code': 'print(1)', 'wrapper_code': 'print(2)', 'attempt': 1})
            recorder._cache_evaluation_artifacts(other, {'candidate_code': 'print(3)', 'wrapper_code': 'print(4)', 'attempt': 2})

            recorder._materialize_top_k_evaluator_artifacts(archive)

            manifest = json.loads((Path(tmp) / 'artifacts' / 'top_evaluators.json').read_text(encoding='utf-8'))
            self.assertEqual(len(manifest), 1)
            self.assertEqual(manifest[0]['iteration'], 1)
            self.assertIn('candidate_code', manifest[0])
            self.assertIn('wrapper_code', manifest[0])
            self.assertIn('metadata', manifest[0])
            self.assertTrue((Path(tmp) / manifest[0]['candidate_code']).exists())
            self.assertTrue((Path(tmp) / manifest[0]['wrapper_code']).exists())
            self.assertTrue((Path(tmp) / manifest[0]['metadata']).exists())
