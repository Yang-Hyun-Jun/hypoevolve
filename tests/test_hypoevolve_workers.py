import unittest
from unittest.mock import patch

from hypoevolve.elg import AtomicNode, Hypothesis, fingerprint
from hypoevolve.skills.elg_compile import ParseError
from hypoevolve.runtime.worker import (
    WorkerResult as WorkerResultContract,
    WorkerTask as WorkerTaskContract,
)
from hypoevolve.runtime.worker import WorkerResult, WorkerTask, run_worker_task


class TestHypoEvolveWorkers(unittest.TestCase):
    def test_workers_module_re_exports_contract_types(self):
        self.assertIs(WorkerTask, WorkerTaskContract)
        self.assertIs(WorkerResult, WorkerResultContract)

    def test_run_worker_task_returns_scored_worker_result_with_child_hypothesis_payload(self):
        task = WorkerTask(
            parent_hypothesis=Hypothesis(root=AtomicNode('A')).to_dict(),
            parent_metrics={"combined_score": 0.1},
            iteration=1,
            parent_score=0.5,
            llm_config={},
            dataset_schema_path="dataset.yaml",
            evaluator_parameters={},
            steering_retries=1,
            recent_history=[],
            top_hypotheses=[],
        )
        fake_decision = type(
            "FakeDecision",
            (),
            {
                "child_hypothesis": Hypothesis(root=AtomicNode("B")),
                "domain_reason": "This local edit is plausible from a domain perspective.",
                "score_reason": "Pick the first legal candidate for score improvement.",
                "operation_score_rankings": {"replace_atomic_feature": 1, "append_atomic": 2},
                "mutation_summary": "Applied a replace_atomic-style local mutation.",
            },
        )()
        with patch("hypoevolve.runtime.worker.LLMClient"), \
             patch("hypoevolve.runtime.worker.load_dataset_schema"), \
             patch("hypoevolve.runtime.worker.LLMEvaluator") as evaluator_cls, \
             patch("hypoevolve.runtime.worker.steer_mutation", return_value=fake_decision):
            evaluator_cls.return_value.evaluate.return_value = {"combined_score": 0.2}
            result = run_worker_task(task)
        self.assertIsInstance(result, WorkerResult)
        self.assertEqual(result.iteration, 1)
        self.assertIn('combined_score', result.metrics)
        self.assertIn('root', result.child_hypothesis)
        self.assertIn("replace_atomic-style", result.mutation_summary)

    def test_run_worker_task_propagates_random_steering_flag_into_mutation_and_result(self):
        task = WorkerTask(
            parent_hypothesis=Hypothesis(root=AtomicNode("A")).to_dict(),
            parent_metrics={"combined_score": 0.1},
            iteration=1,
            parent_score=0.5,
            use_random_steering=True,
            llm_config={},
            dataset_schema_path="dataset.yaml",
            evaluator_parameters={},
            steering_retries=1,
            recent_history=[],
            top_hypotheses=[],
        )
        fake_decision = type(
            "FakeDecision",
            (),
            {
                "child_hypothesis": Hypothesis(root=AtomicNode("B")),
                "domain_reason": "",
                "score_reason": "",
                "mutation_summary": "Applied three exploratory local mutations.",
            },
        )()

        with patch("hypoevolve.runtime.worker.LLMClient"), \
             patch("hypoevolve.runtime.worker.load_dataset_schema"), \
             patch("hypoevolve.runtime.worker.LLMEvaluator") as evaluator_cls, \
             patch("hypoevolve.runtime.worker.steer_mutation", return_value=fake_decision) as steer_mutation_mock:
            evaluator_cls.return_value.evaluate.return_value = {"combined_score": 0.2}
            result = run_worker_task(task)

        self.assertTrue(steer_mutation_mock.call_args.kwargs["use_random_steering"])
        self.assertTrue(result.random_steering)

    def test_run_worker_task_skips_evaluation_for_seen_child_fingerprint(self):
        child = Hypothesis(root=AtomicNode("B"))
        task = WorkerTask(
            parent_hypothesis=Hypothesis(root=AtomicNode("A")).to_dict(),
            parent_metrics={"combined_score": 0.1},
            iteration=1,
            parent_score=0.5,
            llm_config={},
            dataset_schema_path="dataset.yaml",
            evaluator_parameters={},
            steering_retries=1,
            recent_history=[],
            top_hypotheses=[],
            seen_fingerprints=[fingerprint(child)],
        )
        fake_decision = type(
            "FakeDecision",
            (),
            {
                "child_hypothesis": child,
                "domain_reason": "Duplicate child remains plausible.",
                "score_reason": "Skip repeated evaluation.",
                "operation_score_rankings": {"replace_atomic_feature": 1},
                "mutation_summary": "Reproduced an already known child.",
            },
        )()

        with patch("hypoevolve.runtime.worker.LLMClient"), \
             patch("hypoevolve.runtime.worker.load_dataset_schema"), \
             patch("hypoevolve.runtime.worker.LLMEvaluator") as evaluator_cls, \
             patch("hypoevolve.runtime.worker.steer_mutation", return_value=fake_decision):
            result = run_worker_task(task)

        evaluator_cls.return_value.evaluate.assert_not_called()
        self.assertTrue(result.skipped_duplicate)
        self.assertEqual(result.child_fingerprint, fingerprint(child))
        self.assertEqual(result.metrics, {})
        self.assertIn("root", result.child_hypothesis)

    def test_run_worker_task_returns_skipped_steering_error_worker_result(self):
        task = WorkerTask(
            parent_hypothesis=Hypothesis(root=AtomicNode("A")).to_dict(),
            parent_metrics={"combined_score": 0.1},
            iteration=1,
            parent_score=0.5,
            llm_config={},
            dataset_schema_path="dataset.yaml",
            evaluator_parameters={},
            steering_retries=1,
            recent_history=[],
            top_hypotheses=[],
        )

        with patch("hypoevolve.runtime.worker.LLMClient"), \
             patch("hypoevolve.runtime.worker.load_dataset_schema"), \
             patch("hypoevolve.runtime.worker.LLMEvaluator") as evaluator_cls, \
             patch(
                 "hypoevolve.runtime.worker.steer_mutation",
                 side_effect=ParseError("Failed to steer mutation via LLM"),
             ):
            result = run_worker_task(task)

        evaluator_cls.return_value.evaluate.assert_not_called()
        self.assertTrue(result.skipped_steering_error)
        self.assertEqual(result.metrics, {})
        self.assertEqual(result.child_hypothesis, {})
        self.assertIn("Failed to steer mutation via LLM", result.steering_error)


    def test_run_worker_task_passes_rehydrated_top_hypotheses_into_steer_mutation(self):
        task = WorkerTask(
            parent_hypothesis=Hypothesis(root=AtomicNode('A')).to_dict(),
            parent_metrics={'combined_score': 0.1},
            iteration=1,
            parent_score=0.5,
            llm_config={},
            dataset_schema_path='dataset.yaml',
            top_hypotheses=[
                {
                    'hypothesis': Hypothesis(root=AtomicNode('TOP')).to_dict(),
                    'metrics': {'combined_score': 0.9},
                    'fingerprint': 'top-fp',
                    'iteration': 3,
                    'metadata': {'map_elites': {'coverage_bin': 0}},
                }
            ],
        )
        fake_decision = type(
            'FakeDecision',
            (),
            {
                'child_hypothesis': Hypothesis(root=AtomicNode('B')),
                'domain_reason': '',
                'score_reason': '',
                'operation_score_rankings': {},
                'mutation_summary': 'Applied a local mutation.',
            },
        )()

        with patch('hypoevolve.runtime.worker.LLMClient'),              patch('hypoevolve.runtime.worker.load_dataset_schema'),              patch('hypoevolve.runtime.worker.LLMEvaluator') as evaluator_cls,              patch('hypoevolve.runtime.worker.steer_mutation', return_value=fake_decision) as steer_mutation_mock:
            evaluator_cls.return_value.evaluate.return_value = {'combined_score': 0.2}
            run_worker_task(task)

        top_hypotheses = steer_mutation_mock.call_args.kwargs['top_hypotheses']
        self.assertEqual(len(top_hypotheses), 1)
        self.assertEqual(top_hypotheses[0].fingerprint, 'top-fp')
        self.assertEqual(top_hypotheses[0].iteration, 3)
        self.assertEqual(top_hypotheses[0].metadata['map_elites']['coverage_bin'], 0)
        self.assertEqual(top_hypotheses[0].hypothesis.root.name, 'TOP')

    def test_run_worker_task_includes_last_evaluation_artifacts_and_reasons_in_returned_worker_result(self):
        task = WorkerTask(
            parent_hypothesis=Hypothesis(root=AtomicNode('A')).to_dict(),
            parent_metrics={'combined_score': 0.1},
            iteration=2,
            parent_score=0.5,
            llm_config={},
            dataset_schema_path='dataset.yaml',
        )
        fake_decision = type(
            'FakeDecision',
            (),
            {
                'child_hypothesis': Hypothesis(root=AtomicNode('B')),
                'domain_reason': 'domain',
                'score_reason': 'score',
                'operation_score_rankings': {'replace_atomic': 1},
                'mutation_summary': 'Applied a local mutation.',
            },
        )()

        with patch('hypoevolve.runtime.worker.LLMClient'),              patch('hypoevolve.runtime.worker.load_dataset_schema'),              patch('hypoevolve.runtime.worker.LLMEvaluator') as evaluator_cls,              patch('hypoevolve.runtime.worker.steer_mutation', return_value=fake_decision):
            evaluator_cls.return_value.evaluate.return_value = {'combined_score': 0.2}
            evaluator_cls.return_value.last_evaluation_artifacts = {'candidate_code': 'print(1)'}
            result = run_worker_task(task)

        self.assertEqual(result.evaluation_artifacts, {'candidate_code': 'print(1)'})
        self.assertEqual(result.domain_reason, 'domain')
        self.assertEqual(result.score_reason, 'score')
        self.assertEqual(result.operation_score_rankings, {'replace_atomic': 1})
