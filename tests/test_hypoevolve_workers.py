import unittest
from unittest.mock import patch

from elg import AtomicNode, Hypothesis, fingerprint
from hypoevolve.workers import WorkerResult, WorkerTask, run_worker_task


class TestHypoEvolveWorkers(unittest.TestCase):
    def test_worker_task_and_result_execute_one_step(self):
        task = WorkerTask(
            parent_hypothesis=Hypothesis(root=AtomicNode('A')).to_dict(),
            parent_metrics={"combined_score": 0.1},
            iteration=1,
            parent_score=0.5,
            llm_config={},
            dataset_schema_path="dataset.yaml",
            evaluator_parameters={},
            parser_retries=1,
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
        with patch("hypoevolve.workers.LLMClient"), \
             patch("hypoevolve.workers.load_dataset_schema"), \
             patch("hypoevolve.workers.LLMEvaluator") as evaluator_cls, \
             patch("hypoevolve.workers.llm_hypothesis_to_natural_language", return_value="A"), \
             patch("hypoevolve.workers.steer_mutation", return_value=fake_decision):
            evaluator_cls.return_value.evaluate.return_value = {"combined_score": 0.2}
            result = run_worker_task(task)
        self.assertIsInstance(result, WorkerResult)
        self.assertEqual(result.iteration, 1)
        self.assertIn('combined_score', result.metrics)
        self.assertIn('root', result.child_hypothesis)
        self.assertIn("replace_atomic-style", result.mutation_summary)

    def test_worker_reuses_cached_parent_hypothesis_nl(self):
        task = WorkerTask(
            parent_hypothesis=Hypothesis(root=AtomicNode("A")).to_dict(),
            parent_metrics={"combined_score": 0.1},
            iteration=1,
            parent_score=0.5,
            parent_hypothesis_nl="Cached A.",
            llm_config={},
            dataset_schema_path="dataset.yaml",
            evaluator_parameters={},
            parser_retries=1,
            steering_retries=1,
            recent_history=[],
            top_hypotheses=[],
        )
        fake_decision = type(
            "FakeDecision",
            (),
            {
                "child_hypothesis": Hypothesis(root=AtomicNode("B")),
                "domain_reason": "Reuse the cached domain rationale.",
                "score_reason": "Reuse the cached score rationale.",
                "operation_score_rankings": {"replace_atomic_feature": 1},
                "mutation_summary": "Applied a replace_atomic-style local mutation.",
            },
        )()

        with patch("hypoevolve.workers.LLMClient"), \
             patch("hypoevolve.workers.load_dataset_schema"), \
             patch("hypoevolve.workers.LLMEvaluator") as evaluator_cls, \
             patch(
                 "hypoevolve.workers.llm_hypothesis_to_natural_language",
                 side_effect=AssertionError("should reuse cached NL"),
             ), \
             patch("hypoevolve.workers.steer_mutation", return_value=fake_decision) as steer_mutation_mock:
            evaluator_cls.return_value.evaluate.return_value = {"combined_score": 0.2}
            run_worker_task(task)

        self.assertEqual(
            steer_mutation_mock.call_args.kwargs["parent_hypothesis_nl"], "Cached A."
        )

    def test_worker_threads_random_steering_flag(self):
        task = WorkerTask(
            parent_hypothesis=Hypothesis(root=AtomicNode("A")).to_dict(),
            parent_metrics={"combined_score": 0.1},
            iteration=1,
            parent_score=0.5,
            parent_hypothesis_nl="Cached A.",
            use_random_steering=True,
            llm_config={},
            dataset_schema_path="dataset.yaml",
            evaluator_parameters={},
            parser_retries=1,
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

        with patch("hypoevolve.workers.LLMClient"), \
             patch("hypoevolve.workers.load_dataset_schema"), \
             patch("hypoevolve.workers.LLMEvaluator") as evaluator_cls, \
             patch("hypoevolve.workers.steer_mutation", return_value=fake_decision) as steer_mutation_mock:
            evaluator_cls.return_value.evaluate.return_value = {"combined_score": 0.2}
            result = run_worker_task(task)

        self.assertTrue(steer_mutation_mock.call_args.kwargs["use_random_steering"])
        self.assertTrue(result.random_steering)

    def test_worker_skips_evaluation_for_seen_child_fingerprint(self):
        child = Hypothesis(root=AtomicNode("B"))
        task = WorkerTask(
            parent_hypothesis=Hypothesis(root=AtomicNode("A")).to_dict(),
            parent_metrics={"combined_score": 0.1},
            iteration=1,
            parent_score=0.5,
            parent_hypothesis_nl="Cached A.",
            llm_config={},
            dataset_schema_path="dataset.yaml",
            evaluator_parameters={},
            parser_retries=1,
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

        with patch("hypoevolve.workers.LLMClient"), \
             patch("hypoevolve.workers.load_dataset_schema"), \
             patch("hypoevolve.workers.LLMEvaluator") as evaluator_cls, \
             patch("hypoevolve.workers.steer_mutation", return_value=fake_decision):
            result = run_worker_task(task)

        evaluator_cls.return_value.evaluate.assert_not_called()
        self.assertTrue(result.skipped_duplicate)
        self.assertEqual(result.child_fingerprint, fingerprint(child))
        self.assertEqual(result.metrics, {})
