import unittest
from unittest.mock import patch

from elg import AtomicNode, Hypothesis
from hypoevolve.workers import WorkerResult, WorkerTask, run_worker_task


class TestHypoEvolveWorkers(unittest.TestCase):
    def test_worker_task_and_result_execute_one_step(self):
        task = WorkerTask(
            parent_hypothesis=Hypothesis(root=AtomicNode('A')).to_dict(),
            parent_metrics={"combined_score": 0.1},
            iteration=1,
            parent_score=0.5,
            mutation_atomic_pool=['B'],
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
                "reason": "Pick the first legal candidate.",
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
            mutation_atomic_pool=["B"],
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
                "reason": "Reuse the cached NL.",
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
