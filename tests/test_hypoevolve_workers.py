import unittest

from elg import AtomicNode, Hypothesis
from hypoevolve.workers import WorkerResult, WorkerTask, run_worker_task


class TestHypoEvolveWorkers(unittest.TestCase):
    def test_worker_task_and_result_execute_one_step(self):
        task = WorkerTask(
            parent_hypothesis=Hypothesis(root=AtomicNode('A')).to_dict(),
            iteration=1,
            parent_score=0.5,
            mutation_atomic_pool=['B'],
            evaluator_seed=42,
        )
        result = run_worker_task(task)
        self.assertIsInstance(result, WorkerResult)
        self.assertEqual(result.iteration, 1)
        self.assertIn('combined_score', result.metrics)
        self.assertIn('root', result.child_hypothesis)
