import json
import unittest
from pathlib import Path

from elg import AtomicNode, Hypothesis, LogicalNode, RelationNode
from hypoevolve.dataset import ColumnSpec, DataFile, DatasetSchema, IndexSpec
from hypoevolve.evaluator import LLMEvaluator
from hypoevolve.executor import ExecutionResult
from tests.test_hypoevolve_evaluator import FakeExecutor, FakeLLMClient


def _normalize_prompt(text: str) -> str:
    return text.replace("File '/tmp/candidate.py'", "File '<CANDIDATE_PATH>'")


def _normalize_artifacts(payload: dict) -> dict:
    normalized = dict(payload)
    normalized['wrapper_code'] = (
        normalized['wrapper_code']
        .replace('"<PROJECT_ROOT>/dataset.yaml"', '"<DATASET_SCHEMA_PATH>"')
        .replace('"dataset.yaml"', '"<DATASET_SCHEMA_PATH>"')
        .replace('"/home/hjyang/workspace/hypoevolve/dataset.yaml"', '"<DATASET_SCHEMA_PATH>"')
        .replace('/home/hjyang/workspace/hypoevolve', '<PROJECT_ROOT>')
    )
    normalized['work_dir'] = '<WORK_DIR>'
    return normalized


class TestPhase0EvaluatorRepairFixture(unittest.TestCase):
    def test_evaluator_repair_fixture_matches_stubbed_run(self):
        fixture_root = Path('tests/fixtures/parity/evaluator_repair/normalized_expected')
        schema = DatasetSchema(
            files=[DataFile(entity='BTCUSDT', path='BTCUSDT.parquet')],
            index=IndexSpec(name='close_time', dtype='datetime64[us]'),
            columns=[ColumnSpec(name='CLOSE', description='close price')],
            description='Test dataset',
        )
        hypothesis = Hypothesis(
            root=RelationNode(
                'IMPLIES',
                [LogicalNode('AND', [AtomicNode('A'), AtomicNode('B')]), AtomicNode('C')],
            )
        )
        llm = FakeLLMClient(
            outputs=[
                "def evaluate_hypothesis(accessor: DatasetAccessor, parameters: dict[str, object] | None = None) -> dict[str, object]:\n"
                "    df = accessor.load_dataframe('BTCUSDT')\n"
                "    _ = df['UNKNOWN_COL']\n"
                "    return {'combined_score': 0.0, 'precision': 0.0, 'baseline': 0.0, 'coverage': 0.0, 'uplift': 0.0, 'support_count': 0, 'total_count': 0, 'rationale': 'first', 'used_parameters': {}}\n",
                "def evaluate_hypothesis(accessor: DatasetAccessor, parameters: dict[str, object] | None = None) -> dict[str, object]:\n"
                "    return {'combined_score': 0.1, 'precision': 0.2, 'baseline': 0.1, 'coverage': 0.5, 'uplift': 0.1, 'support_count': 1, 'total_count': 2, 'rationale': 'fixed', 'used_parameters': {'HORIZON': 1}}\n",
            ],
            retries=1,
        )
        executor = FakeExecutor(
            results=[
                ExecutionResult(
                    stdout='',
                    stderr="Traceback (most recent call last):\n  File '/tmp/candidate.py', line 3, in evaluate_hypothesis\nKeyError: 'UNKNOWN_COL'",
                    exit_code=1,
                    timed_out=False,
                    duration_sec=0.01,
                    work_dir='/tmp/fake',
                ),
                ExecutionResult(
                    stdout=json.dumps(
                        {
                            'combined_score': 0.1,
                            'precision': 0.2,
                            'baseline': 0.1,
                            'coverage': 0.5,
                            'uplift': 0.1,
                            'support_count': 1,
                            'total_count': 2,
                            'rationale': 'fixed',
                            'used_parameters': {'HORIZON': 1},
                        }
                    ),
                    stderr='',
                    exit_code=0,
                    timed_out=False,
                    duration_sec=0.01,
                    work_dir='/tmp/fake',
                ),
            ]
        )
        evaluator = LLMEvaluator(llm, schema, 'dataset.yaml', executor=executor)
        metrics = evaluator.evaluate(hypothesis)
        artifacts = _normalize_artifacts(evaluator.last_evaluation_artifacts)
        second_user_prompt = _normalize_prompt(llm.calls[1]['user'])

        expected_metrics = json.loads((fixture_root / 'metrics.json').read_text(encoding='utf-8'))
        expected_artifacts = json.loads((fixture_root / 'artifacts.json').read_text(encoding='utf-8'))
        expected_prompt = (fixture_root / 'second_user_prompt.txt').read_text(encoding='utf-8')

        self.assertEqual(metrics, expected_metrics)
        self.assertEqual(artifacts, expected_artifacts)
        self.assertEqual(second_user_prompt, expected_prompt)


if __name__ == '__main__':
    unittest.main()
