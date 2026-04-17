import json
import tempfile
import unittest
from concurrent.futures import Future
from pathlib import Path
from unittest.mock import patch

from hypoevolve.config import HypoEvolveConfig
from hypoevolve.controller import HypoEvolveController
from hypoevolve.workers import WorkerResult


class ImmediateFuture(Future):
    def __init__(self, result):
        super().__init__()
        self.set_result(result)


class FakeExecutor:
    def __init__(self, max_workers=1):
        self.max_workers = max_workers

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def submit(self, fn, task):
        return ImmediateFuture(fn(task))


class FakeLLM:
    def generate_json(self, system, user, **kwargs):
        return {
            'kind': 'relation',
            'name': 'IMPLIES',
            'inputs': [
                {'kind': 'atomic', 'name': 'A'},
                {'kind': 'atomic', 'name': 'B'},
            ],
        }

    def generate_text(self, system, user, **kwargs):
        return 'If A then B.'


def fake_run_worker_task(task):
    return WorkerResult(
        child_hypothesis=task.parent_hypothesis,
        metrics={'combined_score': 0.6},
        iteration=task.iteration,
        mutation_summary='Applied a change_relation_type-style local mutation.',
        parent_score=task.parent_score,
        domain_reason='Use a plausible local relation-type mutation.',
        score_reason='Use a local relation-type mutation.',
        operation_score_rankings={'change_relation_type': 1},
    )


def _normalize_multi_worker_checkpoint(payload: dict) -> dict:
    normalized = json.loads(json.dumps(payload))
    for entry in normalized.get('archive', []):
        if entry.get('iteration') in {1, 2}:
            entry['iteration'] = 1
    return normalized


def _normalize_multi_worker_score_history(payload: list[dict]) -> list[dict]:
    normalized = json.loads(json.dumps(payload))
    concurrent_rows = [row for row in normalized if row.get('iteration') in {1, 2}]
    if len(concurrent_rows) == 2:
        winner = next(row for row in concurrent_rows if row.get('best_updated'))
        loser = next(row for row in concurrent_rows if not row.get('best_updated'))
        winner['iteration'] = 1
        loser['iteration'] = 2
        normalized.sort(key=lambda row: row.get('iteration', -1))
    return normalized


def _normalize_multi_worker_run_summary(payload: dict) -> dict:
    normalized = json.loads(json.dumps(payload))
    if normalized.get('best_iteration') in {1, 2}:
        normalized['best_iteration'] = 1
    return normalized


class TestPhase0MultiWorkerFixture(unittest.TestCase):
    def test_multi_worker_completed_fixture_matches_stubbed_run(self):
        fixture_root = Path('tests/fixtures/parity/multi_worker_completed/normalized_expected')
        with tempfile.TemporaryDirectory() as tmp:
            config = HypoEvolveConfig()
            config.search.iterations = 3
            config.output.base_dir = tmp
            config.workers.enabled = True
            config.workers.count = 2
            config.evaluator.dataset_schema_path = str(Path(tmp) / 'dataset.yaml')
            Path(config.evaluator.dataset_schema_path).write_text(
                "description: test\nindex:\n  name: close_time\n  dtype: datetime64[us]\nfiles:\n  -\n    entity: BTCUSDT\n    path: /tmp/BTCUSDT.parquet\ncolumns:\n  -\n    name: CLOSE\n",
                encoding='utf-8',
            )
            with patch('hypoevolve.controller.run_worker_task', side_effect=fake_run_worker_task):
                controller = HypoEvolveController(
                    config,
                    llm_client=FakeLLM(),
                    evaluator=type('FakeEvaluator', (), {'evaluate': lambda self, hypothesis: {'combined_score': 0.5}})(),
                    executor_factory=FakeExecutor,
                )
                result = controller.run('if A then B')

            run_dir = result.run_dir
            actual = {
                'best.json': json.loads((run_dir / 'best.json').read_text(encoding='utf-8')),
                'checkpoint.json': _normalize_multi_worker_checkpoint(
                    json.loads((run_dir / 'checkpoint.json').read_text(encoding='utf-8'))
                ),
                'run_summary.json': _normalize_multi_worker_run_summary(
                    json.loads((run_dir / 'run_summary.json').read_text(encoding='utf-8'))
                ),
                'score_history.json': _normalize_multi_worker_score_history(
                    json.loads((run_dir / 'score_history.json').read_text(encoding='utf-8'))
                ),
            }
            actual['run_summary.json']['dataset_schema_path'] = 'dataset.yaml'

        for filename, payload in actual.items():
            expected_raw = json.loads((fixture_root / filename).read_text(encoding='utf-8'))
            if filename == 'checkpoint.json':
                expected = _normalize_multi_worker_checkpoint(expected_raw)
            elif filename == 'run_summary.json':
                expected = _normalize_multi_worker_run_summary(expected_raw)
            elif filename == 'score_history.json':
                expected = _normalize_multi_worker_score_history(expected_raw)
            else:
                expected = expected_raw
            self.assertEqual(payload, expected, msg=f'mismatch for {filename}')


if __name__ == '__main__':
    unittest.main()
