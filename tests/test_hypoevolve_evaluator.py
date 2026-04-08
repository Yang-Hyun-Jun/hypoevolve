import json
import unittest

from elg import AtomicNode, Hypothesis, LogicalNode, RelationNode
from hypoevolve.archive import ArchiveEntry
from hypoevolve.config import LLMConfig
from hypoevolve.dataset import ColumnSpec, DataFile, DatasetAccessor, DatasetSchema, IndexSpec
from hypoevolve.evaluator import LLMEvaluator, evaluate_hypothesis, get_evaluation_artifacts
from hypoevolve.executor import ExecutionResult
from hypoevolve.helper import (
    build_evaluator_prompt_variables,
    build_evaluator_runtime_wrapper,
    build_steering_prompt_variables,
)


class FakeLLMClient:
    def __init__(self, outputs, retries: int = 1):
        self.outputs = list(outputs)
        self.config = LLMConfig(retries=retries)
        self.calls = []

    def generate_text(self, system: str, user: str, **kwargs):
        self.calls.append({"system": system, "user": user, "kwargs": kwargs})
        if not self.outputs:
            raise RuntimeError("No fake LLM outputs remaining")
        return self.outputs.pop(0)


class FakeExecutor:
    def __init__(self, results):
        self.results = list(results)
        self.calls = []

    def execute(self, code: str, files=None):
        self.calls.append({"code": code, "files": dict(files or {})})
        if not self.results:
            raise RuntimeError("No fake executor results remaining")
        return self.results.pop(0)


class TestHypoEvolveEvaluator(unittest.TestCase):
    def setUp(self):
        self.schema = DatasetSchema(
            files=[DataFile(entity="BTCUSDT", path="BTCUSDT.parquet")],
            index=IndexSpec(name="close_time", dtype="datetime64[us]"),
            columns=[ColumnSpec(name="CLOSE", description="close price")],
            description="Test dataset",
        )
        self.hypothesis = Hypothesis(
            root=RelationNode(
                "IMPLIES",
                [
                    LogicalNode("AND", [AtomicNode("A"), AtomicNode("B")]),
                    AtomicNode("C"),
                ],
            )
        )

    def test_helper_calls_evaluator(self):
        evaluator = type(
            "FakeEvaluator",
            (),
            {"evaluate": lambda self, hypothesis: {"combined_score": 0.5}},
        )()
        metrics = evaluate_hypothesis(Hypothesis(root=AtomicNode("A")), evaluator)
        self.assertIn("combined_score", metrics)

    def test_build_evaluator_prompt_variables(self):
        class FakeAccessor(DatasetAccessor):
            def load_dataframe(self, entity: str):
                raise AssertionError("build_evaluator_prompt_variables should not read sample data")

            def head(self, entity: str, n: int = 5):
                raise AssertionError("build_evaluator_prompt_variables should not read sample data")

        accessor = FakeAccessor(self.schema)
        variables = build_evaluator_prompt_variables(self.hypothesis, self.schema, accessor)
        self.assertIn("IMPLIES", variables["HYPOTHESIS_PRETTY"])
        self.assertEqual(variables["INDEX_NAME"], "close_time")
        self.assertEqual(variables["INDEX_DTYPE"], "datetime64[us]")
        self.assertIn("BTCUSDT", variables["ENTITIES"])
        self.assertIn("CLOSE", variables["COLUMN_SPECS"])
        self.assertIn("DatasetAccessor methods:", variables["DATASET_ACCESSOR_DOC"])
        self.assertNotIn("HYPOTHESIS_JSON", variables)
        self.assertNotIn("DATASET_SAMPLES", variables)

    def test_build_evaluator_runtime_wrapper(self):
        wrapper = build_evaluator_runtime_wrapper(
            "dataset.yaml",
            {"window": 10},
        )
        self.assertIn("sys.path.insert(0,", wrapper)
        self.assertIn("load_dataset_schema", wrapper)
        self.assertIn("redirect_stdout", wrapper)
        self.assertIn('"window": 10', wrapper)

    def test_build_steering_prompt_variables(self):
        top_hypotheses = [
            ArchiveEntry(
                hypothesis=Hypothesis(root=AtomicNode("BEST")),
                metrics={"combined_score": 0.9},
                fingerprint="best-fp",
                iteration=3,
                metadata={"source": "test"},
            )
        ]
        variables = build_steering_prompt_variables(
            parent_hypothesis=self.hypothesis,
            current_metrics={"combined_score": 0.1, "precision": 0.2},
            recent_history=[{"mutation_summary": "Applied a wrap_not-style local mutation.", "score_delta": -0.1}],
            top_hypotheses=top_hypotheses,
        )
        self.assertIn("IMPLIES", variables["PARENT_HYPOTHESIS_MEASURABLE"])
        self.assertIn("combined_score", variables["CURRENT_METRICS"])
        self.assertIn("precision = P(target | condition)", variables["METRIC_DEFINITIONS"])
        self.assertIn("wrap_not-style", variables["RECENT_HISTORY"])
        self.assertIn("BEST", variables["TOP_HYPOTHESES"])
        self.assertNotIn("MUTATION_CANDIDATES", variables)

    def test_llm_evaluator_returns_normalized_metrics(self):
        llm = FakeLLMClient(
            outputs=[
                "def evaluate_hypothesis(accessor: DatasetAccessor, parameters: dict[str, object] | None = None) -> dict[str, object]:\n"
                "    return {'combined_score': 0.5, 'precision': 0.7, 'baseline': 0.2, 'coverage': 0.4, 'uplift': 0.5, 'support_count': 2, 'total_count': 5, 'rationale': 'ok', 'used_parameters': {'RET_WINDOW': 12}}\n"
            ]
        )
        executor = FakeExecutor(
            results=[
                ExecutionResult(
                    stdout=json.dumps(
                        {
                            "combined_score": 0.5,
                            "precision": 0.7,
                            "baseline": 0.2,
                            "coverage": 0.4,
                            "uplift": 0.5,
                            "support_count": 2,
                            "total_count": 5,
                            "rationale": "ok",
                            "used_parameters": {"RET_WINDOW": 12},
                        }
                    ),
                    stderr="",
                    exit_code=0,
                    timed_out=False,
                    duration_sec=0.01,
                    work_dir="/tmp/fake",
                )
            ]
        )

        evaluator = LLMEvaluator(
            llm_client=llm,
            dataset_schema=self.schema,
            dataset_schema_path="dataset.yaml",
            executor=executor,
        )
        metrics = evaluator.evaluate(self.hypothesis)
        self.assertEqual(metrics["combined_score"], 0.5)
        self.assertEqual(metrics["support_count"], 2)
        self.assertEqual(metrics["rationale"], "ok")
        self.assertEqual(metrics["used_parameters"]["RET_WINDOW"], 12)
        self.assertIn("candidate.py", executor.calls[0]["files"])
        artifacts = get_evaluation_artifacts(evaluator)
        self.assertIn("candidate_code", artifacts)
        self.assertIn("wrapper_code", artifacts)
        self.assertEqual(artifacts["work_dir"], "/tmp/fake")

    def test_llm_evaluator_retries_after_invalid_code(self):
        llm = FakeLLMClient(
            outputs=[
                "def broken(",
                "def evaluate_hypothesis(accessor: DatasetAccessor, parameters: dict[str, object] | None = None) -> dict[str, object]:\n"
                "    return {'combined_score': 0.1, 'precision': 0.2, 'baseline': 0.1, 'coverage': 0.5, 'uplift': 0.1, 'support_count': 1, 'total_count': 2, 'rationale': 'fixed', 'used_parameters': {'HORIZON': 1}}\n",
            ],
            retries=1,
        )
        executor = FakeExecutor(
            results=[
                ExecutionResult(
                    stdout=json.dumps(
                        {
                            "combined_score": 0.1,
                            "precision": 0.2,
                            "baseline": 0.1,
                            "coverage": 0.5,
                            "uplift": 0.1,
                            "support_count": 1,
                            "total_count": 2,
                            "rationale": "fixed",
                            "used_parameters": {"HORIZON": 1},
                        }
                    ),
                    stderr="",
                    exit_code=0,
                    timed_out=False,
                    duration_sec=0.01,
                    work_dir="/tmp/fake",
                )
            ]
        )

        evaluator = LLMEvaluator(llm, self.schema, "dataset.yaml", executor=executor)
        metrics = evaluator.evaluate(self.hypothesis)
        self.assertEqual(metrics["rationale"], "fixed")
        self.assertEqual(metrics["used_parameters"]["HORIZON"], 1)
        self.assertEqual(len(llm.calls), 2)
        self.assertIn("Previous Attempt Failed", llm.calls[1]["user"])
        self.assertIn("Failure Message", llm.calls[1]["user"])
        self.assertIn("Previous Candidate Code", llm.calls[1]["user"])
        self.assertIn("def broken(", llm.calls[1]["user"])

    def test_llm_evaluator_retries_after_runtime_keyerror_with_previous_code(self):
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
                    stdout="",
                    stderr="Traceback (most recent call last):\n  File '/tmp/candidate.py', line 3, in evaluate_hypothesis\nKeyError: 'UNKNOWN_COL'",
                    exit_code=1,
                    timed_out=False,
                    duration_sec=0.01,
                    work_dir="/tmp/fake",
                ),
                ExecutionResult(
                    stdout=json.dumps(
                        {
                            "combined_score": 0.1,
                            "precision": 0.2,
                            "baseline": 0.1,
                            "coverage": 0.5,
                            "uplift": 0.1,
                            "support_count": 1,
                            "total_count": 2,
                            "rationale": "fixed",
                            "used_parameters": {"HORIZON": 1},
                        }
                    ),
                    stderr="",
                    exit_code=0,
                    timed_out=False,
                    duration_sec=0.01,
                    work_dir="/tmp/fake",
                )
            ]
        )

        evaluator = LLMEvaluator(llm, self.schema, "dataset.yaml", executor=executor)
        metrics = evaluator.evaluate(self.hypothesis)

        self.assertEqual(metrics["rationale"], "fixed")
        self.assertEqual(len(llm.calls), 2)
        self.assertIn("KeyError: 'UNKNOWN_COL'", llm.calls[1]["user"])
        self.assertIn("UNKNOWN_COL", llm.calls[1]["user"])
        self.assertIn("Do not reuse any dataframe column name unless it exactly matches", llm.calls[1]["user"])

    def test_llm_evaluator_returns_failure_metrics_after_exhausted_retries(self):
        llm = FakeLLMClient(outputs=["def broken(", "def still_broken("], retries=1)
        executor = FakeExecutor(results=[])
        evaluator = LLMEvaluator(llm, self.schema, "dataset.yaml", executor=executor)
        metrics = evaluator.evaluate(self.hypothesis)
        self.assertEqual(metrics["combined_score"], 0.0)
        self.assertIn("evaluation_failed:", metrics["rationale"])
        self.assertEqual(metrics["used_parameters"], {})

    def test_llm_evaluator_sanitizes_non_finite_metrics(self):
        llm = FakeLLMClient(
            outputs=[
                "def evaluate_hypothesis(accessor: DatasetAccessor, parameters: dict[str, object] | None = None) -> dict[str, object]:\n"
                "    return {'combined_score': float('nan'), 'precision': float('inf'), 'baseline': 0.2, 'coverage': 0.4, 'uplift': float('-inf'), 'support_count': float('nan'), 'total_count': 5, 'rationale': 'raw'}\n"
            ]
        )
        executor = FakeExecutor(
            results=[
                ExecutionResult(
                    stdout='{"combined_score": NaN, "precision": Infinity, "baseline": 0.2, "coverage": 0.4, "uplift": -Infinity, "support_count": NaN, "total_count": 5, "rationale": "raw", "used_parameters": {"RET_WINDOW": 12}}',
                    stderr="",
                    exit_code=0,
                    timed_out=False,
                    duration_sec=0.01,
                    work_dir="/tmp/fake",
                )
            ]
        )
        evaluator = LLMEvaluator(llm, self.schema, "dataset.yaml", executor=executor)
        metrics = evaluator.evaluate(self.hypothesis)
        self.assertEqual(metrics["combined_score"], 0.0)
        self.assertEqual(metrics["precision"], 0.0)
        self.assertEqual(metrics["uplift"], 0.0)
        self.assertEqual(metrics["support_count"], 0)
        self.assertEqual(metrics["used_parameters"]["RET_WINDOW"], 12)
        self.assertTrue(str(metrics["rationale"]).startswith("non_finite_metrics_sanitized"))

    def test_llm_evaluator_fills_missing_required_fields(self):
        llm = FakeLLMClient(
            outputs=[
                "def evaluate_hypothesis(accessor: DatasetAccessor, parameters: dict[str, object] | None = None) -> dict[str, object]:\n"
                "    return {'combined_score': 0.5, 'precision': 0.7, 'coverage': 0.4}\n"
            ]
        )
        executor = FakeExecutor(
            results=[
                ExecutionResult(
                    stdout='{"combined_score": 0.5, "precision": 0.7, "coverage": 0.4}',
                    stderr="",
                    exit_code=0,
                    timed_out=False,
                    duration_sec=0.01,
                    work_dir="/tmp/fake",
                )
            ]
        )
        evaluator = LLMEvaluator(llm, self.schema, "dataset.yaml", executor=executor)
        metrics = evaluator.evaluate(self.hypothesis)
        self.assertEqual(metrics["combined_score"], 0.5)
        self.assertEqual(metrics["precision"], 0.7)
        self.assertEqual(metrics["coverage"], 0.4)
        self.assertEqual(metrics["baseline"], 0.0)
        self.assertEqual(metrics["uplift"], 0.0)
        self.assertEqual(metrics["support_count"], 0)
        self.assertEqual(metrics["total_count"], 0)
        self.assertEqual(metrics["rationale"], "")
        self.assertEqual(metrics["used_parameters"], {})


if __name__ == "__main__":
    unittest.main()
