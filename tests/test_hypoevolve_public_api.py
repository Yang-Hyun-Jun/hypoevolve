import re
import unittest
from pathlib import Path

import elg
import hypoevolve
import hypoevolve.elg as hypoevolve_elg
from hypoevolve.evaluator import LLMEvaluator
from hypoevolve.evaluator_contracts import Evaluator as EvaluatorContract
from hypoevolve.worker_contracts import (
    WorkerResult as WorkerResultContract,
    WorkerTask as WorkerTaskContract,
)
from hypoevolve.workers import run_worker_task


class TestHypoEvolvePublicAPI(unittest.TestCase):
    def test_package_root_all_matches_advertised_snapshot(self):
        self.assertEqual(
            hypoevolve.__all__,
            [
                "ArchiveEntry",
                "ColumnSpec",
                "ConfigError",
                "DataFile",
                "DatasetAccessor",
                "DatasetSchema",
                "DatasetSchemaError",
                "Evaluator",
                "ExecutionResult",
                "LLMEvaluator",
                "HypoEvolveConfig",
                "HypoEvolveController",
                "LLMClient",
                "LLMConfig",
                "MAPElitesArchive",
                "HypothesisGenerationError",
                "ParseError",
                "steer_mutation",
                "generate_random_tree_pair_hypothesis",
                "llm_hypothesis_to_natural_language",
                "llm_make_hypothesis_measurable",
                "parse_hypothesis_text",
                "run_worker_task",
            ],
        )

    def test_every_advertised_export_is_importable(self):
        for name in hypoevolve.__all__:
            with self.subTest(name=name):
                self.assertTrue(hasattr(hypoevolve, name))

    def test_internal_worker_payload_types_are_not_advertised_in_root_all(self):
        self.assertNotIn("WorkerTask", hypoevolve.__all__)
        self.assertNotIn("WorkerResult", hypoevolve.__all__)

    def test_legacy_elg_imports_remain_compatible_with_canonical_package(self):
        self.assertIs(elg.Hypothesis, hypoevolve_elg.Hypothesis)
        self.assertIs(elg.render_pretty, hypoevolve_elg.render_pretty)
        self.assertEqual(elg.__all__, hypoevolve_elg.__all__)

    def test_root_compatibility_attrs_still_point_at_contract_and_runtime_surfaces(self):
        self.assertIs(hypoevolve.Evaluator, EvaluatorContract)
        self.assertIs(hypoevolve.LLMEvaluator, LLMEvaluator)
        self.assertIs(hypoevolve.WorkerTask, WorkerTaskContract)
        self.assertIs(hypoevolve.WorkerResult, WorkerResultContract)
        self.assertIs(hypoevolve.run_worker_task, run_worker_task)

    def test_package_root_sources_boundary_types_from_contract_modules(self):
        init_path = Path(__file__).resolve().parents[1] / "hypoevolve" / "__init__.py"
        text = init_path.read_text(encoding="utf-8")

        self.assertIn("from .evaluator_contracts import Evaluator", text)
        self.assertIn("from .worker_contracts import WorkerResult, WorkerTask", text)
        self.assertNotIn("from .evaluator import Evaluator, LLMEvaluator", text)
        self.assertNotIn("from .workers import WorkerResult, WorkerTask, run_worker_task", text)

    def test_production_modules_do_not_import_package_root_or_cli(self):
        package_root = Path(__file__).resolve().parents[1] / "hypoevolve"
        disallowed_root_patterns = (
            re.compile(r"^\s*from\s+hypoevolve\s+import\b"),
            re.compile(r"^\s*import\s+hypoevolve\b"),
        )
        disallowed_cli_pattern = re.compile(r"^\s*from\s+hypoevolve\.cli\s+import\b|^\s*import\s+hypoevolve\.cli\b")

        for path in sorted(package_root.rglob("*.py")):
            if path.name in {"__init__.py", "cli.py"}:
                continue
            text = path.read_text(encoding="utf-8")
            for line in text.splitlines():
                with self.subTest(file=str(path.relative_to(package_root.parent)), line=line):
                    self.assertFalse(
                        any(pattern.search(line) for pattern in disallowed_root_patterns),
                        msg=f"{path} should not import hypoevolve package root",
                    )
                    self.assertFalse(
                        disallowed_cli_pattern.search(line),
                        msg=f"{path} should not import hypoevolve.cli",
                    )


if __name__ == "__main__":
    unittest.main()
