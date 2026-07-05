import re
import unittest
from pathlib import Path

import elg
import hypoevolve
import hypoevolve.elg as hypoevolve_elg
import hypoevolve.skills.seed_generation as hypoevolve_seedgen
from hypoevolve.skills.evaluation import LLMEvaluator
from hypoevolve.skills.evaluation import Evaluator as EvaluatorContract
from hypoevolve.runtime.worker import (
    WorkerResult as WorkerResultContract,
    WorkerTask as WorkerTaskContract,
)
from hypoevolve.runtime.worker import run_worker_task


class TestHypoEvolvePublicAPI(unittest.TestCase):
    def test_package_root_all_matches_advertised_snapshot(self):
        self.assertEqual(
            hypoevolve.__all__,
            [
                "ArchiveEntry",
                "ColumnSpec",
                "ConfigError",
                "CoulombArchive",
                "CoulombArchiveConfig",
                "CoulombDescriptor",
                "CoulombSelectionPolicy",
                "DataFile",
                "DatasetAccessor",
                "DatasetSchema",
                "DatasetSchemaError",
                "Evaluator",
                "ExecutionResult",
                "LLMEvaluator",
                "HypoEvolveConfig",
                "HypoEvolveController",
                "LAMBDA_NEG",
                "LAMBDA_WRAP",
                "LLMClient",
                "LLMConfig",
                "MAPElitesArchive",
                "HypothesisGenerationError",
                "ParseError",
                "atomic_sim",
                "steer_mutation",
                "tree_distance",
                "tree_kernel",
                "generate_random_tree_pair_hypothesis",
                "llm_hypothesis_to_natural_language",
                "llm_make_hypothesis_measurable",
                "parse_hypothesis_text",
                "run_worker_task",
                # Protocol and policy exports
                "HookBus",
                "SelectionPolicy",
                "StoppingPolicy",
                "ContextProvider",
                "SeedGenerationSkill",
                "CompileSkill",
                "MutationSkill",
                "EvaluationSkill",
                "ReportingSkill",
                "UCBSelectionPolicy",
                "IterationStoppingPolicy",
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
        self.assertIs(
            hypoevolve.generate_random_tree_pair_hypothesis,
            hypoevolve_seedgen.generate_random_tree_pair_hypothesis,
        )
        self.assertIs(hypoevolve.WorkerTask, WorkerTaskContract)
        self.assertIs(hypoevolve.WorkerResult, WorkerResultContract)
        self.assertIs(hypoevolve.run_worker_task, run_worker_task)

    def test_package_root_sources_boundary_types_from_canonical_modules(self):
        init_path = Path(__file__).resolve().parents[1] / "hypoevolve" / "__init__.py"
        text = init_path.read_text(encoding="utf-8")

        self.assertIn("from .skills.evaluation import Evaluator", text)
        self.assertIn("from .runtime.worker import WorkerResult, WorkerTask", text)

    def test_production_modules_do_not_import_package_root_or_cli(self):
        package_root = Path(__file__).resolve().parents[1] / "hypoevolve"
        disallowed_root_patterns = (
            re.compile(r"^\s*from\s+hypoevolve\s+import\b"),
            re.compile(r"^\s*import\s+hypoevolve\b"),
        )
        disallowed_cli_pattern = re.compile(r"^\s*from\s+hypoevolve\.cli\s+import\b|^\s*import\s+hypoevolve\.cli\b")

        for path in sorted(package_root.rglob("*.py")):
            if path.name == "__init__.py":
                continue
            # cli is now a package; skip all files under cli/
            if "cli" in path.relative_to(package_root).parts:
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
