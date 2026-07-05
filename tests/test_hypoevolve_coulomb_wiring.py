import random
import unittest
from unittest.mock import Mock, patch

from hypoevolve.core.config import (
    ArchiveConfig,
    ConfigError,
    CoulombArchiveConfig,
    HypoEvolveConfig,
    _config_from_dict,
)
from hypoevolve.core.orchestrator import (
    HypoEvolveController,
    _build_archive,
    _default_selection_policy,
)
from hypoevolve.memory.archive import MAPElitesArchive
from hypoevolve.memory.artifacts import _descriptor_payload
from hypoevolve.memory.coulomb_archive import CoulombArchive
from hypoevolve.policies.selection import CoulombSelectionPolicy, UCBSelectionPolicy
from hypoevolve.elg import (
    AtomicNode,
    Hypothesis,
    RelationNode,
    RelationType,
)


def _wrap_impl(cond_name: str, target_name: str) -> Hypothesis:
    return Hypothesis(
        root=RelationNode(
            name=RelationType.IMPLIES,
            inputs=[AtomicNode(name=cond_name), AtomicNode(name=target_name)],
        )
    )


class TestArchiveConfigDefaults(unittest.TestCase):
    def test_default_kind_is_map_elites(self):
        cfg = HypoEvolveConfig()
        self.assertEqual(cfg.archive.kind, "map_elites")
        self.assertIsInstance(cfg.archive.coulomb, CoulombArchiveConfig)

    def test_coulomb_defaults_are_reasonable(self):
        cfg = CoulombArchiveConfig()
        self.assertEqual(cfg.capacity, 64)
        self.assertAlmostEqual(cfg.gamma, 0.3)
        self.assertAlmostEqual(cfg.eps, 1e-2)


class TestConfigFromDict(unittest.TestCase):
    _EVAL = {"evaluator": {"dataset_schema_path": "dataset.yaml"}}

    def test_coulomb_kind_loads_with_overrides(self):
        cfg = _config_from_dict(
            {
                "archive": {
                    "kind": "coulomb",
                    "coulomb": {"capacity": 32, "gamma": 0.5, "eps": 0.05},
                },
                **self._EVAL,
            }
        )
        self.assertEqual(cfg.archive.kind, "coulomb")
        self.assertEqual(cfg.archive.coulomb.capacity, 32)
        self.assertAlmostEqual(cfg.archive.coulomb.gamma, 0.5)
        self.assertAlmostEqual(cfg.archive.coulomb.eps, 0.05)

    def test_unknown_kind_rejected(self):
        with self.assertRaisesRegex(ConfigError, "archive.kind"):
            _config_from_dict({"archive": {"kind": "bogus"}, **self._EVAL})

    def test_coulomb_capacity_zero_rejected(self):
        with self.assertRaisesRegex(ConfigError, "archive.coulomb.capacity"):
            _config_from_dict(
                {
                    "archive": {"kind": "coulomb", "coulomb": {"capacity": 0}},
                    **self._EVAL,
                }
            )

    def test_map_elites_still_validates_bins(self):
        # Backwards compatibility: map_elites path keeps its stricter validation.
        with self.assertRaisesRegex(ConfigError, "coverage_bins"):
            _config_from_dict(
                {"archive": {"coverage_bins": [0.5, 0.2, 0.7]}, **self._EVAL}
            )


class TestBuildArchive(unittest.TestCase):
    def test_default_returns_map_elites(self):
        cfg = HypoEvolveConfig()
        archive = _build_archive(cfg)
        self.assertIsInstance(archive, MAPElitesArchive)

    def test_coulomb_config_returns_coulomb_archive(self):
        cfg = HypoEvolveConfig()
        cfg.archive.kind = "coulomb"
        cfg.archive.coulomb.capacity = 8
        cfg.archive.coulomb.gamma = 0.4
        archive = _build_archive(cfg)
        self.assertIsInstance(archive, CoulombArchive)
        self.assertEqual(archive.capacity, 8)
        self.assertAlmostEqual(archive.gamma, 0.4)


class TestDefaultSelectionPolicy(unittest.TestCase):
    def test_map_elites_gets_ucb(self):
        cfg = HypoEvolveConfig()
        policy = _default_selection_policy(cfg)
        self.assertIsInstance(policy, UCBSelectionPolicy)

    def test_coulomb_gets_coulomb_policy(self):
        cfg = HypoEvolveConfig()
        cfg.archive.kind = "coulomb"
        policy = _default_selection_policy(cfg)
        self.assertIsInstance(policy, CoulombSelectionPolicy)


class TestCoulombSelectionPolicy(unittest.TestCase):
    def test_delegates_to_archive_sample_parent(self):
        archive = CoulombArchive(capacity=3, gamma=0.3)
        archive.add(_wrap_impl("a", "b"), {"combined_score": 0.9, "coverage": 0.2})
        archive.add(_wrap_impl("c", "d"), {"combined_score": 0.6, "coverage": 0.2})
        policy = CoulombSelectionPolicy()
        entry = policy.select(archive, random.Random(0))
        self.assertIn(entry.fingerprint, {row.fingerprint for row in archive.entries})

    def test_empty_archive_raises(self):
        policy = CoulombSelectionPolicy()
        with self.assertRaises(ValueError):
            policy.select(CoulombArchive(), random.Random(0))


class TestDescriptorPayloadShim(unittest.TestCase):
    """The artifacts writer needs a shim so both archive kinds emit the same field."""

    def test_map_elites_descriptor_payload(self):
        arc = MAPElitesArchive()
        h = _wrap_impl("a", "b")
        desc = arc.describe(h, {"coverage": 0.2, "combined_score": 0.7})
        payload = _descriptor_payload(desc)
        self.assertIn("coverage_bin", payload)
        self.assertIn("complexity_bin", payload)

    def test_coulomb_descriptor_payload(self):
        arc = CoulombArchive(capacity=4)
        h = _wrap_impl("a", "b")
        desc = arc.describe(h, {"coverage": 0.2, "combined_score": 0.7})
        payload = _descriptor_payload(desc)
        self.assertIn("potential", payload)
        self.assertIn("quality", payload)

    def test_empty_descriptor_is_safe(self):
        self.assertEqual(_descriptor_payload({}), {})


class _FakeCoulombLLM:
    """Minimal stub LLM that returns varied children so archive can grow."""

    def __init__(self):
        self._mutation_count = 0

    def generate_json(self, system, user, **kwargs):
        if "mutation_summary" in system:
            self._mutation_count += 1
            child_condition = f"A{self._mutation_count}"
            return {
                "child_hypothesis": {
                    "kind": "relation",
                    "name": "IMPLIES",
                    "inputs": [
                        {"kind": "atomic", "name": child_condition},
                        {"kind": "atomic", "name": "B"},
                    ],
                },
                "domain_reason": "Vary the condition atom to explore.",
                "score_reason": "Local replacement mutation.",
                "operation_score_rankings": {"replace_atomic_feature": 1},
                "mutation_summary": f"Replaced condition atom to {child_condition}.",
            }
        return {
            "kind": "relation",
            "name": "IMPLIES",
            "inputs": [
                {"kind": "atomic", "name": "A"},
                {"kind": "atomic", "name": "B"},
            ],
        }

    def generate_text(self, system, user, **kwargs):
        return "If A then B."


class TestControllerEndToEndWithCoulombArchive(unittest.TestCase):
    """Actually run the controller for a few iterations under archive.kind='coulomb'."""

    def test_iteration_loop_grows_coulomb_archive_and_produces_artifacts(self):
        import json
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            config = HypoEvolveConfig()
            config.archive.kind = "coulomb"
            config.archive.coulomb.capacity = 8
            config.archive.coulomb.gamma = 0.3
            config.search.iterations = 3
            config.search.random_steering_prob = 1.0  # random path avoids top-hypothesis dependencies
            config.output.base_dir = tmp
            config.evaluator.dataset_schema_path = str(Path(tmp) / "dataset.yaml")
            Path(config.evaluator.dataset_schema_path).write_text(
                "description: test\n"
                "index:\n  name: close_time\n  dtype: datetime64[us]\n"
                "files:\n  -\n    entity: BTCUSDT\n    path: /tmp/BTCUSDT.parquet\n"
                "columns:\n  -\n    name: CLOSE\n",
                encoding="utf-8",
            )

            score_iterator = iter([0.4, 0.5, 0.6, 0.7])

            def evaluate(self, hypothesis):
                return {"combined_score": next(score_iterator, 0.5), "coverage": 0.2}

            fake_evaluator = type("FakeEvaluator", (), {"evaluate": evaluate})()
            controller = HypoEvolveController(
                config, evaluator=fake_evaluator, llm_client=_FakeCoulombLLM()
            )
            result = controller.run("if A then B")
            run_dir_snapshot = result.run_dir
            trace_lines = (run_dir_snapshot / "trace.jsonl").read_text(encoding="utf-8").splitlines()
            checkpoint = json.loads((run_dir_snapshot / "checkpoint.json").read_text(encoding="utf-8"))

        self.assertIsNotNone(result.run_dir)
        self.assertIsNotNone(result.best_hypothesis)
        self.assertEqual(result.iterations, 3)
        # Coulomb archive should have grown across iterations (seed + up to 3 children).
        self.assertGreaterEqual(len(checkpoint["archive"]), 2)
        # Every archive entry under Coulomb should carry a coulomb descriptor and no cell.
        for entry in checkpoint["archive"]:
            self.assertIsNone(entry.get("cell"))
            self.assertIn("coulomb", entry["metadata"])
        # trace.jsonl should contain seed row + per-iteration rows.
        self.assertGreaterEqual(len(trace_lines), 1)


class TestControllerBootstrapWithCoulombArchive(unittest.TestCase):
    """End-to-end guard against ``descriptor['map_elites']`` KeyError under Coulomb."""

    def test_bootstrap_seed_runs_recorder_without_map_elites_key_error(self):
        config = HypoEvolveConfig()
        config.archive.kind = "coulomb"
        config.archive.coulomb.capacity = 4
        config.archive.coulomb.gamma = 0.3

        controller = HypoEvolveController(
            config,
            evaluator=type(
                "FakeEvaluator",
                (),
                {
                    "evaluate": lambda self, hypothesis: {
                        "combined_score": 0.5,
                        "coverage": 0.2,
                    },
                    "last_evaluation_artifacts": {"attempt": 1},
                },
            )(),
            llm_client=object(),
        )
        recorder = Mock()
        seed = Hypothesis(
            root=RelationNode(
                name=RelationType.IMPLIES,
                inputs=[AtomicNode(name="a"), AtomicNode(name="b")],
            )
        )
        with (
            patch(
                "hypoevolve.core.orchestrator.parse_hypothesis_text",
                return_value=seed,
            ),
            patch(
                "hypoevolve.core.orchestrator.llm_make_hypothesis_measurable",
                return_value=seed,
            ),
        ):
            seed_state = controller._bootstrap_seed(
                seed_input_text="if a then b",
                recorder=recorder,
            )

        self.assertIsInstance(seed_state.archive, CoulombArchive)
        self.assertEqual(len(seed_state.archive), 1)
        recorder.record_seed.assert_called_once()
        kwargs = recorder.record_seed.call_args.kwargs
        self.assertIn("descriptor", kwargs)
        # The Coulomb descriptor should provide a coulomb payload the recorder can use.
        self.assertIn("coulomb", kwargs["descriptor"])
        self.assertIsNone(kwargs["descriptor"]["cell"])


if __name__ == "__main__":
    unittest.main()
