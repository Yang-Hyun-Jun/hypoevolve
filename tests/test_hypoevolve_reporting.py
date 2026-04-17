import json
import tempfile
import unittest
from pathlib import Path

from elg import AtomicNode, Hypothesis, LogicalNode, RelationNode
from hypoevolve.reporting import (
    _as_float,
    _build_archive_distribution_svg,
    _build_score_progression_svg,
    _build_seed_vs_best_metrics_svg,
    _build_markdown_report,
    _collect_atomic_names,
    _empty_svg,
    _expand_numeric_range,
    _format_metric,
    _format_short_number,
    _hypothesis_structure,
    _sanitize_pipe_text,
    _read_json,
    generate_run_report,
)
from hypoevolve.runtime import create_run_dir, write_best, write_checkpoint, write_run_summary, write_score_history


class TestHypoEvolveReporting(unittest.TestCase):
    def test_generate_run_report_creates_markdown_and_svg_assets(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = create_run_dir(tmp, run_id="run-report")
            hypothesis = Hypothesis(
                root=RelationNode(
                    "IMPLIES",
                    [LogicalNode("AND", [AtomicNode("A"), AtomicNode("B")]), AtomicNode("C")],
                )
            )
            write_best(run_dir, hypothesis, {"combined_score": 0.8, "precision": 0.7, "coverage": 0.3})
            write_checkpoint(
                run_dir,
                {
                    "iteration": 2,
                    "archive_size": 1,
                    "best_metrics": {"combined_score": 0.8},
                    "best_hypothesis": hypothesis.to_dict(),
                    "archive": [
                        {
                            "fingerprint": "best-fp",
                            "iteration": 2,
                            "coverage": 0.3,
                            "complexity": 2,
                            "hypothesis": hypothesis.to_dict(),
                            "metrics": {"combined_score": 0.8},
                            "metadata": {},
                        }
                    ],
                },
            )
            write_run_summary(
                run_dir,
                {
                    "seed_input_text": "if A and B then C",
                    "iterations_requested": 2,
                    "worker_count": 1,
                    "archive_size": 1,
                    "occupied_cells": 1,
                    "best_iteration": 2,
                    "best_fingerprint": "best-fp",
                    "duplicate_skips_total": 0,
                    "occupancy_summary": "cells=1 cov=[1,0,0,0] cmp=[1,0,0,0]",
                    "best_hypothesis_nl": "If A and B then C.",
                },
            )
            write_score_history(
                run_dir,
                [
                    {"iteration": 0, "score": 0.4, "best_score_after": 0.4, "best_updated": True, "fingerprint": "seed-fp", "hypothesis_nl": "Seed hypothesis", "mutation_summary": "seed initialization"},
                    {"iteration": 2, "score": 0.8, "best_score_after": 0.8, "best_updated": True, "fingerprint": "best-fp", "hypothesis_nl": "If A and B then C.", "mutation_summary": "replace A with B"},
                ],
            )

            artifacts = generate_run_report(run_dir)

            self.assertTrue(artifacts.markdown_path.exists())
            self.assertTrue(artifacts.score_plot_path.exists())
            self.assertTrue(artifacts.metrics_plot_path.exists())
            self.assertTrue(artifacts.archive_plot_path.exists())
            report_text = artifacts.markdown_path.read_text(encoding="utf-8")
            self.assertIn("# HypoEvolve Final Report", report_text)
            self.assertIn("## Best ELG", report_text)
            self.assertIn("If A and B then C.", report_text)


    def test_seed_vs_best_metrics_svg_contains_legend_and_metrics(self):
        svg = _build_seed_vs_best_metrics_svg(
            {"score": 0.2, "precision": 0.1, "baseline": 0.05, "coverage": 0.2, "uplift": 0.05},
            {"score": 0.8, "precision": 0.7, "baseline": 0.1, "coverage": 0.4, "uplift": 0.3},
            {"combined_score": 0.8, "precision": 0.7, "baseline": 0.1, "coverage": 0.4, "uplift": 0.3},
        )
        self.assertIn('Seed vs best metric profile', svg)
        self.assertIn('Seed', svg)
        self.assertIn('Best', svg)
        self.assertIn('precision', svg)

    def test_build_markdown_report_includes_artifact_checklist_and_top_candidates(self):
        hypothesis = Hypothesis(root=RelationNode('IMPLIES', [AtomicNode('A'), AtomicNode('B')]))
        markdown = _build_markdown_report(
            run_dir=Path('/tmp/run'),
            summary={
                'seed_input_text': 'if A then B',
                'iterations_requested': 2,
                'worker_count': 1,
                'archive_size': 1,
                'occupied_cells': 1,
                'best_iteration': 1,
                'best_fingerprint': 'best-fp',
                'duplicate_skips_total': 0,
                'occupancy_summary': 'cells=1 cov=[1,0,0,0] cmp=[1,0,0,0]',
            },
            best_hypothesis=hypothesis,
            best_metrics={'combined_score': 0.8, 'precision': 0.6, 'baseline': 0.2, 'coverage': 0.3, 'uplift': 0.4, 'support_count': 2, 'total_count': 4, 'rationale': 'good'},
            best_nl='If A then B.',
            seed_entry={'score': 0.1},
            archive_entries=[{
                'fingerprint': 'best-fp',
                'iteration': 1,
                'coverage': 0.3,
                'complexity': 2,
                'hypothesis': hypothesis.to_dict(),
                'metrics': {'combined_score': 0.8},
                'metadata': {},
            }],
            score_history=[
                {'iteration': 0, 'score': 0.1, 'best_score_after': 0.1, 'best_updated': True, 'fingerprint': 'seed', 'hypothesis_nl': 'Seed', 'mutation_summary': 'seed initialization'},
                {'iteration': 1, 'score': 0.8, 'best_score_after': 0.8, 'best_updated': True, 'fingerprint': 'best-fp', 'hypothesis_nl': 'If A then B.', 'mutation_summary': 'replace atomic'},
            ],
        )
        self.assertIn('## Artifact Checklist', markdown)
        self.assertIn('## Top Archive Candidates', markdown)
        self.assertIn('trace.jsonl', markdown)
        self.assertIn('If A then B.', markdown)

    def test_svg_builders_return_empty_state_when_input_missing(self):
        self.assertIn("No evaluated scores recorded", _build_score_progression_svg([]))
        self.assertIn("Archive snapshot is empty", _build_archive_distribution_svg([], {}))
        self.assertIn("Empty chart", _empty_svg(100, 50, "Nothing here"))

    def test_reporting_number_and_text_helpers_normalize_values(self):
        self.assertEqual(_as_float(True), 0.0)
        self.assertEqual(_as_float(1.5), 1.5)
        self.assertEqual(_expand_numeric_range(0.0, 0.0), (-1.0, 1.0))
        self.assertEqual(_format_metric(3), "`3`")
        self.assertEqual(_format_metric(1.234567), "`1.234567`")
        self.assertEqual(_format_short_number(1234.0), "1,234")
        self.assertEqual(_sanitize_pipe_text("A|B\nC"), "A\\|B C")

    def test_hypothesis_structure_and_atomic_collection_reflect_tree_shape(self):
        hypothesis = Hypothesis(
            root=RelationNode(
                "IMPLIES",
                [LogicalNode("AND", [AtomicNode("A"), AtomicNode("B")]), AtomicNode("C")],
            )
        )
        structure = _hypothesis_structure(hypothesis)
        atomics = _collect_atomic_names(hypothesis.root)

        self.assertEqual(structure["relation"], "IMPLIES")
        self.assertIn("AND", structure["condition"])
        self.assertIn("A", structure["condition"])
        self.assertIn("B", structure["condition"])
        self.assertEqual(structure["target"], "`C`")
        self.assertEqual(atomics, ["A", "B", "C"])


    def test_read_json_loads_utf8_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'payload.json'
            path.write_text(json.dumps({'hello': 'world'}), encoding='utf-8')
            self.assertEqual(_read_json(path), {'hello': 'world'})


if __name__ == "__main__":
    unittest.main()
