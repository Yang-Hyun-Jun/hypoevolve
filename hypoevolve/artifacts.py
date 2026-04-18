"""Run artifact assembly and persistence helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

from elg import Hypothesis, fingerprint, render_pretty
from hypoevolve.archive import MAPElitesArchive
from hypoevolve.artifact_contracts import (
    build_checkpoint_payload,
    build_history_entry,
    build_run_summary_payload,
    build_trace_event,
)
from hypoevolve.reporting import generate_run_report
from hypoevolve.runtime import (
    write_artifact,
    write_best,
    write_checkpoint,
    write_run_summary,
    write_score_history,
    write_trace,
)


@dataclass(slots=True)
class RunArtifactRecorder:
    """Persist run artifacts while keeping controller orchestration compact."""

    run_dir: Path
    seed_input_text: str
    worker_count: int
    workers_enabled: bool
    dataset_schema_path: str
    score_history: list[Dict[str, object]] = field(default_factory=list)
    duplicate_skips_solo: int = 0
    duplicate_skips_worker: int = 0
    top_k_code_artifacts: int = 5
    evaluation_artifact_cache: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Ensure the run directory has the expected artifact layout."""
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "artifacts").mkdir(parents=True, exist_ok=True)

    def record_seed(
        self,
        *,
        archive: MAPElitesArchive,
        hypothesis: Hypothesis,
        metrics: Dict[str, object],
        metadata: Dict[str, object],
        descriptor: Dict[str, object],
        evaluation_artifacts: Dict[str, Any] | None = None,
    ) -> None:
        """Persist the seed candidate and initialize run history.

        Args:
            archive: The current archive after the seed has been inserted.
            hypothesis: The measurable seed hypothesis.
            metrics: The evaluated seed metrics.
            metadata: Seed metadata such as natural-language rendering.
            descriptor: The archive descriptor computed for the seed.

        Returns:
            None.
        """
        write_trace(
            self.run_dir,
            self._trace_event(
                0,
                None,
                hypothesis,
                metrics,
                {**metadata, "map_elites": descriptor["map_elites"]},
            ),
        )
        write_best(self.run_dir, archive.best.hypothesis, archive.best.metrics)
        write_checkpoint(self.run_dir, self._checkpoint_payload(archive, 0))
        self._cache_evaluation_artifacts(hypothesis, evaluation_artifacts or {})
        write_artifact(
            self.run_dir,
            "seed",
            {
                "input_text": self.seed_input_text,
                "hypothesis": hypothesis.to_dict(),
                "metrics": metrics,
            },
        )
        self.score_history.append(
            self._history_entry(
                iteration=0,
                hypothesis=hypothesis,
                metrics=metrics,
                best_score_after=archive.best.score if archive.best else 0.0,
                best_updated=True,
                status="seed",
                metadata=metadata,
                descriptor=descriptor,
            )
        )

    def record_duplicate_skip(
        self,
        *,
        iteration: int,
        parent_fingerprint: str,
        child_fingerprint: str,
        parent_score: float,
        best_score_after: float,
        worker_mode: bool,
    ) -> None:
        """Record one duplicate-skip event in run history.

        Args:
            iteration: The iteration where the duplicate was detected.
            parent_fingerprint: The sampled parent fingerprint.
            child_fingerprint: The skipped child fingerprint.
            parent_score: The sampled parent score.
            best_score_after: The best archive score after the skip.
            worker_mode: Whether the skip occurred in worker mode.

        Returns:
            None.
        """
        if worker_mode:
            self.duplicate_skips_worker += 1
        else:
            self.duplicate_skips_solo += 1
        self.score_history.append(
            {
                "iteration": iteration,
                "status": "skipped_duplicate",
                "parent_fingerprint": parent_fingerprint,
                "child_fingerprint": child_fingerprint,
                "parent_score": parent_score,
                "best_score_after": best_score_after,
                "best_updated": False,
                "worker_mode": worker_mode,
            }
        )

    def record_steering_skip(
        self,
        *,
        iteration: int,
        parent_fingerprint: str,
        parent_score: float,
        best_score_after: float,
        worker_mode: bool,
        error: str,
    ) -> None:
        """Record one steering-failure skip event in run history."""
        self.score_history.append(
            {
                "iteration": iteration,
                "status": "skipped_steering_error",
                "parent_fingerprint": parent_fingerprint,
                "parent_score": parent_score,
                "best_score_after": best_score_after,
                "best_updated": False,
                "worker_mode": worker_mode,
                "error": error,
            }
        )

    def record_iteration_result(
        self,
        *,
        archive: MAPElitesArchive,
        iteration: int,
        parent_hypothesis: Hypothesis,
        parent_fingerprint: str,
        child_hypothesis: Hypothesis,
        child_metrics: Dict[str, object],
        metadata: Dict[str, object],
        descriptor: Dict[str, object],
        best_updated: bool,
        evaluation_artifacts: Dict[str, Any] | None = None,
    ) -> None:
        """Persist one evaluated iteration result and append history.

        Args:
            archive: The archive after the child has been inserted.
            iteration: The completed iteration index.
            parent_hypothesis: The sampled parent hypothesis.
            parent_fingerprint: The parent fingerprint.
            child_hypothesis: The evaluated child hypothesis.
            child_metrics: The evaluated child metrics.
            metadata: Iteration metadata used for persistence and reporting.
            descriptor: The descriptor computed for the child.
            best_updated: Whether this child changed the global best score.

        Returns:
            None.
        """
        write_trace(
            self.run_dir,
            self._trace_event(
                iteration,
                parent_hypothesis,
                child_hypothesis,
                child_metrics,
                {**metadata, "map_elites": descriptor["map_elites"]},
            ),
        )
        write_checkpoint(self.run_dir, self._checkpoint_payload(archive, iteration))
        write_best(self.run_dir, archive.best.hypothesis, archive.best.metrics)
        self._cache_evaluation_artifacts(child_hypothesis, evaluation_artifacts or {})
        write_artifact(
            self.run_dir,
            f"iteration_{iteration:04d}",
            {
                "mutation_summary": metadata.get("mutation_summary", ""),
                "domain_reason": metadata.get("domain_reason", ""),
                "score_reason": metadata.get("score_reason", ""),
                "operation_score_rankings": metadata.get(
                    "operation_score_rankings", {}
                ),
                "metrics": child_metrics,
                "hypothesis": child_hypothesis.to_dict(),
                "worker_mode": metadata.get("worker_mode", False),
                "map_elites": descriptor["map_elites"],
            },
        )
        self.score_history.append(
            self._history_entry(
                iteration=iteration,
                hypothesis=child_hypothesis,
                metrics=child_metrics,
                best_score_after=archive.best.score if archive.best else 0.0,
                best_updated=best_updated,
                status="evaluated",
                metadata=metadata,
                descriptor=descriptor,
                parent_fingerprint=parent_fingerprint,
            )
        )

    def finalize(
        self,
        *,
        archive: MAPElitesArchive,
        iterations_requested: int,
        known_fingerprint_count: int,
        best_hypothesis_nl: str,
    ) -> Path:
        """Write final run summaries and generate the markdown report.

        Args:
            archive: The final archive state.
            iterations_requested: The configured number of search iterations.
            known_fingerprint_count: The number of unique fingerprints seen in the run.
            best_hypothesis_nl: Natural-language text for the final best hypothesis.

        Returns:
            Path: The generated markdown report path.
        """
        write_score_history(
            self.run_dir,
            sorted(self.score_history, key=lambda item: item["iteration"]),
        )
        write_run_summary(
            self.run_dir,
            build_run_summary_payload(
                archive=archive,
                seed_input_text=self.seed_input_text,
                iterations_requested=iterations_requested,
                worker_count=self.worker_count,
                workers_enabled=self.workers_enabled,
                dataset_schema_path=self.dataset_schema_path,
                duplicate_skips_solo=self.duplicate_skips_solo,
                duplicate_skips_worker=self.duplicate_skips_worker,
                known_fingerprint_count=known_fingerprint_count,
                best_hypothesis_nl=best_hypothesis_nl,
            ),
        )
        self._materialize_top_k_evaluator_artifacts(archive)
        return generate_run_report(self.run_dir).markdown_path

    def _cache_evaluation_artifacts(
        self,
        hypothesis: Hypothesis,
        evaluation_artifacts: Dict[str, Any],
    ) -> None:
        """Keep evaluator artifacts in memory so only top-k entries are materialized."""
        if not evaluation_artifacts:
            return
        self.evaluation_artifact_cache[fingerprint(hypothesis)] = dict(
            evaluation_artifacts
        )

    def _materialize_top_k_evaluator_artifacts(
        self,
        archive: MAPElitesArchive,
    ) -> None:
        """Copy evaluator code for the top-ranked archive entries into one folder."""
        top_entries = archive.entries[: self.top_k_code_artifacts]
        top_dir = self.run_dir / "artifacts" / "top_evaluators"
        top_dir.mkdir(parents=True, exist_ok=True)
        manifest: list[Dict[str, object]] = []

        for rank, entry in enumerate(top_entries, start=1):
            evaluation_artifacts = self.evaluation_artifact_cache.get(
                entry.fingerprint, {}
            )
            source_prefix = (
                "seed" if entry.iteration == 0 else f"iteration_{entry.iteration:04d}"
            )
            label = f"rank_{rank:02d}_{source_prefix}_{entry.fingerprint[:8]}"
            item: Dict[str, object] = {
                "rank": rank,
                "iteration": entry.iteration,
                "fingerprint": entry.fingerprint,
                "score": entry.score,
                "hypothesis": render_pretty(entry.hypothesis),
                "metrics": dict(entry.metrics),
            }

            candidate_code = evaluation_artifacts.get("candidate_code")
            if isinstance(candidate_code, str) and candidate_code.strip():
                dest_candidate = top_dir / f"{label}_candidate.py"
                dest_candidate.write_text(
                    candidate_code.rstrip() + "\n", encoding="utf-8"
                )
                item["candidate_code"] = str(dest_candidate.relative_to(self.run_dir))
            wrapper_code = evaluation_artifacts.get("wrapper_code")
            if isinstance(wrapper_code, str) and wrapper_code.strip():
                dest_wrapper = top_dir / f"{label}_wrapper.py"
                dest_wrapper.write_text(wrapper_code.rstrip() + "\n", encoding="utf-8")
                item["wrapper_code"] = str(dest_wrapper.relative_to(self.run_dir))
            metadata = {
                key: value
                for key, value in evaluation_artifacts.items()
                if key not in {"candidate_code", "wrapper_code"}
            }
            if metadata:
                dest_metadata = top_dir / f"{label}_evaluation.json"
                dest_metadata.write_text(
                    json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
                item["metadata"] = str(dest_metadata.relative_to(self.run_dir))

            manifest.append(item)

        manifest_path = self.run_dir / "artifacts" / "top_evaluators.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def _checkpoint_payload(
        self, archive: MAPElitesArchive, iteration: int
    ) -> Dict[str, object]:
        return build_checkpoint_payload(archive, iteration)

    def _trace_event(
        self,
        iteration: int,
        parent: Hypothesis | None,
        child: Hypothesis,
        metrics: Dict[str, object],
        metadata: Dict[str, object],
    ) -> Dict[str, object]:
        return build_trace_event(iteration, parent, child, metrics, metadata)

    def _history_entry(
        self,
        *,
        iteration: int,
        hypothesis: Hypothesis,
        metrics: Dict[str, object],
        best_score_after: float,
        best_updated: bool,
        status: str,
        metadata: Dict[str, object],
        descriptor: Dict[str, object],
        parent_fingerprint: str | None = None,
    ) -> Dict[str, object]:
        return build_history_entry(
            iteration=iteration,
            hypothesis=hypothesis,
            metrics=metrics,
            best_score_after=best_score_after,
            best_updated=best_updated,
            status=status,
            metadata=metadata,
            descriptor=descriptor,
            parent_fingerprint=parent_fingerprint,
        )
