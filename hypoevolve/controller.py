"""Top-level orchestration for hypothesis evolution runs."""

from __future__ import annotations

import random
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from elg import Hypothesis, fingerprint, hypothesis_from_dict, render_pretty
from hypoevolve.archive import ArchiveEntry, MAPElitesArchive
from hypoevolve.artifacts import RunArtifactRecorder
from hypoevolve.config import HypoEvolveConfig
from hypoevolve.dataset import load_dataset_schema
from hypoevolve.evaluator import (
    Evaluator,
    LLMEvaluator,
    evaluate_hypothesis,
    get_evaluation_artifacts,
)
from hypoevolve.hypo import generate_random_tree_pair_hypothesis
from hypoevolve.llm import LLMClient
from hypoevolve.logger import (
    compact_text,
    configure_logger,
    log_error_event,
    log_info_event,
    summarize_exception,
    summarize_hypothesis,
    summarize_metrics,
)
from hypoevolve.mutation import steer_mutation
from hypoevolve.parser import (
    ParseError,
    llm_hypothesis_to_natural_language,
    llm_make_hypothesis_measurable,
    parse_hypothesis_text,
)
from hypoevolve.runtime import create_run_dir
from hypoevolve.workers import WorkerTask, run_worker_task


@dataclass(slots=True)
class RunResult:
    """Summarize the final outcome of one evolution run."""

    run_dir: Path
    seed_hypothesis: Hypothesis
    best_hypothesis: Hypothesis
    best_metrics: Dict[str, object]
    iterations: int
    report_path: Path
    seed_input_text: str
    seed_generated: bool


class HypoEvolveController:
    """Coordinate parsing, evaluation, mutation steering, and persistence."""

    def __init__(
        self,
        config: HypoEvolveConfig,
        evaluator: Optional[Evaluator] = None,
        llm_client: Optional[LLMClient] = None,
        executor_factory: Optional[Callable[..., Any]] = None,
    ):
        """Initialize one run controller.

        Args:
            config: The active runtime configuration.
            evaluator: Optional evaluator override for tests or custom execution.
            llm_client: Optional shared LLM client override.
            executor_factory: Optional worker executor factory override.

        Returns:
            None.
        """
        self.config = config
        self.llm_client = llm_client or LLMClient(config.llm)
        self.evaluator = evaluator or self._build_evaluator()
        self.rng = random.Random(config.search.random_seed)
        self.executor_factory = executor_factory or ProcessPoolExecutor

    def run(self, hypothesis_text: str | None = None) -> RunResult:
        """Execute the full evolution loop for one natural-language seed."""
        seed_input_text, seed_generated = self._resolve_seed_input_text(hypothesis_text)
        run_dir = create_run_dir(self.config.output.base_dir)
        recorder = RunArtifactRecorder(
            run_dir=run_dir,
            seed_input_text=seed_input_text,
            worker_count=self.config.workers.count,
            workers_enabled=self.config.workers.enabled,
            dataset_schema_path=self.config.evaluator.dataset_schema_path,
            top_k_code_artifacts=self.config.output.top_k_evaluator_code_artifacts,
        )
        configure_logger(
            self.config.logging.level,
            run_dir / "hypoevolve.log",
        )
        log_info_event(
            "run.start",
            run=run_dir.name,
            iterations=self.config.search.iterations,
            workers=self.config.workers.count,
            ds=self.config.evaluator.dataset_schema_path,
        )
        hypothesis = parse_hypothesis_text(
            seed_input_text,
            llm=self.llm_client,
            retries=self.config.parser.retries,
        )
        log_info_event("seed.parse", **summarize_hypothesis(hypothesis))
        hypothesis = llm_make_hypothesis_measurable(
            hypothesis,
            llm=self.llm_client,
            retries=self.config.parser.retries,
        )
        log_info_event("seed.measurable", **summarize_hypothesis(hypothesis))
        seed_metadata = {"source": "seed"}

        archive = MAPElitesArchive(
            coverage_bins=self.config.archive.coverage_bins,
            complexity_bins=self.config.archive.complexity_bins,
            per_cell_top_k=self.config.archive.per_cell_top_k,
            parent_sampling_mode=self.config.archive.parent_sampling_mode,
        )
        seed_metrics = evaluate_hypothesis(hypothesis, self.evaluator)
        seed_evaluation_artifacts = get_evaluation_artifacts(self.evaluator)
        log_info_event(
            "seed.eval",
            **summarize_hypothesis(hypothesis),
            **summarize_metrics(seed_metrics),
        )
        seed_descriptor = archive.describe(hypothesis, seed_metrics)
        archive.add(hypothesis, seed_metrics, iteration=0, metadata=seed_metadata)
        known_fingerprints = {fingerprint(hypothesis)}
        known_fingerprint_count = len(known_fingerprints)
        log_info_event(
            "seed.archive",
            archive_size=len(archive),
            best_score=archive.best.score if archive.best else 0.0,
            best_cell=archive.best.cell if archive.best else None,
            occupancy=archive.occupancy_summary(),
        )
        recorder.record_seed(
            archive=archive,
            hypothesis=hypothesis,
            metrics=seed_metrics,
            metadata=seed_metadata,
            descriptor=seed_descriptor,
            evaluation_artifacts=seed_evaluation_artifacts,
        )

        if not self.config.workers.enabled or self.config.workers.count == 1:
            recent_history: list[Dict[str, object]] = []
            for iteration in range(1, self.config.search.iterations + 1):
                parent_entry = archive.sample_parent(self.rng)
                parent_summary = summarize_hypothesis(parent_entry.hypothesis)
                parent_summary.pop("fp", None)
                log_info_event(
                    "iter.parent",
                    i=iteration,
                    parent_score=parent_entry.score,
                    parent_fp=parent_entry.fingerprint[:12],
                    parent_cell=parent_entry.cell,
                    **parent_summary,
                )
                try:
                    mutation_sample, steering_metadata = self._choose_mutation(
                        parent_entry,
                        recent_history,
                        archive,
                    )
                except ParseError as exc:
                    log_error_event(
                        "iter.skip_steering_error",
                        i=iteration,
                        parent_fp=parent_entry.fingerprint[:12],
                        **summarize_exception(exc),
                    )
                    archive.record_parent_outcome(parent_entry.fingerprint, 0.0)
                    recorder.record_steering_skip(
                        iteration=iteration,
                        parent_fingerprint=parent_entry.fingerprint,
                        parent_score=parent_entry.score,
                        best_score_after=archive.best.score if archive.best else 0.0,
                        worker_mode=False,
                        error=str(exc),
                    )
                    continue
                log_info_event(
                    "iter.steer",
                    i=iteration,
                    summary=compact_text(
                        steering_metadata.get("mutation_summary", ""), max_len=96
                    ),
                    random=steering_metadata.get("random_steering"),
                )
                child_fingerprint = fingerprint(mutation_sample)
                if child_fingerprint in known_fingerprints:
                    log_info_event(
                        "iter.skip_duplicate",
                        i=iteration,
                        child_fp=child_fingerprint[:12],
                    )
                    archive.record_parent_outcome(parent_entry.fingerprint, 0.0)
                    recorder.record_duplicate_skip(
                        iteration=iteration,
                        parent_fingerprint=parent_entry.fingerprint,
                        child_fingerprint=child_fingerprint,
                        parent_score=parent_entry.score,
                        best_score_after=archive.best.score if archive.best else 0.0,
                        worker_mode=False,
                    )
                    continue
                child_metrics = evaluate_hypothesis(mutation_sample, self.evaluator)
                child_evaluation_artifacts = get_evaluation_artifacts(self.evaluator)
                log_info_event(
                    "iter.eval",
                    i=iteration,
                    child_fp=child_fingerprint[:12],
                    **summarize_metrics(child_metrics),
                )
                score_delta = (
                    float(child_metrics.get("combined_score", 0.0)) - parent_entry.score
                )
                descriptor, best_updated = self._record_archive_result(
                    archive,
                    iteration,
                    mutation_sample,
                    child_metrics,
                    {
                        "parent_score": parent_entry.score,
                        "score_delta": score_delta,
                        **steering_metadata,
                    },
                )
                recorder.record_iteration_result(
                    archive=archive,
                    iteration=iteration,
                    parent_hypothesis=parent_entry.hypothesis,
                    parent_fingerprint=parent_entry.fingerprint,
                    child_hypothesis=mutation_sample,
                    child_metrics=child_metrics,
                    metadata={
                        "parent_score": parent_entry.score,
                        "score_delta": score_delta,
                        **steering_metadata,
                    },
                    descriptor=descriptor,
                    best_updated=best_updated,
                    evaluation_artifacts=child_evaluation_artifacts,
                )
                known_fingerprints.add(child_fingerprint)
                archive.record_parent_outcome(parent_entry.fingerprint, score_delta)
                recent_history.append(
                    {
                        "score_delta": score_delta,
                        "result_hypothesis": render_pretty(mutation_sample),
                        **steering_metadata,
                    }
                )
        else:
            log_info_event("run.workers", workers=self.config.workers.count, mode="parallel")
            _, known_fingerprint_count = self._run_with_workers(
                archive,
                self.config.search.iterations,
                recorder=recorder,
            )

        best = archive.best
        report_path = recorder.finalize(
            archive=archive,
            iterations_requested=self.config.search.iterations,
            known_fingerprint_count=known_fingerprint_count,
            best_hypothesis_nl=self._render_hypothesis_nl(best.hypothesis) if best else "",
        )
        log_info_event(
            "run.duplicate_summary",
            total_skips=recorder.duplicate_skips_solo + recorder.duplicate_skips_worker,
            solo_skips=recorder.duplicate_skips_solo,
            worker_skips=recorder.duplicate_skips_worker,
            known_fps=known_fingerprint_count,
        )
        log_info_event(
            "run.done",
            run=run_dir.name,
            best_score=best.score if best else 0.0,
            archive_size=len(archive),
            occupancy=archive.occupancy_summary(),
            best_fp=best.fingerprint[:12] if best else None,
        )
        return RunResult(
            run_dir=run_dir,
            seed_hypothesis=hypothesis,
            best_hypothesis=best.hypothesis,
            best_metrics=best.metrics,
            iterations=self.config.search.iterations,
            report_path=report_path,
            seed_input_text=seed_input_text,
            seed_generated=seed_generated,
        )

    def _resolve_seed_input_text(self, hypothesis_text: str | None) -> tuple[str, bool]:
        """Return the explicit seed text or synthesize one when absent."""
        if hypothesis_text and hypothesis_text.strip():
            return hypothesis_text.strip(), False
        generated = generate_random_tree_pair_hypothesis(
            llm=self.llm_client,
            dataset_schema_path=self.config.evaluator.dataset_schema_path,
        )
        log_info_event(
            "seed.generate",
            chars=len(generated.hypothesis),
            preview=compact_text(generated.hypothesis, max_len=96),
        )
        return generated.hypothesis, True

    def _run_with_workers(
        self,
        archive: MAPElitesArchive,
        total_iterations: int,
        recorder: RunArtifactRecorder,
    ) -> tuple[int, int]:
        """Execute the mutation loop with local worker processes.

        Args:
            archive: The mutable archive shared by the leader process.
            total_iterations: The total number of search iterations to schedule.
            recorder: The artifact recorder for persistence and report inputs.

        Returns:
            tuple[int, int]: Worker duplicate skips and known fingerprint count.
        """
        worker_count = self.config.workers.count
        pending: dict[object, ArchiveEntry] = {}
        submitted = 0
        recent_history: list[Dict[str, object]] = []
        known_fingerprints = {entry.fingerprint for entry in archive.entries}

        with self.executor_factory(max_workers=worker_count) as executor:
            while submitted < min(worker_count, total_iterations):
                submitted += 1
                task, parent_entry = self._make_worker_task(
                    archive, submitted, recent_history[-2:]
                )
                log_info_event(
                    "worker.submit",
                    i=submitted,
                    parent_score=archive.best.score if archive.best else 0.0,
                    parent_cell=parent_entry.cell,
                    parent_fp=parent_entry.fingerprint[:12],
                )
                future = executor.submit(run_worker_task, task)
                pending[future] = parent_entry

            while pending:
                done, _ = wait(tuple(pending), return_when=FIRST_COMPLETED)

                for future in done:
                    parent_entry = pending.pop(future)
                    result = future.result()
                    if result.skipped_steering_error:
                        log_error_event(
                            "worker.skip_steering_error",
                            i=result.iteration,
                            **summarize_exception(result.steering_error),
                        )
                        archive.record_parent_outcome(parent_entry.fingerprint, 0.0)
                        recorder.record_steering_skip(
                            iteration=result.iteration,
                            parent_fingerprint=parent_entry.fingerprint,
                            parent_score=parent_entry.score,
                            best_score_after=archive.best.score if archive.best else 0.0,
                            worker_mode=True,
                            error=result.steering_error,
                        )
                        if submitted < total_iterations:
                            submitted += 1
                            task, next_parent_entry = self._make_worker_task(
                                archive,
                                submitted,
                                recent_history[-2:],
                                known_fingerprints=known_fingerprints,
                            )
                            log_info_event(
                                "worker.submit",
                                i=submitted,
                                parent_score=archive.best.score if archive.best else 0.0,
                                parent_cell=next_parent_entry.cell,
                                parent_fp=next_parent_entry.fingerprint[:12],
                            )
                            next_future = executor.submit(run_worker_task, task)
                            pending[next_future] = next_parent_entry
                        continue
                    if result.skipped_duplicate:
                        log_info_event(
                            "worker.skip_duplicate",
                            i=result.iteration,
                            child_fp=result.child_fingerprint[:12],
                        )
                        archive.record_parent_outcome(parent_entry.fingerprint, 0.0)
                        recorder.record_duplicate_skip(
                            iteration=result.iteration,
                            parent_fingerprint=parent_entry.fingerprint,
                            child_fingerprint=result.child_fingerprint,
                            parent_score=parent_entry.score,
                            best_score_after=archive.best.score if archive.best else 0.0,
                            worker_mode=True,
                        )
                        if submitted < total_iterations:
                            submitted += 1
                            task, next_parent_entry = self._make_worker_task(
                                archive,
                                submitted,
                                recent_history[-2:],
                                known_fingerprints=known_fingerprints,
                            )
                            log_info_event(
                                "worker.submit",
                                i=submitted,
                                parent_score=archive.best.score if archive.best else 0.0,
                                parent_cell=next_parent_entry.cell,
                                parent_fp=next_parent_entry.fingerprint[:12],
                            )
                            next_future = executor.submit(run_worker_task, task)
                            pending[next_future] = next_parent_entry
                        continue
                    child = hypothesis_from_dict(result.child_hypothesis)
                    log_info_event(
                        "worker.result",
                        i=result.iteration,
                        child_fp=(result.child_fingerprint[:12] if result.child_fingerprint else None),
                        summary=compact_text(result.mutation_summary, max_len=96),
                        **summarize_metrics(result.metrics),
                    )
                    descriptor, best_updated = self._record_archive_result(
                        archive,
                        result.iteration,
                        child,
                        result.metrics,
                        {
                            "parent_score": result.parent_score,
                            "score_delta": float(result.metrics.get("combined_score", 0.0))
                            - result.parent_score,
                            "worker_mode": True,
                            "steered": True,
                            "mutation_summary": result.mutation_summary,
                            "domain_reason": result.domain_reason,
                            "score_reason": result.score_reason,
                            "operation_score_rankings": result.operation_score_rankings,
                            "random_steering": result.random_steering,
                        },
                    )
                    recorder.record_iteration_result(
                        archive=archive,
                        iteration=result.iteration,
                        parent_hypothesis=parent_entry.hypothesis,
                        parent_fingerprint=parent_entry.fingerprint,
                        child_hypothesis=child,
                        child_metrics=result.metrics,
                        metadata={
                            "parent_score": result.parent_score,
                            "score_delta": float(result.metrics.get("combined_score", 0.0))
                            - result.parent_score,
                            "worker_mode": True,
                            "steered": True,
                            "mutation_summary": result.mutation_summary,
                            "domain_reason": result.domain_reason,
                            "score_reason": result.score_reason,
                            "operation_score_rankings": result.operation_score_rankings,
                            "random_steering": result.random_steering,
                        },
                        descriptor=descriptor,
                        best_updated=best_updated,
                        evaluation_artifacts=result.evaluation_artifacts,
                    )
                    if result.child_fingerprint:
                        known_fingerprints.add(result.child_fingerprint)
                    archive.record_parent_outcome(
                        parent_entry.fingerprint,
                        float(result.metrics.get("combined_score", 0.0))
                        - result.parent_score,
                    )
                    recent_history.append(
                        {
                            "score_delta": float(result.metrics.get("combined_score", 0.0))
                            - result.parent_score,
                            "result_hypothesis": render_pretty(child),
                            "steered": True,
                            "mutation_summary": result.mutation_summary,
                            "domain_reason": result.domain_reason,
                            "score_reason": result.score_reason,
                            "operation_score_rankings": result.operation_score_rankings,
                            "random_steering": result.random_steering,
                        }
                    )
                    if submitted < total_iterations:
                        submitted += 1
                        task, next_parent_entry = self._make_worker_task(
                            archive,
                            submitted,
                            recent_history[-2:],
                            known_fingerprints=known_fingerprints,
                        )
                        log_info_event(
                            "worker.submit",
                            i=submitted,
                            parent_score=archive.best.score if archive.best else 0.0,
                            parent_cell=next_parent_entry.cell,
                            parent_fp=next_parent_entry.fingerprint[:12],
                        )
                        next_future = executor.submit(run_worker_task, task)
                        pending[next_future] = next_parent_entry
        return recorder.duplicate_skips_worker, len(known_fingerprints)

    def _make_worker_task(
        self,
        archive: MAPElitesArchive,
        iteration: int,
        recent_history: list[Dict[str, object]],
        known_fingerprints: set[str] | None = None,
    ) -> tuple[WorkerTask, ArchiveEntry]:
        """Build one worker task from the current archive state.

        Args:
            archive: The current archive used for parent sampling.
            iteration: The iteration assigned to the worker.
            recent_history: Recent mutation outcomes used for steering context.
            known_fingerprints: Optional known fingerprint set for duplicate avoidance.

        Returns:
            tuple[WorkerTask, ArchiveEntry]: The worker task and sampled parent entry.
        """
        parent_entry = archive.sample_parent(self.rng)
        task = WorkerTask(
            parent_hypothesis=parent_entry.hypothesis.to_dict(),
            parent_metrics=dict(parent_entry.metrics),
            iteration=iteration,
            parent_score=parent_entry.score,
            use_random_steering=(
                self.rng.random() < self.config.search.random_steering_prob
            ),
            llm_config=asdict(self.config.llm),
            dataset_schema_path=self.config.evaluator.dataset_schema_path,
            evaluator_parameters=dict(self.config.evaluator.parameters),
            parser_retries=self.config.parser.retries,
            steering_retries=self.config.search.steering_retries,
            recent_history=list(recent_history),
            top_hypotheses=archive.snapshot()[:3],
            seen_fingerprints=sorted(
                known_fingerprints
                if known_fingerprints is not None
                else {entry.fingerprint for entry in archive.entries}
            ),
        )
        return task, parent_entry

    def _build_evaluator(self) -> Evaluator:
        """Construct the default evaluator from the active config.

        Args:
            None.

        Returns:
            Evaluator: The configured LLM-backed evaluator instance.
        """
        schema = load_dataset_schema(self.config.evaluator.dataset_schema_path)
        return LLMEvaluator(
            llm_client=self.llm_client,
            dataset_schema=schema,
            dataset_schema_path=self.config.evaluator.dataset_schema_path,
            parameters=self.config.evaluator.parameters or None,
        )

    def _choose_mutation(
        self,
        parent_entry,
        recent_history: list[Dict[str, object]],
        archive: MAPElitesArchive,
    ) -> tuple[Any, Dict[str, object]]:
        """Choose one steered mutation from the current parent entry.

        Args:
            parent_entry: The sampled archive entry used as mutation parent.
            recent_history: Recent mutation outcomes for steering context.
            archive: The current archive used for top-hypothesis context.

        Returns:
            tuple[Any, dict[str, object]]: The mutated hypothesis and steering metadata.
        """
        use_random_steering = (
            self.rng.random() < self.config.search.random_steering_prob
        )
        decision = steer_mutation(
            parent_hypothesis=parent_entry.hypothesis,
            current_metrics=parent_entry.metrics,
            llm=self.llm_client,
            recent_history=recent_history[-2:],
            top_hypotheses=archive.entries[:3],
            use_random_steering=use_random_steering,
            retries=self.config.search.steering_retries,
        )
        return decision.child_hypothesis, {
            "steered": True,
            "domain_reason": decision.domain_reason,
            "score_reason": decision.score_reason,
            "operation_score_rankings": dict(
                getattr(decision, "operation_score_rankings", {})
            ),
            "mutation_summary": decision.mutation_summary,
            "random_steering": use_random_steering,
        }

    def _record_archive_result(
        self,
        archive: MAPElitesArchive,
        iteration: int,
        child_hypothesis: Hypothesis,
        child_metrics: Dict[str, object],
        metadata: Dict[str, object],
    ) -> tuple[Dict[str, object], bool]:
        """Insert one evaluated child into the archive and log the outcome.

        Args:
            archive: The mutable archive receiving the child.
            iteration: The iteration that produced the child.
            child_hypothesis: The evaluated child hypothesis.
            child_metrics: The evaluated child metrics.
            metadata: Metadata attached to the archive entry.

        Returns:
            tuple[dict[str, object], bool]: The descriptor and whether the best score changed.
        """
        previous_best_score = archive.best.score if archive.best else None
        descriptor = archive.describe(child_hypothesis, child_metrics)
        archive.add(
            child_hypothesis,
            child_metrics,
            iteration=iteration,
            metadata=metadata,
        )
        log_info_event(
            "archive.add",
            i=iteration,
            archive_size=len(archive),
            child_cell=descriptor["cell"],
            occupancy=archive.occupancy_summary(),
            **summarize_metrics(child_metrics),
        )
        best = archive.best
        best_updated = previous_best_score is None or (
            best is not None and best.score != previous_best_score
        )
        log_info_event(
            "iter.archive",
            i=iteration,
            delta=float(metadata.get("score_delta", 0.0)),
            best_updated=best_updated,
            best_score=best.score if best else 0.0,
        )
        if best is not None and best_updated:
            best_summary = summarize_hypothesis(best.hypothesis)
            best_summary.pop("fp", None)
            log_info_event(
                "best.update",
                i=iteration,
                best_score=best.score,
                best_fp=best.fingerprint[:12],
                **best_summary,
            )
        return descriptor, best_updated

    def _render_hypothesis_nl(self, hypothesis: Hypothesis) -> str:
        """Render one hypothesis back into natural language.

        Args:
            hypothesis: The hypothesis to describe.

        Returns:
            str: Natural-language text or a pretty-printed ELG fallback.
        """
        try:
            return llm_hypothesis_to_natural_language(
                hypothesis,
                llm=self.llm_client,
                retries=self.config.parser.retries,
            )
        except ParseError:
            return render_pretty(hypothesis)
