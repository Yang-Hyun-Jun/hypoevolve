"""Hypothesis evolution run orchestration."""

from __future__ import annotations

import random
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from hypoevolve.memory.archive import ArchiveEntry, MAPElitesArchive
from hypoevolve.memory.artifacts import RunArtifactRecorder
from hypoevolve.memory.coulomb_archive import CoulombArchive
from hypoevolve.core.config import HypoEvolveConfig
from hypoevolve.core.events import HookBus
from hypoevolve.data.dataset import load_dataset_schema
from hypoevolve.elg import Hypothesis, fingerprint, hypothesis_from_dict, render_pretty
from hypoevolve.skills.evaluation import LLMEvaluator
from hypoevolve.skills.evaluation import Evaluator
from hypoevolve.runtime.llm_client import LLMClient
from hypoevolve.observability.logger import (
    compact_text,
    configure_logger,
    log_error_event,
    log_info_event,
    summarize_exception,
    summarize_hypothesis,
    summarize_metrics,
)
from hypoevolve.skills.mutation import steer_mutation
from hypoevolve.skills.elg_compile import (
    ParseError,
    llm_hypothesis_to_natural_language,
    llm_make_hypothesis_measurable,
    parse_hypothesis_text,
)
from hypoevolve.runtime.checkpoint import create_run_dir
from hypoevolve.skills.seed_generation import generate_random_tree_pair_hypothesis
from hypoevolve.runtime.worker import WorkerTask
from hypoevolve.runtime.worker import run_worker_task
from hypoevolve.policies.protocols import SelectionPolicy
from hypoevolve.policies.selection import CoulombSelectionPolicy, UCBSelectionPolicy


@dataclass(slots=True)
class RunResult:
    """Summarize the final outcome of one evolution run."""

    run_dir: Path
    seed_hypothesis: Hypothesis
    best_hypothesis: Hypothesis
    best_metrics: dict[str, object]
    iterations: int
    report_path: Path
    seed_input_text: str
    seed_generated: bool


@dataclass(slots=True)
class _CompletedChildState:
    """Carry normalized bookkeeping outputs for one evaluated child."""

    child_fingerprint: str
    history_entry: dict[str, object]


@dataclass(slots=True)
class _SeedBootstrapState:
    """Carry the initialized seed state for a controller run."""

    hypothesis: Hypothesis
    archive: MAPElitesArchive
    known_fingerprints: set[str]


@dataclass(slots=True)
class _RunPreparationState:
    """Carry top-level run setup outputs before search execution starts."""

    seed_input_text: str
    seed_generated: bool
    run_dir: Path
    recorder: RunArtifactRecorder


def _build_steering_metadata(
    source: object,
    *,
    random_steering: bool,
) -> dict[str, object]:
    """Normalize steering metadata from local or worker mutation results."""
    return {
        "steered": True,
        "domain_reason": getattr(source, "domain_reason", ""),
        "score_reason": getattr(source, "score_reason", ""),
        "operation_score_rankings": dict(
            getattr(source, "operation_score_rankings", {}) or {}
        ),
        "mutation_summary": getattr(source, "mutation_summary", ""),
        "random_steering": random_steering,
    }


def _record_skip(
    *,
    archive: MAPElitesArchive,
    recorder: RunArtifactRecorder,
    iteration: int,
    parent_entry: ArchiveEntry,
    worker_mode: bool,
    child_fingerprint: str | None = None,
    error: str | None = None,
) -> None:
    """Record one non-evaluated iteration and preserve skip semantics."""
    best_score_after = archive.best.score if archive.best else 0.0
    archive.record_parent_outcome(parent_entry.fingerprint, 0.0)
    if error is not None:
        recorder.record_steering_skip(
            iteration=iteration,
            parent_fingerprint=parent_entry.fingerprint,
            parent_score=parent_entry.score,
            best_score_after=best_score_after,
            worker_mode=worker_mode,
            error=error,
        )
        return
    recorder.record_duplicate_skip(
        iteration=iteration,
        parent_fingerprint=parent_entry.fingerprint,
        child_fingerprint=child_fingerprint or "",
        parent_score=parent_entry.score,
        best_score_after=best_score_after,
        worker_mode=worker_mode,
    )


def _record_completed_child(
    *,
    archive: MAPElitesArchive,
    recorder: RunArtifactRecorder,
    iteration: int,
    parent_entry: ArchiveEntry,
    child_hypothesis: Hypothesis,
    child_metrics: dict[str, object],
    steering_metadata: dict[str, object],
    evaluation_artifacts: dict[str, object],
    worker_mode: bool,
) -> _CompletedChildState:
    """Persist one completed child evaluation and return shared loop outputs."""
    child_fingerprint = fingerprint(child_hypothesis)
    score_delta = float(child_metrics.get("combined_score", 0.0)) - parent_entry.score
    iteration_metadata = {
        "parent_score": parent_entry.score,
        "score_delta": score_delta,
        **steering_metadata,
    }
    if worker_mode:
        iteration_metadata["worker_mode"] = True

    previous_best_score = archive.best.score if archive.best else None
    descriptor = archive.describe(child_hypothesis, child_metrics)
    archive.add(
        child_hypothesis,
        child_metrics,
        iteration=iteration,
        metadata=iteration_metadata,
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
        delta=float(iteration_metadata.get("score_delta", 0.0)),
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

    recorder.record_iteration_result(
        archive=archive,
        iteration=iteration,
        parent_hypothesis=parent_entry.hypothesis,
        parent_fingerprint=parent_entry.fingerprint,
        child_hypothesis=child_hypothesis,
        child_metrics=child_metrics,
        metadata=iteration_metadata,
        descriptor=descriptor,
        best_updated=best_updated,
        evaluation_artifacts=evaluation_artifacts,
    )
    archive.record_parent_outcome(parent_entry.fingerprint, score_delta)
    return _CompletedChildState(
        child_fingerprint=child_fingerprint,
        history_entry={
            "score_delta": score_delta,
            "result_hypothesis": render_pretty(child_hypothesis),
            **steering_metadata,
        },
    )


class HypoEvolveController:
    """Coordinate one hypothesis evolution run."""

    def __init__(
        self,
        config: HypoEvolveConfig,
        evaluator: Evaluator | None = None,
        llm_client: LLMClient | None = None,
        executor_factory: Callable[..., object] | None = None,
        hooks: HookBus | None = None,
        selection_policy: SelectionPolicy | None = None,
    ):
        """Initialize one run controller.

        Args:
            config: The active runtime configuration.
            evaluator: Optional evaluator override for tests or custom execution.
            llm_client: Optional shared LLM client override.
            executor_factory: Optional worker executor factory override.
            hooks: Optional event bus for lifecycle hooks.
            selection_policy: Optional parent selection policy (defaults to UCB).

        Returns:
            None.
        """
        self.config = config
        self.llm_client = llm_client or LLMClient(config.llm)
        self.evaluator = evaluator or LLMEvaluator(
            llm_client=self.llm_client,
            dataset_schema=load_dataset_schema(config.evaluator.dataset_schema_path),
            dataset_schema_path=config.evaluator.dataset_schema_path,
            parameters=config.evaluator.parameters or None,
        )
        self.rng = random.Random(config.search.random_seed)
        self.executor_factory = executor_factory or ProcessPoolExecutor
        self.hooks = hooks or HookBus()
        self.selection_policy = selection_policy or _default_selection_policy(config)

    def _bootstrap_seed(
        self,
        *,
        seed_input_text: str,
        recorder: RunArtifactRecorder,
    ) -> _SeedBootstrapState:
        """Parse, evaluate, archive, and persist the seed hypothesis."""
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
        archive = _build_archive(self.config)
        seed_metrics = self.evaluator.evaluate(hypothesis)
        seed_evaluation_artifacts = dict(
            getattr(self.evaluator, "last_evaluation_artifacts", {}) or {}
        )
        log_info_event(
            "seed.eval",
            **summarize_hypothesis(hypothesis),
            **summarize_metrics(seed_metrics),
        )
        seed_descriptor = archive.describe(hypothesis, seed_metrics)
        archive.add(hypothesis, seed_metrics, iteration=0, metadata=seed_metadata)
        known_fingerprints = {fingerprint(hypothesis)}
        best = archive.best
        log_info_event(
            "seed.archive",
            archive_size=len(archive),
            best_score=best.score if best else 0.0,
            best_cell=best.cell if best else None,
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
        return _SeedBootstrapState(
            hypothesis=hypothesis,
            archive=archive,
            known_fingerprints=known_fingerprints,
        )

    def _finalize_run_result(
        self,
        *,
        archive: MAPElitesArchive,
        recorder: RunArtifactRecorder,
        known_fingerprint_count: int,
        run_dir: Path,
        seed_hypothesis: Hypothesis,
        seed_input_text: str,
        seed_generated: bool,
    ) -> RunResult:
        """Finalize persisted outputs and return the public run summary."""
        best = archive.best
        best_hypothesis_nl = ""
        if best:
            try:
                best_hypothesis_nl = llm_hypothesis_to_natural_language(
                    best.hypothesis,
                    llm=self.llm_client,
                    retries=self.config.parser.retries,
                )
            except ParseError:
                best_hypothesis_nl = render_pretty(best.hypothesis)
        report_path = recorder.finalize(
            archive=archive,
            iterations_requested=self.config.search.iterations,
            known_fingerprint_count=known_fingerprint_count,
            best_hypothesis_nl=best_hypothesis_nl,
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
        self.hooks.emit(
            "run.done",
            run=run_dir.name,
            best_score=best.score if best else 0.0,
            archive_size=len(archive),
        )
        return RunResult(
            run_dir=run_dir,
            seed_hypothesis=seed_hypothesis,
            best_hypothesis=best.hypothesis,
            best_metrics=best.metrics,
            iterations=self.config.search.iterations,
            report_path=report_path,
            seed_input_text=seed_input_text,
            seed_generated=seed_generated,
        )

    def _choose_mutation(
        self,
        *,
        parent_entry: ArchiveEntry,
        recent_history: list[dict[str, object]],
        archive: MAPElitesArchive,
    ) -> tuple[Hypothesis, dict[str, object]]:
        """Choose one steered mutation for the solo execution path."""
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
        return decision.child_hypothesis, _build_steering_metadata(
            decision,
            random_steering=use_random_steering,
        )

    def _run_single_process_iterations(
        self,
        *,
        archive: MAPElitesArchive,
        recorder: RunArtifactRecorder,
        known_fingerprints: set[str],
    ) -> None:
        """Run the single-process mutation/evaluation loop."""
        recent_history: list[dict[str, object]] = []
        for iteration in range(1, self.config.search.iterations + 1):
            parent_entry = self.selection_policy.select(archive, self.rng)
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
                    parent_entry=parent_entry,
                    recent_history=recent_history,
                    archive=archive,
                )
            except ParseError as exc:
                log_error_event(
                    "iter.skip_steering_error",
                    i=iteration,
                    parent_fp=parent_entry.fingerprint[:12],
                    **summarize_exception(exc),
                )
                _record_skip(
                    archive=archive,
                    recorder=recorder,
                    iteration=iteration,
                    parent_entry=parent_entry,
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
                _record_skip(
                    archive=archive,
                    recorder=recorder,
                    iteration=iteration,
                    parent_entry=parent_entry,
                    worker_mode=False,
                    child_fingerprint=child_fingerprint,
                )
                continue
            child_metrics = self.evaluator.evaluate(mutation_sample)
            child_evaluation_artifacts = dict(
                getattr(self.evaluator, "last_evaluation_artifacts", {}) or {}
            )
            log_info_event(
                "iter.eval",
                i=iteration,
                child_fp=child_fingerprint[:12],
                **summarize_metrics(child_metrics),
            )
            completed_child = _record_completed_child(
                archive=archive,
                recorder=recorder,
                iteration=iteration,
                parent_entry=parent_entry,
                child_hypothesis=mutation_sample,
                child_metrics=child_metrics,
                steering_metadata=steering_metadata,
                evaluation_artifacts=child_evaluation_artifacts,
                worker_mode=False,
            )
            known_fingerprints.add(completed_child.child_fingerprint)
            recent_history.append(completed_child.history_entry)

    def _run_worker_iterations(
        self,
        *,
        archive: MAPElitesArchive,
        recorder: RunArtifactRecorder,
        known_fingerprints: set[str],
    ) -> int:
        """Run the worker submission/result integration loop and return known-fingerprint count."""
        log_info_event(
            "run.workers",
            workers=self.config.workers.count,
            mode="parallel",
        )
        pending: dict[object, ArchiveEntry] = {}
        submitted = 0
        recent_history: list[dict[str, object]] = []

        with self.executor_factory(max_workers=self.config.workers.count) as executor:

            def submit_next() -> None:
                nonlocal submitted
                submitted += 1
                parent_entry = self.selection_policy.select(archive, self.rng)
                task = WorkerTask(
                    parent_hypothesis=parent_entry.hypothesis.to_dict(),
                    parent_metrics=dict(parent_entry.metrics),
                    iteration=submitted,
                    parent_score=parent_entry.score,
                    use_random_steering=(
                        self.rng.random() < self.config.search.random_steering_prob
                    ),
                    llm_config=asdict(self.config.llm),
                    dataset_schema_path=self.config.evaluator.dataset_schema_path,
                    evaluator_parameters=dict(self.config.evaluator.parameters),
                    steering_retries=self.config.search.steering_retries,
                    recent_history=list(recent_history[-2:]),
                    top_hypotheses=archive.snapshot()[:3],
                    seen_fingerprints=sorted(known_fingerprints),
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

            while submitted < min(
                self.config.workers.count, self.config.search.iterations
            ):
                submit_next()

            while pending:
                done, _ = wait(tuple(pending), return_when=FIRST_COMPLETED)
                for future in done:
                    parent_entry = pending.pop(future)
                    result = future.result()
                    if result.skipped_steering_error:
                        log_error_event(
                            "worker.skip_steering_error",
                            i=result.iteration,
                            parent_fp=parent_entry.fingerprint[:12],
                            **summarize_exception(result.steering_error),
                        )
                        _record_skip(
                            archive=archive,
                            recorder=recorder,
                            iteration=result.iteration,
                            parent_entry=parent_entry,
                            worker_mode=True,
                            error=str(result.steering_error),
                        )
                        if submitted < self.config.search.iterations:
                            submit_next()
                        continue
                    if result.skipped_duplicate:
                        log_info_event(
                            "worker.skip_duplicate",
                            i=result.iteration,
                            child_fp=result.child_fingerprint[:12],
                        )
                        _record_skip(
                            archive=archive,
                            recorder=recorder,
                            iteration=result.iteration,
                            parent_entry=parent_entry,
                            worker_mode=True,
                            child_fingerprint=result.child_fingerprint,
                        )
                        if submitted < self.config.search.iterations:
                            submit_next()
                        continue
                    child = hypothesis_from_dict(result.child_hypothesis)
                    worker_steering_metadata = _build_steering_metadata(
                        result,
                        random_steering=result.random_steering,
                    )
                    log_info_event(
                        "worker.result",
                        i=result.iteration,
                        child_fp=result.child_fingerprint[:12]
                        if result.child_fingerprint
                        else None,
                        **summarize_metrics(result.metrics),
                        summary=compact_text(result.mutation_summary, max_len=96),
                    )
                    completed_child = _record_completed_child(
                        archive=archive,
                        recorder=recorder,
                        iteration=result.iteration,
                        parent_entry=parent_entry,
                        child_hypothesis=child,
                        child_metrics=result.metrics,
                        steering_metadata=worker_steering_metadata,
                        evaluation_artifacts=result.evaluation_artifacts,
                        worker_mode=True,
                    )
                    if result.child_fingerprint:
                        known_fingerprints.add(completed_child.child_fingerprint)
                    recent_history.append(completed_child.history_entry)
                    if submitted < self.config.search.iterations:
                        submit_next()
        return len(known_fingerprints)

    def _resolve_seed_input_text(
        self,
        hypothesis_text: str | None,
    ) -> tuple[str, bool]:
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

    def _prepare_run(
        self,
        hypothesis_text: str | None,
    ) -> _RunPreparationState:
        """Resolve seed input and initialize run-scoped filesystem/logging state."""
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
        self.hooks.emit(
            "run.start",
            run=run_dir.name,
            iterations=self.config.search.iterations,
        )
        return _RunPreparationState(
            seed_input_text=seed_input_text,
            seed_generated=seed_generated,
            run_dir=run_dir,
            recorder=recorder,
        )

    def _execute_search_branch(
        self,
        *,
        seed_state: _SeedBootstrapState,
        recorder: RunArtifactRecorder,
    ) -> int:
        """Dispatch to the appropriate search branch and return the final known-fingerprint count."""
        if not self.config.workers.enabled or self.config.workers.count == 1:
            self._run_single_process_iterations(
                archive=seed_state.archive,
                recorder=recorder,
                known_fingerprints=seed_state.known_fingerprints,
            )
            return len(seed_state.known_fingerprints)
        return self._run_worker_iterations(
            archive=seed_state.archive,
            recorder=recorder,
            known_fingerprints=seed_state.known_fingerprints,
        )

    def run(self, hypothesis_text: str | None = None) -> RunResult:
        """Execute the full evolution loop for one natural-language seed."""
        prepared = self._prepare_run(hypothesis_text)
        seed_state = self._bootstrap_seed(
            seed_input_text=prepared.seed_input_text,
            recorder=prepared.recorder,
        )
        known_fingerprint_count = self._execute_search_branch(
            seed_state=seed_state,
            recorder=prepared.recorder,
        )
        return self._finalize_run_result(
            archive=seed_state.archive,
            recorder=prepared.recorder,
            known_fingerprint_count=known_fingerprint_count,
            run_dir=prepared.run_dir,
            seed_hypothesis=seed_state.hypothesis,
            seed_input_text=prepared.seed_input_text,
            seed_generated=prepared.seed_generated,
        )


def _default_selection_policy(config: HypoEvolveConfig) -> SelectionPolicy:
    """Return the default selection policy that matches ``config.archive.kind``."""
    if config.archive.kind == "coulomb":
        return CoulombSelectionPolicy()
    return UCBSelectionPolicy()


def _build_archive(
    config: HypoEvolveConfig,
) -> MAPElitesArchive | CoulombArchive:
    """Instantiate the archive implementation selected by ``config.archive.kind``."""
    if config.archive.kind == "coulomb":
        return CoulombArchive(
            capacity=config.archive.coulomb.capacity,
            gamma=config.archive.coulomb.gamma,
            eps=config.archive.coulomb.eps,
        )
    return MAPElitesArchive(
        coverage_bins=config.archive.coverage_bins,
        complexity_bins=config.archive.complexity_bins,
        per_cell_top_k=config.archive.per_cell_top_k,
        parent_sampling_mode=config.archive.parent_sampling_mode,
    )
