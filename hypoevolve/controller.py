"""Top-level orchestration for hypothesis evolution runs."""

from __future__ import annotations

import random
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from elg import Hypothesis, hypothesis_from_dict, render_pretty
from hypoevolve.archive import ArchiveEntry, MAPElitesArchive
from hypoevolve.config import HypoEvolveConfig
from hypoevolve.dataset import load_dataset_schema
from hypoevolve.evaluator import Evaluator, LLMEvaluator, evaluate_hypothesis
from hypoevolve.llm import LLMClient
from hypoevolve.logger import configure_logger, logger
from hypoevolve.mutation import steer_mutation
from hypoevolve.parser import (
    ParseError,
    llm_hypothesis_to_natural_language,
    llm_make_hypothesis_measurable,
    parse_hypothesis_text,
)
from hypoevolve.runtime import (
    create_run_dir,
    write_artifact,
    write_best,
    write_checkpoint,
    write_trace,
)
from hypoevolve.workers import WorkerTask, run_worker_task


@dataclass(slots=True)
class RunResult:
    """Summarize the final outcome of one evolution run."""

    run_dir: Path
    best_hypothesis: Hypothesis
    best_metrics: Dict[str, object]
    iterations: int


class HypoEvolveController:
    """Coordinate parsing, evaluation, mutation steering, and persistence."""

    def __init__(
        self,
        config: HypoEvolveConfig,
        evaluator: Optional[Evaluator] = None,
        llm_client: Optional[LLMClient] = None,
        executor_factory: Optional[Callable[..., Any]] = None,
    ):
        self.config = config
        self.llm_client = llm_client or LLMClient(config.llm)
        self.evaluator = evaluator or self._build_evaluator()
        self.rng = random.Random(config.search.random_seed)
        self.executor_factory = executor_factory or ProcessPoolExecutor

    def run(self, hypothesis_text: str) -> RunResult:
        """Execute the full evolution loop for one natural-language seed."""
        run_dir = create_run_dir(self.config.output.base_dir)
        configure_logger(
            self.config.logging.level,
            run_dir / "hypoevolve.log",
        )
        logger.info(
            "[run.start] run_dir={} iterations={} workers={} dataset_schema_path={}",
            run_dir,
            self.config.search.iterations,
            self.config.workers.count,
            self.config.evaluator.dataset_schema_path,
        )
        hypothesis = parse_hypothesis_text(
            hypothesis_text,
            llm=self.llm_client,
            retries=self.config.parser.retries,
        )
        logger.info(
            "[seed.parse] hypothesis={}",
            render_pretty(hypothesis).replace("\n", " "),
        )
        hypothesis = llm_make_hypothesis_measurable(
            hypothesis,
            llm=self.llm_client,
            retries=self.config.parser.retries,
        )
        logger.info(
            "[seed.measurable] hypothesis={}",
            render_pretty(hypothesis).replace("\n", " "),
        )
        seed_metadata = {
            "source": "seed",
            "hypothesis_nl": self._render_hypothesis_nl(hypothesis),
        }

        archive = MAPElitesArchive(
            coverage_bins=self.config.archive.coverage_bins,
            complexity_bins=self.config.archive.complexity_bins,
            per_cell_top_k=self.config.archive.per_cell_top_k,
        )
        seed_metrics = evaluate_hypothesis(hypothesis, self.evaluator)
        logger.info(
            "[seed.eval] score={:.6f} precision={} baseline={} coverage={} uplift={}",
            float(seed_metrics.get("combined_score", 0.0)),
            seed_metrics.get("precision"),
            seed_metrics.get("baseline"),
            seed_metrics.get("coverage"),
            seed_metrics.get("uplift"),
        )
        seed_descriptor = archive.describe(hypothesis, seed_metrics)
        archive.add(hypothesis, seed_metrics, iteration=0, metadata=seed_metadata)
        logger.info(
            "[seed.archive] archive_size={} best_score={:.6f} best_cell={} occupancy={}",
            len(archive),
            archive.best.score if archive.best else 0.0,
            archive.best.cell if archive.best else None,
            archive.occupancy_summary(),
        )
        write_trace(
            run_dir,
            self._trace_event(
                0,
                None,
                hypothesis,
                seed_metrics,
                {**seed_metadata, "map_elites": seed_descriptor["map_elites"]},
            ),
        )
        write_best(run_dir, archive.best.hypothesis, archive.best.metrics)
        write_checkpoint(run_dir, self._checkpoint_payload(archive, 0))
        write_artifact(
            run_dir,
            "seed",
            {
                "input_text": hypothesis_text,
                "hypothesis": hypothesis.to_dict(),
                "metrics": seed_metrics,
            },
        )

        if not self.config.workers.enabled or self.config.workers.count == 1:
            recent_history: list[Dict[str, object]] = []
            for iteration in range(1, self.config.search.iterations + 1):
                parent_entry = archive.sample_parent(self.rng)
                logger.info(
                    "[iter.parent] i={} parent_score={:.6f} parent_fp={} parent_cell={} hypothesis={}",
                    iteration,
                    parent_entry.score,
                    parent_entry.fingerprint,
                    parent_entry.cell,
                    render_pretty(parent_entry.hypothesis).replace("\n", " "),
                )
                mutation_sample, steering_metadata = self._choose_mutation(
                    parent_entry,
                    recent_history,
                    archive,
                )
                logger.info(
                    "[iter.steer] i={} mutation_summary={} score_reason={} domain_reason={}",
                    iteration,
                    str(steering_metadata.get("mutation_summary", "")).replace(
                        "\n", " "
                    ),
                    str(steering_metadata.get("score_reason", "")).replace("\n", " "),
                    str(steering_metadata.get("domain_reason", "")).replace("\n", " "),
                )
                child_metrics = evaluate_hypothesis(mutation_sample, self.evaluator)
                logger.info(
                    "[iter.eval] i={} score={:.6f} precision={} baseline={} coverage={} uplift={}",
                    iteration,
                    float(child_metrics.get("combined_score", 0.0)),
                    child_metrics.get("precision"),
                    child_metrics.get("baseline"),
                    child_metrics.get("coverage"),
                    child_metrics.get("uplift"),
                )
                score_delta = (
                    float(child_metrics.get("combined_score", 0.0)) - parent_entry.score
                )
                self._reflect_result(
                    archive,
                    run_dir,
                    iteration,
                    parent_entry.hypothesis,
                    mutation_sample,
                    child_metrics,
                    {
                        "parent_score": parent_entry.score,
                        "score_delta": score_delta,
                        **steering_metadata,
                    },
                )
                archive.record_parent_outcome(parent_entry.fingerprint, score_delta)
                recent_history.append(
                    {
                        "score_delta": score_delta,
                        "result_hypothesis": render_pretty(mutation_sample),
                        **steering_metadata,
                    }
                )
        else:
            logger.info(
                "[run.workers] workers={} mode=parallel",
                self.config.workers.count,
            )
            self._run_with_workers(archive, run_dir, self.config.search.iterations)

        best = archive.best
        logger.info(
            "[run.done] run_dir={} best_score={:.6f} archive_size={} occupancy={} best_hypothesis={}",
            run_dir,
            best.score if best else 0.0,
            len(archive),
            archive.occupancy_summary(),
            render_pretty(best.hypothesis).replace("\n", " ") if best else None,
        )
        return RunResult(
            run_dir=run_dir,
            best_hypothesis=best.hypothesis,
            best_metrics=best.metrics,
            iterations=self.config.search.iterations,
        )

    def _run_with_workers(
        self, archive: MAPElitesArchive, run_dir: Path, total_iterations: int
    ) -> None:
        worker_count = self.config.workers.count
        pending: dict[object, ArchiveEntry] = {}
        submitted = 0
        recent_history: list[Dict[str, object]] = []

        with self.executor_factory(max_workers=worker_count) as executor:
            while submitted < min(worker_count, total_iterations):
                submitted += 1
                task, parent_entry = self._make_worker_task(
                    archive, submitted, recent_history[-3:]
                )
                logger.info(
                    "[worker.submit] i={} parent_score={:.6f} parent_cell={} parent_hypothesis={}",
                    submitted,
                    archive.best.score if archive.best else 0.0,
                    parent_entry.cell,
                    render_pretty(parent_entry.hypothesis).replace("\n", " "),
                )
                future = executor.submit(run_worker_task, task)
                pending[future] = parent_entry

            while pending:
                future = next((candidate for candidate in pending if candidate.done()), None)
                if future is None:
                    continue

                parent_entry = pending.pop(future)
                result = future.result()
                child = hypothesis_from_dict(result.child_hypothesis)
                logger.info(
                    "[worker.result] i={} mutation_summary={} score={:.6f}",
                    result.iteration,
                    result.mutation_summary.replace("\n", " "),
                    float(result.metrics.get("combined_score", 0.0)),
                )
                self._reflect_result(
                    archive,
                    run_dir,
                    result.iteration,
                    parent_entry.hypothesis,
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
                archive.record_parent_outcome(
                    parent_entry.fingerprint,
                    float(result.metrics.get("combined_score", 0.0)) - result.parent_score,
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
                        archive, submitted, recent_history[-3:]
                    )
                    logger.info(
                        "[worker.submit] i={} parent_score={:.6f} parent_cell={} parent_hypothesis={}",
                        submitted,
                        archive.best.score if archive.best else 0.0,
                        next_parent_entry.cell,
                        render_pretty(next_parent_entry.hypothesis).replace("\n", " "),
                    )
                    next_future = executor.submit(run_worker_task, task)
                    pending[next_future] = next_parent_entry

    def _make_worker_task(
        self,
        archive: MAPElitesArchive,
        iteration: int,
        recent_history: list[Dict[str, object]],
    ) -> tuple[WorkerTask, ArchiveEntry]:
        parent_entry = archive.sample_parent(self.rng)
        task = WorkerTask(
            parent_hypothesis=parent_entry.hypothesis.to_dict(),
            parent_hypothesis_nl=self._get_entry_hypothesis_nl(parent_entry),
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
        )
        return task, parent_entry

    def _build_evaluator(self) -> Evaluator:
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
        use_random_steering = (
            self.rng.random() < self.config.search.random_steering_prob
        )
        decision = steer_mutation(
            parent_hypothesis=parent_entry.hypothesis,
            parent_hypothesis_nl=self._get_entry_hypothesis_nl(parent_entry),
            current_metrics=parent_entry.metrics,
            llm=self.llm_client,
            recent_history=recent_history[-3:],
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

    def _reflect_result(
        self,
        archive: MAPElitesArchive,
        run_dir: Path,
        iteration: int,
        parent_hypothesis: Hypothesis,
        child_hypothesis: Hypothesis,
        child_metrics: Dict[str, object],
        metadata: Dict[str, object],
    ) -> None:
        previous_best_score = archive.best.score if archive.best else None
        metadata = {
            **metadata,
            "hypothesis_nl": self._render_hypothesis_nl(child_hypothesis),
        }
        descriptor = archive.describe(child_hypothesis, child_metrics)
        archive.add(
            child_hypothesis,
            child_metrics,
            iteration=iteration,
            metadata=metadata,
        )
        logger.info(
            "[archive.add] i={} score={:.6f} archive_size={} child_cell={} occupancy={}",
            iteration,
            float(child_metrics.get("combined_score", 0.0)),
            len(archive),
            descriptor["cell"],
            archive.occupancy_summary(),
        )
        write_trace(
            run_dir,
            self._trace_event(
                iteration,
                parent_hypothesis,
                child_hypothesis,
                child_metrics,
                {**metadata, "map_elites": descriptor["map_elites"]},
            ),
        )
        write_checkpoint(run_dir, self._checkpoint_payload(archive, iteration))
        write_best(run_dir, archive.best.hypothesis, archive.best.metrics)
        write_artifact(
            run_dir,
            f"iteration_{iteration:04d}",
            {
                "mutation_summary": metadata.get("mutation_summary", ""),
                "domain_reason": metadata.get("domain_reason", ""),
                "score_reason": metadata.get("score_reason", ""),
                "operation_score_rankings": metadata.get("operation_score_rankings", {}),
                "metrics": child_metrics,
                "hypothesis": child_hypothesis.to_dict(),
                "worker_mode": metadata.get("worker_mode", False),
                "map_elites": descriptor["map_elites"],
            },
        )
        best = archive.best
        logger.info(
            "[iter.archive] i={} score_delta={:.6f} best_updated={} best_score={:.6f}",
            iteration,
            float(metadata.get("score_delta", 0.0)),
            previous_best_score is None or (best is not None and best.score != previous_best_score),
            best.score if best else 0.0,
        )
        if best is not None and (previous_best_score is None or best.score != previous_best_score):
            logger.info(
                "[best.update] i={} best_score={:.6f} hypothesis={}",
                iteration,
                best.score,
                render_pretty(best.hypothesis).replace("\n", " "),
            )

    def _checkpoint_payload(
        self, archive: MAPElitesArchive, iteration: int
    ) -> Dict[str, object]:
        best = archive.best
        return {
            "iteration": iteration,
            "archive_size": len(archive),
            "best_metrics": dict(best.metrics) if best else {},
            "best_hypothesis": best.hypothesis.to_dict() if best else None,
            "archive": archive.snapshot(),
        }

    def _render_hypothesis_nl(self, hypothesis: Hypothesis) -> str:
        try:
            return llm_hypothesis_to_natural_language(
                hypothesis,
                llm=self.llm_client,
                retries=self.config.parser.retries,
            )
        except ParseError:
            return render_pretty(hypothesis)

    def _get_entry_hypothesis_nl(self, entry) -> str:
        cached = str(entry.metadata.get("hypothesis_nl", "")).strip()
        if cached:
            return cached
        rendered = self._render_hypothesis_nl(entry.hypothesis)
        entry.metadata["hypothesis_nl"] = rendered
        return rendered

    def _trace_event(
        self,
        iteration: int,
        parent: Optional[Hypothesis],
        child: Hypothesis,
        metrics: Dict[str, float],
        metadata: Dict[str, object],
    ) -> Dict[str, object]:
        return {
            "iteration": iteration,
            "parent": parent.to_dict() if parent else None,
            "child": child.to_dict(),
            "metrics": metrics,
            "metadata": metadata,
        }
