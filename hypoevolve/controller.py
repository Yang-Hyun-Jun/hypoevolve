from __future__ import annotations

import random
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Callable, Dict, Optional

from elg import Hypothesis, hypothesis_from_dict, hypothesis_to_json, render_pretty, sample_mutation
from hypoevolve.archive import Archive
from hypoevolve.config import HypoEvolveConfig
from hypoevolve.evaluator import Evaluator, PlaceholderEvaluator, evaluate_hypothesis
from hypoevolve.llm import LLMClient
from hypoevolve.parser import ParseError, parse_hypothesis_text
from hypoevolve.runtime import create_run_dir, write_artifact, write_best, write_checkpoint, write_trace
from hypoevolve.workers import WorkerResult, WorkerTask, run_worker_task


@dataclass(slots=True)
class RunResult:
    run_dir: Path
    best_hypothesis: Hypothesis
    best_metrics: Dict[str, float]
    iterations: int


class HypoEvolveController:
    def __init__(
        self,
        config: HypoEvolveConfig,
        evaluator: Optional[Evaluator] = None,
        llm_client: Optional[LLMClient] = None,
        executor_factory: Optional[Callable[..., Any]] = None,
    ):
        self.config = config
        self.evaluator = evaluator or PlaceholderEvaluator(seed=config.evaluator.seed)
        self.llm_client = llm_client or LLMClient(config.llm)
        self.rng = random.Random(config.search.random_seed)
        self.executor_factory = executor_factory or ProcessPoolExecutor

    def run(self, hypothesis_text: str) -> RunResult:
        run_dir = create_run_dir(self.config.output.base_dir)
        hypothesis = parse_hypothesis_text(
            hypothesis_text,
            llm=self.llm_client,
            retries=self.config.parser.retries,
        )

        archive = Archive(top_k=self.config.archive.top_k)
        seed_metrics = evaluate_hypothesis(hypothesis, self.evaluator)
        archive.add(hypothesis, seed_metrics, iteration=0, metadata={"source": "seed"})
        write_trace(run_dir, self._trace_event(0, None, hypothesis, seed_metrics, {"source": "seed"}))
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

        if not self.config.workers.enabled or self.config.workers.count == 1 or not isinstance(self.evaluator, PlaceholderEvaluator):
            for iteration in range(1, self.config.search.iterations + 1):
                parent_entry = archive.sample_parent(self.rng)
                mutation_sample = sample_mutation(
                    parent_entry.hypothesis,
                    rng=self.rng,
                    atomic_pool=self.config.search.mutation_atomic_pool,
                )
                child_metrics = evaluate_hypothesis(mutation_sample.result, self.evaluator)
                self._reflect_result(
                    archive,
                    run_dir,
                    iteration,
                    parent_entry.hypothesis,
                    mutation_sample.result,
                    child_metrics,
                    {
                        "operation": mutation_sample.operation,
                        "path": list(mutation_sample.path),
                        "details": mutation_sample.details,
                        "parent_score": parent_entry.score,
                    },
                )
        else:
            self._run_with_workers(archive, run_dir, self.config.search.iterations)

        best = archive.best
        return RunResult(
            run_dir=run_dir,
            best_hypothesis=best.hypothesis,
            best_metrics=best.metrics,
            iterations=self.config.search.iterations,
        )

    def _run_with_workers(self, archive: Archive, run_dir: Path, total_iterations: int) -> None:
        worker_count = self.config.workers.count
        pending = []
        submitted = 0

        with self.executor_factory(max_workers=worker_count) as executor:
            while submitted < min(worker_count, total_iterations):
                submitted += 1
                task, parent = self._make_worker_task(archive, submitted)
                future = executor.submit(run_worker_task, task)
                pending.append((future, parent))

            while pending:
                completed_index = next(
                    (index for index, (future, _parent) in enumerate(pending) if future.done()),
                    None,
                )
                if completed_index is None:
                    time.sleep(0.001)
                    continue

                future, parent = pending.pop(completed_index)
                result = future.result()
                child = hypothesis_from_dict(result.child_hypothesis)
                self._reflect_result(
                    archive,
                    run_dir,
                    result.iteration,
                    parent,
                    child,
                    result.metrics,
                    {
                        "operation": result.mutation_operation,
                        "path": result.mutation_path,
                        "details": result.mutation_details,
                        "parent_score": result.parent_score,
                        "worker_mode": True,
                    },
                )
                if submitted < total_iterations:
                    submitted += 1
                    task, next_parent = self._make_worker_task(archive, submitted)
                    next_future = executor.submit(run_worker_task, task)
                    pending.append((next_future, next_parent))

    def _make_worker_task(self, archive: Archive, iteration: int) -> tuple[WorkerTask, Hypothesis]:
        parent_entry = archive.sample_parent(self.rng)
        task = WorkerTask(
            parent_hypothesis=parent_entry.hypothesis.to_dict(),
            iteration=iteration,
            parent_score=parent_entry.score,
            mutation_atomic_pool=list(self.config.search.mutation_atomic_pool),
            evaluator_seed=self.config.evaluator.seed,
        )
        return task, parent_entry.hypothesis

    def _reflect_result(
        self,
        archive: Archive,
        run_dir: Path,
        iteration: int,
        parent_hypothesis: Hypothesis,
        child_hypothesis: Hypothesis,
        child_metrics: Dict[str, float],
        metadata: Dict[str, object],
    ) -> None:
        archive.add(
            child_hypothesis,
            child_metrics,
            iteration=iteration,
            metadata=metadata,
        )
        write_trace(
            run_dir,
            self._trace_event(iteration, parent_hypothesis, child_hypothesis, child_metrics, metadata),
        )
        write_checkpoint(run_dir, self._checkpoint_payload(archive, iteration))
        write_best(run_dir, archive.best.hypothesis, archive.best.metrics)
        write_artifact(
            run_dir,
            f"iteration_{iteration:04d}",
            {
                "mutation": metadata.get("operation"),
                "path": metadata.get("path", []),
                "details": metadata.get("details", {}),
                "metrics": child_metrics,
                "hypothesis": child_hypothesis.to_dict(),
                "worker_mode": metadata.get("worker_mode", False),
            },
        )

    def _checkpoint_payload(self, archive: Archive, iteration: int) -> Dict[str, object]:
        best = archive.best
        return {
            "iteration": iteration,
            "archive_size": len(archive),
            "best_metrics": dict(best.metrics) if best else {},
            "best_hypothesis": best.hypothesis.to_dict() if best else None,
            "archive": archive.snapshot(),
        }

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
