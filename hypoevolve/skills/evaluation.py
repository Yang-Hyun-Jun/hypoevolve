"""Evaluator contracts and LLM-driven evaluation flow.

Canonical merged module -- consolidates evaluator_contracts and evaluator.
"""

from __future__ import annotations

import ast
import json
import math
from typing import Protocol

from hypoevolve.context.providers import (
    build_evaluator_prompt_variables,
    build_evaluator_runtime_wrapper,
)
from hypoevolve.data.dataset import DatasetAccessor, DatasetSchema
from hypoevolve.elg import Hypothesis
from hypoevolve.observability.logger import (
    log_debug_event,
    log_error_event,
    log_warning_event,
    summarize_exception,
)
from hypoevolve.prompts import load_and_render_prompt, load_prompt
from hypoevolve.runtime.llm_client import LLMClient
from hypoevolve.runtime.sandbox import CodeExecutor, LocalSubprocessExecutor

REQUIRED_EVALUATION_KEYS = (
    "combined_score",
    "precision",
    "baseline",
    "coverage",
    "uplift",
    "support_count",
    "total_count",
    "rationale",
    "used_parameters",
)


class Evaluator(Protocol):
    """Protocol for objects that can score a hypothesis."""

    def evaluate(self, hypothesis: Hypothesis) -> dict[str, object]:
        """Evaluate one hypothesis and return a normalized payload."""
        ...


class LLMEvaluator:
    """Generate and execute evaluator code for a hypothesis."""

    REQUIRED_KEYS = REQUIRED_EVALUATION_KEYS

    def __init__(
        self,
        llm_client: LLMClient,
        dataset_schema: DatasetSchema,
        dataset_schema_path: str,
        executor: CodeExecutor | None = None,
        parameters: dict[str, object] | None = None,
    ):
        """Initialize one evaluator around an LLM and dataset schema.

        Args:
            llm_client: The LLM client used for evaluator code generation.
            dataset_schema: The loaded dataset schema available to generated code.
            dataset_schema_path: The on-disk schema path passed into the runtime wrapper.
            executor: Optional code executor override.
            parameters: Optional evaluator parameters forwarded to generated code.

        Returns:
            None.
        """
        self.llm_client = llm_client
        self.dataset_schema = dataset_schema
        self.dataset_schema_path = dataset_schema_path
        self.executor = executor or LocalSubprocessExecutor()
        self.parameters = dict(parameters) if parameters is not None else None
        self.codegen_retries = max(0, llm_client.config.retries)
        self.last_evaluation_artifacts: dict[str, object] = {}

    def evaluate(self, hypothesis: Hypothesis) -> dict[str, object]:
        """Score one hypothesis and return a normalized metric payload."""
        accessor = DatasetAccessor(self.dataset_schema)
        system_prompt = load_prompt("evaluator", "system.md")
        base_user_prompt = load_and_render_prompt(
            "evaluator",
            "user.md",
            variables=build_evaluator_prompt_variables(
                hypothesis,
                self.dataset_schema,
                accessor,
            ),
        )
        wrapper = build_evaluator_runtime_wrapper(
            self.dataset_schema_path,
            self.parameters,
        )
        last_error = ""
        last_candidate_code = ""
        self.last_evaluation_artifacts = {}

        for attempt in range(self.codegen_retries + 1):
            attempt_number = attempt + 1
            user_prompt = base_user_prompt
            if last_error:
                user_prompt = (
                    f"{base_user_prompt}\n\n"
                    "# Previous Attempt Failed\n\n"
                    "## Failure Message\n\n"
                    f"{last_error}\n\n"
                    "# Previous Candidate Code\n\n"
                    f"{last_candidate_code or '<no previous candidate code>'}\n\n"
                    "# Repair Requirements\n\n"
                    "- Diagnose the failure using both the traceback/error message and the previous code.\n"
                    "- Keep the same function signature and output contract.\n"
                    "- Do not reuse any dataframe column name unless it exactly matches a provided schema column or is created earlier in the function.\n"
                    "- If the failure mentions a missing column or KeyError, fix the exact mismatch instead of guessing a similar name.\n"
                    "- Write a complete corrected version.\n"
                )

            try:
                code = self.llm_client.generate_text(system_prompt, user_prompt).strip()
                if code.startswith("```"):
                    lines = code.splitlines()
                    if lines and lines[0].startswith("```"):
                        lines = lines[1:]
                    if lines and lines[-1].startswith("```"):
                        lines = lines[:-1]
                    code = "\n".join(lines).strip()
                log_debug_event("eval.codegen", attempt=attempt_number)
                last_candidate_code = code
                self.last_evaluation_artifacts = {
                    "candidate_code": code,
                    "wrapper_code": wrapper,
                    "attempt": attempt_number,
                }
                try:
                    ast.parse(code)
                except SyntaxError as exc:
                    raise ValueError(
                        f"Generated code has invalid syntax: {exc}"
                    ) from exc
                execution = self.executor.execute(
                    wrapper,
                    files={"candidate.py": f"{code.rstrip()}\n"},
                )

                if execution.timed_out:
                    raise RuntimeError("Generated evaluator code timed out")
                if execution.exit_code != 0:
                    stderr = execution.stderr.strip() or "<empty stderr>"
                    raise RuntimeError(
                        f"Generated evaluator code failed with exit_code={execution.exit_code}: {stderr}"
                    )

                self.last_evaluation_artifacts["work_dir"] = execution.work_dir
                stdout = execution.stdout.strip()
                if not stdout:
                    raise ValueError("Generated evaluator code produced empty stdout")
                payload = json.loads(stdout)
                if not isinstance(payload, dict):
                    raise ValueError("Generated evaluator output must be a JSON object")
                duration_ms = int(execution.duration_sec * 1000)
                log_debug_event("eval.exec", attempt=attempt_number, dur_ms=duration_ms)
                sanitized = {
                    "combined_score": 0.0,
                    "precision": 0.0,
                    "baseline": 0.0,
                    "coverage": 0.0,
                    "uplift": 0.0,
                    "support_count": 0,
                    "total_count": 0,
                    "rationale": "",
                    "used_parameters": {},
                }
                sanitized.update(payload)

                if not isinstance(sanitized.get("used_parameters"), dict):
                    sanitized["used_parameters"] = {}

                rationale = sanitized.get("rationale")
                sanitized["rationale"] = rationale if isinstance(rationale, str) else ""

                for key in (
                    "combined_score",
                    "precision",
                    "baseline",
                    "coverage",
                    "uplift",
                ):
                    value = sanitized.get(key)
                    if not isinstance(value, (int, float)) or isinstance(value, bool):
                        sanitized[key] = 0.0

                for key in ("support_count", "total_count"):
                    value = sanitized.get(key)
                    if isinstance(value, bool) or not isinstance(value, (int, float)):
                        sanitized[key] = 0

                non_finite_keys: list[str] = []

                for key in (
                    "combined_score",
                    "precision",
                    "baseline",
                    "coverage",
                    "uplift",
                ):
                    value = sanitized.get(key)
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        if not math.isfinite(float(value)):
                            sanitized[key] = 0.0
                            non_finite_keys.append(key)

                for key in ("support_count", "total_count"):
                    value = sanitized.get(key)
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        if not math.isfinite(float(value)):
                            sanitized[key] = 0
                            non_finite_keys.append(key)

                if non_finite_keys:
                    log_warning_event(
                        "eval.sanitize_non_finite",
                        keys=sorted(non_finite_keys),
                    )
                    rationale = str(sanitized.get("rationale", "")).strip()
                    prefix = "non_finite_metrics_sanitized"
                    sanitized["rationale"] = (
                        f"{prefix}: {rationale}" if rationale else prefix
                    )

                return sanitized
            except Exception as exc:  # noqa: BLE001
                message = str(exc).strip() or exc.__class__.__name__
                last_error = f"{exc.__class__.__name__}: {message}"
                if attempt < self.codegen_retries:
                    log_warning_event(
                        "eval.retry",
                        attempt=attempt_number,
                        **summarize_exception(exc),
                    )
                if attempt >= self.codegen_retries:
                    break

        log_error_event("eval.fail", **summarize_exception(last_error))
        payload = {k: 0.0 for k in self.REQUIRED_KEYS}
        payload["rationale"] = f"evaluation_failed: {last_error}"
        payload["used_parameters"] = {}
        return payload


__all__ = ["Evaluator", "LLMEvaluator", "REQUIRED_EVALUATION_KEYS"]
