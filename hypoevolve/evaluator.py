"""Evaluator interfaces and LLM-driven evaluation orchestration."""

from __future__ import annotations

import ast
import json
import math
from typing import Dict, Protocol

from elg import Hypothesis
from hypoevolve.dataset import DatasetAccessor, DatasetSchema
from hypoevolve.executor import CodeExecutor, LocalSubprocessExecutor
from hypoevolve.helper import (
    build_evaluator_prompt_variables,
    build_evaluator_runtime_wrapper,
)
from hypoevolve.llm import LLMClient
from hypoevolve.logger import logger
from hypoevolve.prompts import load_and_render_prompt, load_prompt


class Evaluator(Protocol):
    """Protocol for objects that can score a hypothesis."""

    def evaluate(self, hypothesis: Hypothesis) -> Dict[str, object]:
        """Evaluate one hypothesis.

        Args:
            hypothesis: The hypothesis to score.

        Returns:
            dict[str, object]: The normalized evaluation payload.
        """
        ...


class LLMEvaluator:
    """Generate and execute evaluator code to score a hypothesis candidate."""

    REQUIRED_KEYS = (
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

    def evaluate(self, hypothesis: Hypothesis) -> Dict[str, object]:
        """Score one hypothesis and return a normalized metric payload."""
        accessor = DatasetAccessor(self.dataset_schema)
        system_prompt = load_prompt("evaluator", "system.md")
        base_user_prompt = load_and_render_prompt(
            "evaluator",
            "user.md",
            variables=build_evaluator_prompt_variables(
                hypothesis, self.dataset_schema, accessor
            ),
        )
        wrapper = build_evaluator_runtime_wrapper(
            self.dataset_schema_path,
            self.parameters,
        )
        last_error = ""

        for attempt in range(self.codegen_retries + 1):
            user_prompt = base_user_prompt

            # Error 로 인한 Retry 에서 Error 메시지를 프롬프트에 반영
            if last_error:
                user_prompt = (
                    f"{base_user_prompt}\n\n"
                    "# Previous Attempt Failed\n\n"
                    f"{last_error}\n\n"
                    "Write a complete corrected version. Keep the same function signature and output contract."
                )

            try:
                # Code Generation
                code = self._strip_code_fences(
                    self.llm_client.generate_text(system_prompt, user_prompt)
                )
                logger.info("evaluator code generation attempt {}", attempt + 1)

                # Code Syntax Check
                try:
                    ast.parse(code)

                except SyntaxError as exc:
                    raise ValueError(
                        f"Generated code has invalid syntax: {exc}"
                    ) from exc

                # Code Execution
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

                stdout = execution.stdout.strip()
                if not stdout:
                    raise ValueError("Generated evaluator code produced empty stdout")
                payload = json.loads(stdout)

                if not isinstance(payload, dict):
                    raise ValueError("Generated evaluator output must be a JSON object")

                payload = self._sanitize_payload(payload)
                logger.info("evaluator execution succeeded")
                return payload

            except Exception as exc:  # noqa: BLE001
                message = str(exc).strip() or exc.__class__.__name__
                last_error = f"{exc.__class__.__name__}: {message}"

                if attempt >= self.codegen_retries:
                    break

        # Fail Result
        logger.error("evaluator failed after retries: {}", last_error)
        payload = {k: 0.0 for k in self.REQUIRED_KEYS}
        payload["rationale"] = f"evaluation_failed: {last_error}"
        payload["used_parameters"] = {}
        return payload

    def _sanitize_payload(self, payload: Dict[str, object]) -> Dict[str, object]:
        """Normalize non-finite numeric values after contract validation.

        Args:
            payload: The raw payload returned by generated evaluator code.

        Returns:
            dict[str, object]: A sanitized payload safe for archive insertion.
        """
        sanitized = self._validate_payload_contract(payload)
        non_finite_keys: list[str] = []

        for key in ("combined_score", "precision", "baseline", "coverage", "uplift"):
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
            logger.warning(
                "non-finite evaluator metrics detected; coercing keys={} to finite defaults",
                sorted(non_finite_keys),
            )
            rationale = str(sanitized.get("rationale", "")).strip()
            prefix = "non_finite_metrics_sanitized"
            sanitized["rationale"] = f"{prefix}: {rationale}" if rationale else prefix

        return sanitized

    def _validate_payload_contract(
        self, payload: Dict[str, object]
    ) -> Dict[str, object]:
        """Enforce required output keys and basic value types.

        Args:
            payload: The raw payload returned by generated evaluator code.

        Returns:
            dict[str, object]: A payload with required keys and safe fallback types.
        """
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

        for key in ("combined_score", "precision", "baseline", "coverage", "uplift"):
            value = sanitized.get(key)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                sanitized[key] = 0.0

        for key in ("support_count", "total_count"):
            value = sanitized.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                sanitized[key] = 0

        return sanitized

    def _strip_code_fences(self, text: str) -> str:
        """Remove leading and trailing Markdown code fences.

        Args:
            text: The raw model output text.

        Returns:
            str: The unfenced code body.
        """
        stripped = text.strip()
        if not stripped.startswith("```"):
            return stripped
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines).strip()


def evaluate_hypothesis(
    hypothesis: Hypothesis, evaluator: Evaluator
) -> Dict[str, object]:
    """Delegate hypothesis evaluation through the configured evaluator."""
    return evaluator.evaluate(hypothesis)
