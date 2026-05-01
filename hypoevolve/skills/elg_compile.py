"""Natural-language parsing and ELG validation helpers."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from hypoevolve.elg import Hypothesis, hypothesis_from_dict, normalize_hypothesis
from hypoevolve.observability.logger import (
    log_debug_event,
    log_error_event,
    summarize_exception,
)
from hypoevolve.runtime.llm_client import LLMClient
from hypoevolve.prompts import load_prompt


class ParseError(ValueError):
    """Raised when an LLM payload cannot be parsed into valid ELG."""

    def __init__(self, message: str, errors: Optional[List[str]] = None):
        super().__init__(message)
        self.errors = errors or []


PARSER_SYSTEM_PROMPT = load_prompt("parser", "system.md")
JSON_RETRY_PROMPT = load_prompt("common", "json-retry.md")
MEASURABLE_SYSTEM_PROMPT = load_prompt("measurable", "system.md")
NL_SYSTEM_PROMPT = load_prompt("nl", "system.md")


def parse_hypothesis_text(
    text: str,
    llm: LLMClient,
    retries: int = 1,
) -> Hypothesis:
    """Convert natural-language hypothesis text into a normalized ELG object."""
    return llm_parse_hypothesis(text, llm=llm, retries=retries)


def llm_parse_hypothesis(
    text: str,
    llm: LLMClient,
    retries: int = 2,
) -> Hypothesis:
    """Parse natural-language text into a normalized ELG hypothesis."""
    if not text or not text.strip():
        raise ParseError("Hypothesis text must be non-empty")

    errors: List[str] = []
    attempts = retries + 1
    log_debug_event(
        "parser.start",
        attempts=attempts,
        chars=len(text.strip()),
    )
    for attempt in range(1, attempts + 1):
        try:
            log_debug_event("parser.attempt", attempt=attempt)
            system_prompt = (
                PARSER_SYSTEM_PROMPT
                if attempt == 1
                else f"{PARSER_SYSTEM_PROMPT}\n\n{JSON_RETRY_PROMPT}"
            )
            payload = llm.generate_json(
                system_prompt,
                f"Convert this natural-language hypothesis into ELG JSON root node:\\n\\n{text}",
                json_retries=0,
            )
            _validate_parser_payload(payload)
            hypothesis = hypothesis_from_dict({"root": payload})
            log_debug_event("parser.ok", attempt=attempt)
            return normalize_hypothesis(hypothesis)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"attempt {attempt}: {exc}")
            log_error_event("parser.fail", attempt=attempt, **summarize_exception(exc))
    raise ParseError(
        "Failed to convert natural-language hypothesis to ELG via LLM", errors=errors
    )


def llm_make_hypothesis_measurable(
    hypothesis: Hypothesis,
    llm: LLMClient,
    retries: int = 2,
) -> Hypothesis:
    """Rewrite a hypothesis into a more measurable ELG form."""
    errors: List[str] = []
    attempts = retries + 1
    log_debug_event("measurable.start", attempts=attempts)

    for attempt in range(1, attempts + 1):
        try:
            log_debug_event("measurable.attempt", attempt=attempt)
            system_prompt = (
                MEASURABLE_SYSTEM_PROMPT
                if attempt == 1
                else f"{MEASURABLE_SYSTEM_PROMPT}\n\n{JSON_RETRY_PROMPT}"
            )
            payload = llm.generate_json(
                system_prompt,
                "Convert this ELG hypothesis into a more measurable ELG root node while preserving structure as much as possible:\n\n"
                f"{hypothesis.to_dict()['root']}",
                json_retries=0,
            )
            _validate_parser_payload(payload)
            measurable = hypothesis_from_dict({"root": payload})
            log_debug_event("measurable.ok", attempt=attempt)
            return normalize_hypothesis(measurable)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"attempt {attempt}: {exc}")
            log_error_event(
                "measurable.fail", attempt=attempt, **summarize_exception(exc)
            )

    raise ParseError(
        "Failed to convert ELG hypothesis into measurable ELG via LLM",
        errors=errors,
    )


def llm_hypothesis_to_natural_language(
    hypothesis: Hypothesis,
    llm: LLMClient,
    retries: int = 2,
) -> str:
    """Render a measurable ELG hypothesis back into natural language."""
    errors: List[str] = []
    attempts = retries + 1
    log_debug_event("nl_render.start", attempts=attempts)

    for attempt in range(1, attempts + 1):
        try:
            log_debug_event("nl_render.attempt", attempt=attempt)
            system_prompt = NL_SYSTEM_PROMPT
            response = llm.generate_text(
                system_prompt,
                "Convert this ELG hypothesis into natural language:\n\n"
                f"{hypothesis.to_dict()['root']}",
            )
            rendered = response.strip()
            if not rendered:
                raise ParseError("Natural-language rendering returned empty text")
            log_debug_event("nl_render.ok", attempt=attempt, chars=len(rendered))
            return rendered
        except Exception as exc:  # noqa: BLE001
            errors.append(f"attempt {attempt}: {exc}")
            log_error_event(
                "nl_render.fail", attempt=attempt, **summarize_exception(exc)
            )

    raise ParseError(
        "Failed to convert ELG hypothesis into natural language via LLM",
        errors=errors,
    )


def _validate_parser_payload(payload: Dict[str, Any]) -> None:
    """Validate that a payload matches the supported ELG JSON schema."""
    if not isinstance(payload, dict):
        raise ParseError("Parser payload must be a JSON object")

    kind = payload.get("kind")
    if kind not in {"atomic", "logical", "relation"}:
        raise ParseError(f"Unsupported node kind: {kind!r}")

    name = payload.get("name")

    if not isinstance(name, str) or not name.strip():
        raise ParseError(f"{kind.capitalize()} node must include a non-empty 'name'")

    if kind == "atomic":
        return

    inputs = payload.get("inputs")
    if not isinstance(inputs, list):
        raise ParseError(f"{kind} node must include list 'inputs'")

    if kind == "logical":
        if name not in {"AND", "OR", "NOT"}:
            raise ParseError(f"Unsupported logical operator: {name!r}")
        if name == "NOT" and len(inputs) != 1:
            raise ParseError("NOT node must contain exactly one input")
        if name in {"AND", "OR"} and len(inputs) < 2:
            raise ParseError(f"{name} node must contain at least two inputs")
    else:
        if name not in {"IMPLIES", "SUPPORT", "CONTRADICT", "CORRELATE"}:
            raise ParseError(f"Unsupported relation type: {name!r}")
        if len(inputs) != 2:
            raise ParseError("Relation node must contain exactly two inputs")

    for child in inputs:
        _validate_parser_payload(child)
