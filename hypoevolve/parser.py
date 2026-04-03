"""Natural-language parsing and ELG validation helpers."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from elg import Hypothesis, hypothesis_from_dict, normalize_hypothesis
from hypoevolve.llm import LLMClient
from hypoevolve.logger import logger
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
    logger.info("parser started")
    for attempt in range(1, attempts + 1):
        try:
            logger.info("parser attempt {}", attempt)
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
            hypothesis = hypothesis_from_dict({"root": payload, "params": {}})
            logger.info("parser succeeded on attempt {}", attempt)
            return normalize_hypothesis(hypothesis)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"attempt {attempt}: {exc}")
            logger.error("parser attempt {} failed: {}", attempt, exc)
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
    logger.info("measurable conversion started")

    for attempt in range(1, attempts + 1):
        try:
            logger.info("measurable conversion attempt {}", attempt)
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
            measurable = hypothesis_from_dict(
                {"root": payload, "params": dict(hypothesis.params)}
            )
            logger.info("measurable conversion succeeded on attempt {}", attempt)
            return normalize_hypothesis(measurable)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"attempt {attempt}: {exc}")
            logger.error("measurable conversion attempt {} failed: {}", attempt, exc)

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
    logger.info("natural-language rendering started")

    for attempt in range(1, attempts + 1):
        try:
            logger.info("natural-language rendering attempt {}", attempt)
            system_prompt = NL_SYSTEM_PROMPT
            response = llm.generate_text(
                system_prompt,
                "Convert this ELG hypothesis into natural language:\n\n"
                f"{hypothesis.to_dict()['root']}",
            )
            rendered = response.strip()
            if not rendered:
                raise ParseError("Natural-language rendering returned empty text")
            logger.info("natural-language rendering succeeded on attempt {}", attempt)
            return rendered
        except Exception as exc:  # noqa: BLE001
            errors.append(f"attempt {attempt}: {exc}")
            logger.error("natural-language rendering attempt {} failed: {}", attempt, exc)

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

    if kind == "atomic":
        if not isinstance(payload.get("name"), str) or not payload["name"].strip():
            raise ParseError("Atomic node must include a non-empty 'name'")
        return

    inputs = payload.get("inputs")
    if not isinstance(inputs, list):
        raise ParseError(f"{kind} node must include list 'inputs'")

    if kind == "logical":
        op = payload.get("op")
        if op not in {"AND", "OR", "NOT"}:
            raise ParseError(f"Unsupported logical operator: {op!r}")
        if op == "NOT" and len(inputs) != 1:
            raise ParseError("NOT node must contain exactly one input")
        if op in {"AND", "OR"} and len(inputs) < 2:
            raise ParseError(f"{op} node must contain at least two inputs")
    else:
        relation_type = payload.get("type")
        if relation_type not in {"IMPLIES", "SUPPORT", "CONTRADICT", "CORRELATE"}:
            raise ParseError(f"Unsupported relation type: {relation_type!r}")
        if len(inputs) != 2:
            raise ParseError("Relation node must contain exactly two inputs")

    for child in inputs:
        _validate_parser_payload(child)
