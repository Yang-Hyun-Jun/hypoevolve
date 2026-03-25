from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from elg import Hypothesis, hypothesis_from_dict, normalize_hypothesis
from hypoevolve.llm import LLMClient
from hypoevolve.prompts import load_prompt


class ParseError(ValueError):
    def __init__(self, message: str, errors: Optional[List[str]] = None):
        super().__init__(message)
        self.errors = errors or []


ParserCallable = Callable[[str], Hypothesis]


PARSER_SYSTEM_PROMPT = load_prompt("parser", "system.md")
PARSER_RETRY_PROMPT = load_prompt("parser", "retry.md")


def parse_hypothesis_text(
    text: str,
    llm: LLMClient,
    retries: int = 1,
) -> Hypothesis:
    return llm_parse_hypothesis(text, llm=llm, retries=retries)


def llm_parse_hypothesis(
    text: str,
    llm: LLMClient,
    retries: int = 2,
) -> Hypothesis:
    if not text or not text.strip():
        raise ParseError("Hypothesis text must be non-empty")

    errors: List[str] = []
    attempts = retries + 1
    for attempt in range(1, attempts + 1):
        try:
            system_prompt = (
                PARSER_SYSTEM_PROMPT
                if attempt == 1
                else f"{PARSER_SYSTEM_PROMPT}\n\n{PARSER_RETRY_PROMPT}"
            )
            payload = llm.generate_json(
                system_prompt,
                f"Convert this natural-language hypothesis into ELG JSON root node:\\n\\n{text}",
                json_retries=0,
            )
            _validate_parser_payload(payload)
            hypothesis = hypothesis_from_dict({"root": payload, "params": {}})
            return normalize_hypothesis(hypothesis)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"attempt {attempt}: {exc}")
    raise ParseError(
        "Failed to convert natural-language hypothesis to ELG via LLM", errors=errors
    )
def _validate_parser_payload(payload: Dict[str, Any]) -> None:
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
