from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from elg import Hypothesis, hypothesis_from_dict, normalize_hypothesis
from hypoevolve.archive import ArchiveEntry
from hypoevolve.helper import build_steering_prompt_variables
from hypoevolve.llm import LLMClient
from hypoevolve.logger import logger
from hypoevolve.parser import JSON_RETRY_PROMPT, ParseError
from hypoevolve.prompts import load_and_render_prompt, load_prompt


@dataclass(slots=True)
class MutationDecision:
    child_hypothesis: Hypothesis
    reason: str
    mutation_summary: str


STEERING_SYSTEM_PROMPT = load_prompt("steering", "system.md")
STEERING_RANDOM_SYSTEM_PROMPT = load_prompt("steering-random", "system.md")


def steer_mutation(
    parent_hypothesis: Hypothesis,
    parent_hypothesis_nl: str,
    current_metrics: Mapping[str, object],
    llm: LLMClient,
    recent_history: Sequence[Mapping[str, object]] | None = None,
    top_hypotheses: Sequence[ArchiveEntry] | None = None,
    use_random_steering: bool = False,
    retries: int = 2,
) -> MutationDecision:
    prompt_dir = "steering-random" if use_random_steering else "steering"
    system_prompt_template = (
        STEERING_RANDOM_SYSTEM_PROMPT if use_random_steering else STEERING_SYSTEM_PROMPT
    )
    base_user_prompt = load_and_render_prompt(
        prompt_dir,
        "user.md",
        variables=build_steering_prompt_variables(
            parent_hypothesis=parent_hypothesis,
            parent_hypothesis_nl=parent_hypothesis_nl,
            current_metrics=current_metrics,
            recent_history=recent_history,
            top_hypotheses=top_hypotheses,
        ),
    )

    errors: list[str] = []
    attempts = retries + 1

    for attempt in range(1, attempts + 1):
        try:
            system_prompt = (
                system_prompt_template
                if attempt == 1
                else f"{system_prompt_template}\n\n{JSON_RETRY_PROMPT}"
            )
            user_prompt = base_user_prompt

            if errors:
                correction_text = (
                    "Return a corrected JSON object with non-empty `child_hypothesis`, `reason`, and `mutation_summary` fields."
                    if not use_random_steering
                    else "Return a corrected JSON object with non-empty `child_hypothesis` and `mutation_summary` fields."
                )
                user_prompt = (
                    f"{base_user_prompt}\n\n"
                    "# Previous Attempt Failed\n\n"
                    f"{errors[-1]}\n\n"
                    f"{correction_text}"
                )

            payload = llm.generate_json(system_prompt, user_prompt, json_retries=0)
            reason = str(payload.get("reason", "")).strip()
            if not use_random_steering and not reason:
                raise ParseError("Steering output must include a non-empty reason")

            mutation_summary = str(payload.get("mutation_summary", "")).strip()
            if not mutation_summary:
                raise ParseError(
                    "Steering output must include a non-empty mutation_summary"
                )

            child_root = payload.get("child_hypothesis")
            if not isinstance(child_root, dict):
                raise ParseError(
                    "Steering output must include child_hypothesis as a JSON object"
                )

            child_hypothesis = normalize_hypothesis(
                hypothesis_from_dict(
                    {
                        "root": child_root,
                        "params": dict(parent_hypothesis.params),
                    }
                )
            )
            
            logger.info("steering proposed child hypothesis successfully")
            return MutationDecision(
                child_hypothesis=child_hypothesis,
                reason=reason,
                mutation_summary=mutation_summary,
            )

        except Exception as exc:  # noqa: BLE001
            errors.append(f"attempt {attempt}: {exc}")

    raise ParseError("Failed to steer mutation via LLM", errors=errors)
