"""LLM-guided mutation steering for measurable ELG hypotheses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from hypoevolve.archive import ArchiveEntry
from hypoevolve.elg import Hypothesis, hypothesis_from_dict, normalize_hypothesis
from hypoevolve.helper import build_steering_prompt_variables
from hypoevolve.llm import LLMClient
from hypoevolve.logger import compact_text, log_info_event
from hypoevolve.parser import JSON_RETRY_PROMPT, ParseError
from hypoevolve.prompts import load_and_render_prompt, load_prompt


@dataclass(slots=True)
class MutationDecision:
    """Bundle the child hypothesis and reasoning returned by steering."""

    child_hypothesis: Hypothesis
    domain_reason: str
    score_reason: str
    operation_score_rankings: dict[str, int]
    mutation_summary: str


STEERING_SYSTEM_PROMPT = load_prompt("steering", "system.md")
STEERING_RANDOM_SYSTEM_PROMPT = load_prompt("steering-random", "system.md")


def steer_mutation(
    parent_hypothesis: Hypothesis,
    current_metrics: Mapping[str, object],
    llm: LLMClient,
    recent_history: Sequence[Mapping[str, object]] | None = None,
    top_hypotheses: Sequence[ArchiveEntry] | None = None,
    use_random_steering: bool = False,
    retries: int = 2,
) -> MutationDecision:
    """Ask the LLM for a locally mutated child hypothesis and rationale."""
    prompt_dir = "steering-random" if use_random_steering else "steering"
    system_prompt_template = (
        STEERING_RANDOM_SYSTEM_PROMPT if use_random_steering else STEERING_SYSTEM_PROMPT
    )
    base_user_prompt = load_and_render_prompt(
        prompt_dir,
        "user.md",
        variables=build_steering_prompt_variables(
            parent_hypothesis=parent_hypothesis,
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
                    "Return a corrected JSON object with non-empty `domain_reason`, `score_reason`, `operation_score_rankings`, `child_hypothesis`, and `mutation_summary` fields."
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
            domain_reason = str(payload.get("domain_reason", "")).strip()
            score_reason = str(payload.get("score_reason", "")).strip()
            if not use_random_steering:
                if not domain_reason:
                    raise ParseError(
                        "Steering output must include a non-empty domain_reason"
                    )
                if not score_reason:
                    raise ParseError(
                        "Steering output must include a non-empty score_reason"
                    )
                operation_score_rankings = payload.get("operation_score_rankings")
                if (
                    not isinstance(operation_score_rankings, dict)
                    or not operation_score_rankings
                ):
                    raise ParseError(
                        "Steering output must include a non-empty operation_score_rankings dictionary"
                    )
                normalized_rankings: dict[str, int] = {}
                for key, value in operation_score_rankings.items():
                    if not isinstance(key, str) or not key.strip():
                        raise ParseError(
                            "operation_score_rankings keys must be non-empty strings"
                        )
                    try:
                        rank = int(value)
                    except Exception as exc:  # noqa: BLE001
                        raise ParseError(
                            f"operation_score_rankings value for {key!r} must be an integer"
                        ) from exc
                    if rank < 1:
                        raise ParseError(
                            f"operation_score_rankings value for {key!r} must be >= 1"
                        )
                    normalized_rankings[key.strip()] = rank
            else:
                normalized_rankings = {}

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
                hypothesis_from_dict({"root": child_root})
            )

            log_info_event(
                "steer.ok",
                mode="random" if use_random_steering else "guided",
                summary=compact_text(mutation_summary, max_len=96),
            )
            return MutationDecision(
                child_hypothesis=child_hypothesis,
                domain_reason=domain_reason,
                score_reason=score_reason,
                operation_score_rankings=normalized_rankings,
                mutation_summary=mutation_summary,
            )

        except Exception as exc:  # noqa: BLE001
            errors.append(f"attempt {attempt}: {exc}")

    raise ParseError("Failed to steer mutation via LLM", errors=errors)
