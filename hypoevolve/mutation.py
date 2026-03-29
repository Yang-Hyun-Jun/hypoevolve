from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from elg import AtomicNode, Hypothesis, MutationSample, generate_mutation_candidates
from hypoevolve.archive import ArchiveEntry
from hypoevolve.helper import build_steering_prompt_variables
from hypoevolve.llm import LLMClient
from hypoevolve.logger import logger
from hypoevolve.parser import JSON_RETRY_PROMPT, ParseError
from hypoevolve.prompts import load_and_render_prompt, load_prompt


@dataclass(slots=True)
class MutationDecision:
    selected_candidate_index: int
    reason: str
    mutation: MutationSample


STEERING_SYSTEM_PROMPT = load_prompt("steering", "system.md")


def steer_mutation(
    parent_hypothesis: Hypothesis,
    parent_hypothesis_nl: str,
    current_metrics: Mapping[str, object],
    llm: LLMClient,
    atomic_pool: Sequence[AtomicNode | str] | None = None,
    recent_history: Sequence[Mapping[str, object]] | None = None,
    top_hypotheses: Sequence[ArchiveEntry] | None = None,
    retries: int = 2,
) -> MutationDecision:
    candidates = generate_mutation_candidates(
        parent_hypothesis, atomic_pool=atomic_pool
    )
    if not candidates:
        raise ParseError("No legal mutation candidates available for steering")
    logger.info("generated {} mutation candidates for steering", len(candidates))

    base_user_prompt = load_and_render_prompt(
        "steering",
        "user.md",
        variables=build_steering_prompt_variables(
            parent_hypothesis=parent_hypothesis,
            parent_hypothesis_nl=parent_hypothesis_nl,
            current_metrics=current_metrics,
            mutation_candidates=candidates,
            recent_history=recent_history,
            top_hypotheses=top_hypotheses,
        ),
    )

    errors: list[str] = []
    attempts = retries + 1

    for attempt in range(1, attempts + 1):
        try:
            system_prompt = (
                STEERING_SYSTEM_PROMPT
                if attempt == 1
                else f"{STEERING_SYSTEM_PROMPT}\n\n{JSON_RETRY_PROMPT}"
            )
            user_prompt = base_user_prompt

            if errors:
                user_prompt = (
                    f"{base_user_prompt}\n\n"
                    "# Previous Attempt Failed\n\n"
                    f"{errors[-1]}\n\n"
                    "Return a corrected JSON object that selects exactly one valid candidate index."
                )

            payload = llm.generate_json(system_prompt, user_prompt, json_retries=0)
            index = int(payload["selected_candidate_index"])

            if index < 0 or index >= len(candidates):
                raise ParseError(f"Selected candidate index out of range: {index}")

            reason = str(payload.get("reason", "")).strip()
            if not reason:
                raise ParseError("Steering output must include a non-empty reason")
            logger.info("steering selected candidate {}", index)
            return MutationDecision(index, reason, candidates[index])

        except Exception as exc:  # noqa: BLE001
            errors.append(f"attempt {attempt}: {exc}")

    raise ParseError("Failed to steer mutation via LLM", errors=errors)
