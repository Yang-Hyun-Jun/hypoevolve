from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from hypoevolve.hypo.tree.base import HypoTree
from hypoevolve.llm import LLMClient
from hypoevolve.logger import compact_text, log_error_event, log_info_event, summarize_exception
from hypoevolve.prompts import load_and_render_prompt, load_prompt


class HypothesisGenerationError(ValueError):
    """Raised when an LLM response cannot be converted into one hypothesis."""

    def __init__(self, message: str, errors: Optional[List[str]] = None):
        super().__init__(message)
        self.errors = errors or []


@dataclass(slots=True)
class TreePairHypothesis:
    """Bundle the generated tree pair, prompt payload, and hypothesis text."""

    tree_a: HypoTree
    tree_b: HypoTree
    hypothesis: str


HYPO_SYSTEM_PROMPT = load_prompt("hypo", "system.md")
HYPO_OUTPUT_RETRY_PROMPT = (
    "Your previous output did not follow the required format. "
    "Return exactly one <hypothesis>...</hypothesis> block and nothing else."
)


def build_hypothesis_prompt_variables(
    tree_a: HypoTree,
    tree_b: HypoTree,
) -> dict[str, str]:
    """Build prompt variables for one feature-tree hypothesis generation call."""
    tree_a_str = str(tree_a.render(return_str=True)).strip()
    tree_b_str = str(tree_b.render(return_str=True)).strip()
    desc_a = str(tree_a.get_node_descriptions()).strip()
    desc_b = str(tree_b.get_node_descriptions()).strip()
    node_desc_str = "\n".join(part for part in [desc_a, desc_b] if part)

    return {
        "TREE_A": tree_a_str,
        "TREE_B": tree_b_str,
        "NODE_DESCRIPTIONS": node_desc_str,
    }


def llm_generate_hypothesis_from_trees(
    tree_a: HypoTree,
    tree_b: HypoTree,
    llm: LLMClient,
    retries: int = 2,
    **kwargs,
) -> str:
    """Generate one natural-language hypothesis from two feature trees."""
    variables = build_hypothesis_prompt_variables(
        tree_a=tree_a,
        tree_b=tree_b,
    )
    user_prompt = load_and_render_prompt("hypo", "user.md", variables=variables)
    errors: List[str] = []
    attempts = retries + 1
    log_info_event("tree_hypothesis.start", attempts=attempts)

    for attempt in range(1, attempts + 1):
        try:
            log_info_event("tree_hypothesis.attempt", attempt=attempt)
            system_prompt = (
                HYPO_SYSTEM_PROMPT
                if attempt == 1
                else f"{HYPO_SYSTEM_PROMPT}\n\n{HYPO_OUTPUT_RETRY_PROMPT}"
            )
            response = llm.generate_text(system_prompt, user_prompt, **kwargs)
            hypothesis = _extract_hypothesis_text(response)
            if not hypothesis:
                raise HypothesisGenerationError(
                    "LLM returned an empty hypothesis payload"
                )
            log_info_event(
                "tree_hypothesis.ok",
                attempt=attempt,
                chars=len(hypothesis),
                preview=compact_text(hypothesis, max_len=96),
            )
            return hypothesis
        except Exception as exc:  # noqa: BLE001
            errors.append(f"attempt {attempt}: {exc}")
            log_error_event(
                "tree_hypothesis.fail",
                attempt=attempt,
                **summarize_exception(exc),
            )

    raise HypothesisGenerationError(
        "Failed to generate a hypothesis from feature trees via LLM",
        errors=errors,
    )


def generate_random_tree_pair_hypothesis(
    llm: LLMClient,
    max_depth: int = 3,
    generator=None,
    dataset_schema_path: str | Path = "dataset.yaml",
    retries: int = 2,
    **kwargs,
) -> TreePairHypothesis:
    """Generate two random feature trees and synthesize one hypothesis from them."""
    if generator is None:
        get_tree_generator, generate_trees = _load_tree_generation_helpers()
        generator = get_tree_generator(dataset_schema_path=dataset_schema_path)
    else:
        _, generate_trees = _load_tree_generation_helpers()

    trees = generate_trees(generator, max_depth=max_depth, num_trees=2)
    tree_a, tree_b = trees[0], trees[1]
    hypothesis = llm_generate_hypothesis_from_trees(
        tree_a=tree_a,
        tree_b=tree_b,
        llm=llm,
        retries=retries,
        **kwargs,
    )
    return TreePairHypothesis(
        tree_a=tree_a,
        tree_b=tree_b,
        hypothesis=hypothesis,
    )


def _extract_hypothesis_text(response: str) -> str:
    """Extract the content of one <hypothesis> block, falling back to raw text."""
    stripped = str(response).strip()
    match = re.search(r"<hypothesis>\s*(.*?)\s*</hypothesis>", stripped, re.DOTALL)
    if match:
        return match.group(1).strip()
    return stripped


def _load_tree_generation_helpers() -> Tuple[Callable, Callable]:
    """Lazily import tree-generation helpers to avoid eager optional deps."""
    from hypoevolve.hypo.helper import generate_trees, get_tree_generator

    return get_tree_generator, generate_trees
