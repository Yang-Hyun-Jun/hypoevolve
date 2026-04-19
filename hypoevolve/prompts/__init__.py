"""Small helpers for loading and rendering packaged markdown prompt templates."""

from __future__ import annotations

from importlib import resources
from typing import Mapping

PROMPTS_DIR = resources.files("hypoevolve.prompts")


def load_prompt(*parts: str) -> str:
    """Load one prompt file relative to the packaged prompt directory."""
    path = PROMPTS_DIR.joinpath(*parts)
    return path.read_text(encoding="utf-8").strip()


def render_prompt(template: str, variables: Mapping[str, object]) -> str:
    """Render a prompt template by replacing ``{{NAME}}`` placeholders."""
    rendered = template
    for key, value in variables.items():
        rendered = rendered.replace(f"{{{{{key}}}}}", str(value))
    return rendered


def load_and_render_prompt(*parts: str, variables: Mapping[str, object]) -> str:
    """Load a prompt file and render it with the provided variables."""
    return render_prompt(load_prompt(*parts), variables)
