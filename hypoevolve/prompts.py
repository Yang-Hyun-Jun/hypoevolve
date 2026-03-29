from __future__ import annotations

from pathlib import Path
from typing import Mapping

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


def load_prompt(*parts: str) -> str:
    path = PROMPTS_DIR.joinpath(*parts)
    return path.read_text(encoding="utf-8").strip()


def render_prompt(template: str, variables: Mapping[str, object]) -> str:
    rendered = template
    for key, value in variables.items():
        rendered = rendered.replace(f"{{{{{key}}}}}", str(value))
    return rendered


def load_and_render_prompt(*parts: str, variables: Mapping[str, object]) -> str:
    return render_prompt(load_prompt(*parts), variables)
