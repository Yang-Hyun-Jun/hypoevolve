from __future__ import annotations

from pathlib import Path

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


def load_prompt(*parts: str) -> str:
    path = PROMPTS_DIR.joinpath(*parts)
    return path.read_text(encoding="utf-8").strip()
