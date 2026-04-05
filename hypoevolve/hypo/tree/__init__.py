"""Lazy exports for hypo tree utilities."""

from __future__ import annotations

__all__ = ["HypoTree", "HypoTreeGenerator"]


def __getattr__(name: str):
    if name == "HypoTree":
        from .base import HypoTree

        return HypoTree
    if name == "HypoTreeGenerator":
        from .generator import HypoTreeGenerator

        return HypoTreeGenerator
    raise AttributeError(name)
