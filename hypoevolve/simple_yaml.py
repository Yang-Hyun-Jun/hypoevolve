"""Minimal YAML-like parsing helpers used by config and dataset loaders."""

from __future__ import annotations

from typing import Any


class SimpleYAMLError(ValueError):
    """Raised when the lightweight YAML parser encounters invalid input."""

    pass


def ensure_mapping(data: Any, name: str) -> None:
    """Require that a parsed object is a mapping."""
    if not isinstance(data, dict):
        raise SimpleYAMLError(f"{name} must be a mapping")


def parse_simple_yaml(text: str) -> dict[str, Any]:
    """Parse a small YAML subset into Python dict/list/scalar structures."""
    lines: list[tuple[int, str]] = []
    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        lines.append((indent, raw_line.strip()))

    if not lines:
        return {}

    index, result = _parse_block(lines, 0, 0)
    if index != len(lines):
        raise SimpleYAMLError("Failed to parse configuration fully")
    if not isinstance(result, dict):
        raise SimpleYAMLError("Top-level config must be a mapping")
    return result


def _parse_block(
    lines: list[tuple[int, str]], index: int, indent: int
) -> tuple[int, Any]:
    container: Any = None

    while index < len(lines):
        current_indent, text = lines[index]
        if current_indent < indent:
            break
        if current_indent > indent:
            raise SimpleYAMLError(f"Unexpected indentation near: {text}")

        if text == "-" or text.startswith("- "):
            if container is None:
                container = []
            elif not isinstance(container, list):
                raise SimpleYAMLError("Cannot mix list and mapping items at same level")
            value_text = "" if text == "-" else text[2:].strip()
            if not value_text:
                index, value = _parse_block(lines, index + 1, indent + 2)
            else:
                value = _parse_scalar(value_text)
                index += 1
            container.append(value)
            continue

        if container is None:
            container = {}
        elif not isinstance(container, dict):
            raise SimpleYAMLError("Cannot mix mapping and list items at same level")

        if ":" not in text:
            raise SimpleYAMLError(f"Invalid mapping line: {text}")
        key, rest = text.split(":", 1)
        key = key.strip()
        rest = rest.strip()
        if rest:
            container[key] = _parse_scalar(rest)
            index += 1
        else:
            index, value = _parse_block(lines, index + 1, indent + 2)
            container[key] = value

    if container is None:
        container = {}
    return index, container


def _parse_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none"}:
        return None
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value
