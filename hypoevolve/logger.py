"""Project logging setup with a Loguru-first fallback implementation."""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path
from typing import Any

try:
    from loguru import logger as _logger
except Exception:  # noqa: BLE001
    import logging

    class _FallbackLogger:
        def __init__(self):
            self._logger = logging.getLogger("hypoevolve")

        def remove(self) -> None:
            for handler in list(self._logger.handlers):
                self._logger.removeHandler(handler)
                try:
                    handler.close()
                except Exception:
                    pass

        def add(
            self,
            sink: Any,
            level: str = "INFO",
            format: str | None = None,
            **kwargs: Any,
        ) -> None:
            handler = (
                logging.StreamHandler(sink)
                if hasattr(sink, "write")
                else logging.FileHandler(sink)
            )
            handler.setLevel(getattr(logging, level.upper(), logging.INFO))
            handler.setFormatter(
                logging.Formatter("%(asctime)s | %(levelname)-5s | %(message)s")
            )
            self._logger.addHandler(handler)
            self._logger.setLevel(getattr(logging, level.upper(), logging.INFO))

        def __getattr__(self, name: str) -> Any:
            return getattr(self._logger, name)

        def _log(self, level: str, message: str, *args: Any) -> None:
            if args:
                try:
                    message = message.format(*args)
                except Exception:
                    pass
            getattr(self._logger, level.lower())(message)

        def debug(self, message: str, *args: Any) -> None:
            self._log("debug", message, *args)

        def info(self, message: str, *args: Any) -> None:
            self._log("info", message, *args)

        def warning(self, message: str, *args: Any) -> None:
            self._log("warning", message, *args)

        def error(self, message: str, *args: Any) -> None:
            self._log("error", message, *args)

    _logger = _FallbackLogger()


def configure_logger(level: str = "INFO", log_path: str | Path | None = None) -> Any:
    """Configure stderr logging and an optional file sink for the project."""
    _logger.remove()
    _logger.add(
        sys.stderr,
        level=level.upper(),
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <5}</level> | <level>{message}</level>",
        backtrace=False,
        diagnose=False,
    )
    if log_path is not None:
        path = Path(log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        _logger.add(
            str(path),
            level=level.upper(),
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <5} | {message}",
            backtrace=False,
            diagnose=False,
        )
    return _logger


logger = _logger

_KEYERROR_RE = re.compile(r"KeyError:\s*['\"]([^'\"]+)['\"]")
_EXIT_CODE_RE = re.compile(r"exit_code=(\d+)")


def compact_text(text: Any, max_len: int = 120) -> str:
    """Collapse whitespace and cap long text for compact logging."""
    rendered = " ".join(str(text).split())
    if len(rendered) <= max_len:
        return rendered
    return f"{rendered[: max_len - 1]}…"


def event_message(event: str, **fields: Any) -> str:
    """Render one compact logfmt-style event message."""
    parts = [f"event={event}"]
    for key, value in fields.items():
        if value is None:
            continue
        parts.append(f"{key}={_format_log_value(value)}")
    return " ".join(parts)


def log_info_event(event: str, **fields: Any) -> None:
    """Emit one compact info-level event."""
    logger.info(event_message(event, **fields))


def log_debug_event(event: str, **fields: Any) -> None:
    """Emit one compact debug-level event."""
    logger.debug(event_message(event, **fields))


def log_warning_event(event: str, **fields: Any) -> None:
    """Emit one compact warning-level event."""
    logger.warning(event_message(event, **fields))


def log_error_event(event: str, **fields: Any) -> None:
    """Emit one compact error-level event."""
    logger.error(event_message(event, **fields))


def summarize_hypothesis(hypothesis: Any) -> dict[str, Any]:
    """Return a compact structural summary for one hypothesis."""
    from elg import (
        count_atomics,
        count_logicals,
        count_nodes,
        count_relations,
        fingerprint,
        tree_depth,
    )

    root_name = getattr(hypothesis.root, "name", "")
    if hasattr(root_name, "value"):
        root_name = root_name.value
    return {
        "fp": fingerprint(hypothesis)[:12],
        "root": str(root_name),
        "nodes": count_nodes(hypothesis),
        "depth": tree_depth(hypothesis),
        "atomics": count_atomics(hypothesis),
        "logicals": count_logicals(hypothesis),
        "relations": count_relations(hypothesis),
    }


def summarize_exception(exc: Any, max_len: int = 160) -> dict[str, Any]:
    """Return compact exception fields for structured logging."""
    text = compact_text(exc, max_len=max_len)
    fields: dict[str, Any] = {
        "err_type": getattr(exc, "__class__", type(exc)).__name__,
        "err": text,
    }
    keyerror_match = _KEYERROR_RE.search(str(exc))
    if keyerror_match:
        fields["missing_col"] = keyerror_match.group(1)
    exit_code_match = _EXIT_CODE_RE.search(str(exc))
    if exit_code_match:
        fields["exit_code"] = int(exit_code_match.group(1))
    return fields


def summarize_metrics(metrics: dict[str, object] | None) -> dict[str, Any]:
    """Return the main scalar metrics for compact event logs."""
    metrics = metrics or {}
    return {
        "score": metrics.get("combined_score"),
        "prec": metrics.get("precision"),
        "base": metrics.get("baseline"),
        "cov": metrics.get("coverage"),
        "up": metrics.get("uplift"),
    }


def _format_log_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return "nan"
        rendered = f"{value:.6f}".rstrip("0").rstrip(".")
        return rendered if rendered else "0"
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple, dict)):
        return _quote_if_needed(compact_text(json.dumps(value, ensure_ascii=False)))
    return _quote_if_needed(compact_text(value))


def _quote_if_needed(value: str) -> str:
    if not value:
        return '""'
    if any(ch.isspace() for ch in value) or any(ch in value for ch in ['"', "="]):
        return json.dumps(value, ensure_ascii=False)
    return value


configure_logger()
