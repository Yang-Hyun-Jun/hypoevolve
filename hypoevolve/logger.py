"""Project logging setup with a Loguru-first fallback implementation."""

from __future__ import annotations

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

        def add(self, sink: Any, level: str = "INFO", format: str | None = None, **kwargs: Any) -> None:
            handler = logging.StreamHandler(sink) if hasattr(sink, "write") else logging.FileHandler(sink)
            handler.setLevel(getattr(logging, level.upper(), logging.INFO))
            handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s"))
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
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        backtrace=False,
        diagnose=False,
    )
    if log_path is not None:
        path = Path(log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        _logger.add(
            str(path),
            level=level.upper(),
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            backtrace=False,
            diagnose=False,
        )
    return _logger


logger = _logger
