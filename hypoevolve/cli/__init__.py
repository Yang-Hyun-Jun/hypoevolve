"""CLI package for HypoEvolve."""

from __future__ import annotations

import importlib.metadata as metadata  # noqa: F401

import click  # noqa: F401

from hypoevolve.core.config import ConfigError  # noqa: F401

from .app import app, main  # noqa: F401
from .display import (  # noqa: F401
    _detect_version,
    _echo_banner,
    _echo_block,
    _echo_error,
    _echo_json,
    _echo_metric_highlights,
    _render_banner,
    _render_kv_section,
)
from .runs import (  # noqa: F401
    _latest_run_dir,
    _read_json_if_exists,
    _resolve_runs_base_dir,
    _run_dir_from_id,
    _status_payload,
)
