"""Configuration models and lightweight config loading for HypoEvolve."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

from .simple_yaml import SimpleYAMLError, ensure_mapping, parse_simple_yaml


class ConfigError(ValueError):
    """Raised when a HypoEvolve configuration file is invalid."""

    pass


@dataclass(slots=True)
class LLMConfig:
    """Settings for the LLM client used across the pipeline."""

    model: str = "deepseek/deepseek-v3.2"
    temperature: float = 0.2
    max_tokens: int = 2000
    api_key: str | None = None
    api_base: str = "https://openrouter.ai/api/v1"
    timeout: int = 60
    retries: int = 1
    retry_delay: float = 1.0


@dataclass(slots=True)
class ParserConfig:
    """Settings for natural-language parsing retries."""

    retries: int = 1


@dataclass(slots=True)
class EvaluatorConfig:
    """Settings for hypothesis evaluation against dataset-backed code."""

    dataset_schema_path: str = "dataset.yaml"
    parameters: Dict[str, Any] = field(default_factory=dict)
    seed: int = 42


@dataclass(slots=True)
class SearchConfig:
    """Settings for the outer search loop and steering behavior."""

    iterations: int = 5
    steering_retries: int = 2
    random_steering_prob: float = 0.2
    random_seed: int = 42


@dataclass(slots=True)
class ArchiveConfig:
    """Settings for archive bucketing and per-cell elite retention."""

    coverage_bins: List[float] = field(default_factory=lambda: [0.05, 0.15, 0.30])
    complexity_bins: List[int] = field(default_factory=lambda: [3, 5, 8])
    per_cell_top_k: int = 10


@dataclass(slots=True)
class OutputConfig:
    """Settings for runtime artifact output paths."""

    base_dir: str = ".hypoevolve/runs"
    top_k_evaluator_code_artifacts: int = 5


@dataclass(slots=True)
class LoggingConfig:
    """Settings for CLI and runtime logging output."""

    level: str = "INFO"


@dataclass(slots=True)
class WorkerConfig:
    """Settings for optional multi-process worker execution."""

    enabled: bool = False
    count: int = 1


@dataclass(slots=True)
class HypoEvolveConfig:
    """Top-level configuration bundle for a HypoEvolve run."""

    llm: LLMConfig = field(default_factory=LLMConfig)
    parser: ParserConfig = field(default_factory=ParserConfig)
    evaluator: EvaluatorConfig = field(default_factory=EvaluatorConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    archive: ArchiveConfig = field(default_factory=ArchiveConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    workers: WorkerConfig = field(default_factory=WorkerConfig)


def load_config(path: str | Path | None = None) -> HypoEvolveConfig:
    """Load a config file or return default settings when no path is given."""
    if path is None:
        return HypoEvolveConfig()

    config_path = Path(path)
    if not config_path.exists():
        raise ConfigError(f"Config file not found: {config_path}")

    try:
        raw = parse_simple_yaml(config_path.read_text(encoding="utf-8"))
    except SimpleYAMLError as exc:
        raise ConfigError(str(exc)) from exc
    return _config_from_dict(raw)


def _config_from_dict(data: Dict[str, Any]) -> HypoEvolveConfig:
    _ensure_mapping(data, "root")

    llm = LLMConfig(
        **_filter_known(
            data.get("llm", {}),
            {
                "model",
                "temperature",
                "max_tokens",
                "api_key",
                "api_base",
                "timeout",
                "retries",
                "retry_delay",
            },
        )
    )
    parser = ParserConfig(**_filter_known(data.get("parser", {}), {"retries"}))
    evaluator = EvaluatorConfig(
        **_filter_known(
            data.get("evaluator", {}),
            {"dataset_schema_path", "parameters", "seed"},
        )
    )
    search = SearchConfig(
        **_filter_known(
            data.get("search", {}),
            {
                "iterations",
                "steering_retries",
                "random_steering_prob",
                "random_seed",
            },
        )
    )
    archive = ArchiveConfig(
        **_filter_known(
            data.get("archive", {}),
            {"coverage_bins", "complexity_bins", "per_cell_top_k"},
        )
    )
    output = OutputConfig(
        **_filter_known(
            data.get("output", {}),
            {"base_dir", "top_k_evaluator_code_artifacts"},
        )
    )
    logging = LoggingConfig(**_filter_known(data.get("logging", {}), {"level"}))
    workers = WorkerConfig(
        **_filter_known(data.get("workers", {}), {"enabled", "count"})
    )

    if parser.retries < 0 or parser.retries > 2:
        raise ConfigError("parser.retries must be between 0 and 2 for MVP")
    if search.iterations < 1:
        raise ConfigError("search.iterations must be >= 1")
    if search.steering_retries < 0 or search.steering_retries > 3:
        raise ConfigError("search.steering_retries must be between 0 and 3")
    if search.random_steering_prob < 0.0 or search.random_steering_prob > 1.0:
        raise ConfigError("search.random_steering_prob must be between 0.0 and 1.0")
    if workers.count < 1:
        raise ConfigError("workers.count must be >= 1")
    if not evaluator.dataset_schema_path:
        raise ConfigError("evaluator.dataset_schema_path is required")
    _validate_archive_bins(archive.coverage_bins, archive.complexity_bins)
    if archive.per_cell_top_k < 1:
        raise ConfigError("archive.per_cell_top_k must be >= 1")
    if output.top_k_evaluator_code_artifacts < 1:
        raise ConfigError("output.top_k_evaluator_code_artifacts must be >= 1")

    return HypoEvolveConfig(
        llm=llm,
        parser=parser,
        evaluator=evaluator,
        search=search,
        archive=archive,
        output=output,
        logging=logging,
        workers=workers,
    )


def _validate_archive_bins(
    coverage_bins: List[float],
    complexity_bins: List[int],
) -> None:
    if not coverage_bins:
        raise ConfigError("archive.coverage_bins must not be empty")
    if not complexity_bins:
        raise ConfigError("archive.complexity_bins must not be empty")
    if coverage_bins != sorted(coverage_bins):
        raise ConfigError("archive.coverage_bins must be sorted ascending")
    if complexity_bins != sorted(complexity_bins):
        raise ConfigError("archive.complexity_bins must be sorted ascending")
    if any(not isinstance(value, (int, float)) for value in coverage_bins):
        raise ConfigError("archive.coverage_bins must contain only numeric values")
    if any(not 0.0 < float(value) < 1.0 for value in coverage_bins):
        raise ConfigError("archive.coverage_bins values must be between 0.0 and 1.0")
    if any(int(value) != value for value in complexity_bins):
        raise ConfigError("archive.complexity_bins must contain only integers")
    if any(int(value) <= 0 for value in complexity_bins):
        raise ConfigError("archive.complexity_bins values must be > 0")


def _filter_known(data: Any, allowed: set[str]) -> Dict[str, Any]:
    if data is None:
        return {}
    _ensure_mapping(data, "config section")
    unknown = set(data) - allowed
    if unknown:
        raise ConfigError(f"Unknown config keys: {sorted(unknown)}")
    return dict(data)


def _ensure_mapping(data: Any, name: str) -> None:
    try:
        ensure_mapping(data, name)
    except SimpleYAMLError as exc:
        raise ConfigError(str(exc)) from exc
