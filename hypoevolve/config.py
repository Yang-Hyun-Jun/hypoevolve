from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple


class ConfigError(ValueError):
    pass


@dataclass(slots=True)
class LLMConfig:
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
    retries: int = 1


@dataclass(slots=True)
class EvaluatorConfig:
    dataset_schema_path: str = "dataset.yaml"
    parameters: Dict[str, Any] = field(default_factory=dict)
    seed: int = 42


@dataclass(slots=True)
class SearchConfig:
    iterations: int = 5
    parent_explore_prob: float = 0.3
    steering_retries: int = 2
    random_steering_prob: float = 0.2
    random_seed: int = 42


@dataclass(slots=True)
class ArchiveConfig:
    top_k: int = 5


@dataclass(slots=True)
class OutputConfig:
    base_dir: str = ".hypoevolve/runs"


@dataclass(slots=True)
class LoggingConfig:
    level: str = "INFO"


@dataclass(slots=True)
class WorkerConfig:
    enabled: bool = False
    count: int = 1


@dataclass(slots=True)
class HypoEvolveConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    parser: ParserConfig = field(default_factory=ParserConfig)
    evaluator: EvaluatorConfig = field(default_factory=EvaluatorConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    archive: ArchiveConfig = field(default_factory=ArchiveConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    workers: WorkerConfig = field(default_factory=WorkerConfig)


def load_config(path: str | Path | None = None) -> HypoEvolveConfig:
    if path is None:
        return HypoEvolveConfig()

    config_path = Path(path)
    if not config_path.exists():
        raise ConfigError(f"Config file not found: {config_path}")

    raw = _parse_simple_yaml(config_path.read_text(encoding="utf-8"))
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
                "parent_explore_prob",
                "steering_retries",
                "random_steering_prob",
                "random_seed",
            },
        )
    )
    archive = ArchiveConfig(**_filter_known(data.get("archive", {}), {"top_k"}))
    output = OutputConfig(**_filter_known(data.get("output", {}), {"base_dir"}))
    logging = LoggingConfig(**_filter_known(data.get("logging", {}), {"level"}))
    workers = WorkerConfig(
        **_filter_known(data.get("workers", {}), {"enabled", "count"})
    )

    if archive.top_k != 5:
        raise ConfigError("archive.top_k must remain 5 in MVP")
    if parser.retries < 0 or parser.retries > 2:
        raise ConfigError("parser.retries must be between 0 and 2 for MVP")
    if search.iterations < 1:
        raise ConfigError("search.iterations must be >= 1")
    if search.steering_retries < 0 or search.steering_retries > 3:
        raise ConfigError("search.steering_retries must be between 0 and 3")
    if search.parent_explore_prob < 0.0 or search.parent_explore_prob > 1.0:
        raise ConfigError("search.parent_explore_prob must be between 0.0 and 1.0")
    if search.random_steering_prob < 0.0 or search.random_steering_prob > 1.0:
        raise ConfigError("search.random_steering_prob must be between 0.0 and 1.0")
    if workers.count < 1:
        raise ConfigError("workers.count must be >= 1")
    if not evaluator.dataset_schema_path:
        raise ConfigError("evaluator.dataset_schema_path is required")

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


def _filter_known(data: Any, allowed: set[str]) -> Dict[str, Any]:
    if data is None:
        return {}
    _ensure_mapping(data, "config section")
    unknown = set(data) - allowed
    if unknown:
        raise ConfigError(f"Unknown config keys: {sorted(unknown)}")
    return dict(data)


def _ensure_mapping(data: Any, name: str) -> None:
    if not isinstance(data, dict):
        raise ConfigError(f"{name} must be a mapping")


def _parse_simple_yaml(text: str) -> Dict[str, Any]:
    lines: List[Tuple[int, str]] = []
    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        lines.append((indent, raw_line.strip()))

    if not lines:
        return {}

    index, result = _parse_block(lines, 0, 0)
    if index != len(lines):
        raise ConfigError("Failed to parse configuration fully")
    if not isinstance(result, dict):
        raise ConfigError("Top-level config must be a mapping")
    return result


def _parse_block(
    lines: List[Tuple[int, str]], index: int, indent: int
) -> Tuple[int, Any]:
    container: Any = None

    while index < len(lines):
        current_indent, text = lines[index]
        if current_indent < indent:
            break
        if current_indent > indent:
            raise ConfigError(f"Unexpected indentation near: {text}")

        if text == "-" or text.startswith("- "):
            if container is None:
                container = []
            elif not isinstance(container, list):
                raise ConfigError("Cannot mix list and mapping items at same level")
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
            raise ConfigError("Cannot mix mapping and list items at same level")

        if ":" not in text:
            raise ConfigError(f"Invalid mapping line: {text}")
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
