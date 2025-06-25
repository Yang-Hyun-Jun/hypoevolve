"""
Configuration management for HypoEvolve
"""

from dataclasses import dataclass
from typing import Dict, Optional, Any
import yaml
import os


@dataclass
class Config:
    """Main configuration for HypoEvolve"""

    # LLM Configuration
    api_key: str = ""
    api_base: str = "https://api.openai.com/v1"
    model: str = "gpt-4"
    temperature: float = 0.7
    max_tokens: int = 4096

    # Evolution Parameters
    max_iterations: int = 100
    population_size: int = 50
    elite_ratio: float = 0.2
    mutation_rate: float = 0.8

    # Evaluation Settings
    timeout: int = 30
    max_retries: int = 3

    # Output Settings
    output_dir: str = "hypoevolve_output"
    save_interval: int = 10

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return self.__dict__

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        """Create configuration from dictionary"""
        return cls(**{k: v for k, v in config_dict.items() if hasattr(cls, k)})

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load configuration from YAML file"""
        if not os.path.exists(path):
            return cls()

        with open(path, "r") as f:
            config_dict = yaml.safe_load(f) or {}

        return cls.from_dict(config_dict)


def load_config(config_path: Optional[str] = None) -> Config:
    """Load configuration with environment variable support"""
    config = Config()

    # Load from file if provided
    if config_path:
        config = Config.from_yaml(config_path)

    # Override with environment variables
    if os.getenv("OPENAI_API_KEY"):
        config.api_key = os.getenv("OPENAI_API_KEY")

    if os.getenv("OPENAI_API_BASE"):
        config.api_base = os.getenv("OPENAI_API_BASE")

    return config
