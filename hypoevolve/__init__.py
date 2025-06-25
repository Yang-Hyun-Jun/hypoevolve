"""
Hypoevolve: A simplified and efficient implementation of the AlphaEvolve mechanism
"""

__version__ = "0.1.0"

from hypoevolve.core.controller import HypoEvolve
from hypoevolve.utils import configure_root_logger

# 패키지 로드 시 루트 로거 설정
configure_root_logger()

__all__ = ["HypoEvolve"]
