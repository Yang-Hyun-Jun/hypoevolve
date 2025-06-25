"""
HypoEvolve 로거 설정 유틸리티
전역에서 사용할 수 있는 로거 설정 함수 제공
"""

import logging
from typing import Optional

# 전역 설정 플래그
_root_logger_configured = False


def setup_logger(
    name: str, level: int = logging.DEBUG, formatter_string: Optional[str] = None
) -> logging.Logger:
    """
    HypoEvolve용 로거를 설정하고 반환합니다.

    Args:
        name: 로거 이름 (보통 __name__ 사용)
        level: 로그 레벨 (기본값: DEBUG)
        formatter_string: 커스텀 포매터 문자열 (선택사항)

    Returns:
        logging.Logger: 설정된 로거 객체
    """
    global _root_logger_configured

    # 루트 로거가 설정되지 않았다면 먼저 설정
    if not _root_logger_configured:
        configure_root_logger()

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # propagate를 True로 설정하여 루트 로거의 핸들러를 사용
    logger.propagate = True

    # 개별 핸들러는 추가하지 않음 (루트 로거의 핸들러 사용)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    기본 설정으로 로거를 가져옵니다.

    Args:
        name: 로거 이름 (보통 __name__ 사용)

    Returns:
        logging.Logger: 설정된 로거 객체
    """
    return setup_logger(name)


def configure_root_logger(level: int = logging.DEBUG) -> None:
    """
    루트 로거를 설정합니다.

    Args:
        level: 루트 로거 레벨 (기본값: DEBUG)
    """
    global _root_logger_configured

    if _root_logger_configured:
        return

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # 기존 핸들러 모두 제거
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # 새로운 StreamHandler 추가
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - [%(levelname)s] %(name)s - %(message)s"
    )
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)

    # 외부 라이브러리 로거들의 레벨을 조정하여 상세한 로그 숨기기
    _configure_external_loggers()

    _root_logger_configured = True


def _configure_external_loggers() -> None:
    """외부 라이브러리 로거들의 레벨을 조정합니다."""
    # HTTP 관련 라이브러리들의 상세한 DEBUG 로그 숨기기
    external_loggers = [
        "httpx",
        "httpcore",
        "httpcore.http11",
        "httpcore.connection",
        "openai._base_client",
        "urllib3.connectionpool",
        "requests.packages.urllib3.connectionpool",
    ]

    for logger_name in external_loggers:
        external_logger = logging.getLogger(logger_name)
        external_logger.setLevel(logging.WARNING)  # WARNING 이상만 출력


def set_external_log_level(level: int = logging.WARNING) -> None:
    """
    외부 라이브러리 로그 레벨을 설정합니다.

    Args:
        level: 외부 라이브러리 로그 레벨 (기본값: WARNING)
    """
    external_loggers = [
        "httpx",
        "httpcore",
        "httpcore.http11",
        "httpcore.connection",
        "openai._base_client",
        "urllib3.connectionpool",
        "requests.packages.urllib3.connectionpool",
    ]

    for logger_name in external_loggers:
        external_logger = logging.getLogger(logger_name)
        external_logger.setLevel(level)
