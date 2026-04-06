"""Thin OpenAI-compatible client wrappers for text and JSON generation."""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List
from urllib.parse import urlparse

from .config import LLMConfig


class LLMError(RuntimeError):
    """Raised when the configured LLM client cannot produce a valid response."""

    pass


class LLMClient:
    """Wrap an OpenAI-compatible chat-completions client for project use."""

    def __init__(self, config: LLMConfig):
        """Initialize one reusable LLM client wrapper.

        Args:
            config: The LLM configuration to use for requests.

        Returns:
            None.
        """
        self.config = config
        self.api_base = config.api_base
        self.api_key = _resolve_api_key(config.api_key, self.api_base)
        self._client = None

    def generate_text(self, system: str, user: str, **kwargs: Any) -> str:
        """Generate free-form text with retry handling."""
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        retries = kwargs.get("retries", self.config.retries)
        retry_delay = kwargs.get("retry_delay", self.config.retry_delay)
        last_error: Exception | None = None

        for attempt in range(retries + 1):
            try:
                return self._call_openai(messages, **kwargs)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt >= retries:
                    break
                time.sleep(retry_delay)

        raise LLMError(f"LLM text generation failed: {last_error}")

    def generate_json(self, system: str, user: str, **kwargs: Any) -> Dict[str, Any]:
        """Generate a JSON object by parsing the model's text output."""
        text = self.generate_text(system, user, **kwargs)
        payload = _extract_json_payload(text)
        retries = kwargs.get("json_retries", 0)
        retry_delay = kwargs.get("retry_delay", self.config.retry_delay)
        last_error: Exception | None = None

        for attempt in range(retries + 1):
            try:
                return json.loads(payload)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt >= retries:
                    break
                time.sleep(retry_delay)
                text = self.generate_text(system, user, **kwargs)
                payload = _extract_json_payload(text)

        raise LLMError(f"LLM JSON generation failed: {last_error}")

    def _call_openai(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        """Submit one chat-completions request.

        Args:
            messages: The chat message list sent to the model.
            **kwargs: Optional request overrides such as model or temperature.

        Returns:
            str: The text content returned by the model.
        """
        client = self._get_client()
        response = client.chat.completions.create(
            model=kwargs.get("model", self.config.model),
            messages=messages,
            temperature=kwargs.get("temperature", self.config.temperature),
            max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
        )
        content = response.choices[0].message.content
        if content is None:
            raise LLMError("LLM returned empty content")
        return str(content)

    def _get_client(self) -> Any:
        """Lazily construct and cache the OpenAI client.

        Args:
            None.

        Returns:
            Any: The initialized OpenAI client instance.
        """
        if self._client is not None:
            return self._client
        if not self.api_key:
            raise LLMError("No API key configured for LLM client")
        try:
            import openai
        except Exception as exc:  # noqa: BLE001
            raise LLMError(f"OpenAI client import failed: {exc}") from exc

        self._client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.api_base,
            timeout=self.config.timeout,
            max_retries=0,
        )
        return self._client


def _resolve_api_key(explicit_api_key: str | None, api_base: str) -> str | None:
    """Resolve the API key, allowing local OpenAI-compatible endpoints to omit one."""
    if explicit_api_key:
        return explicit_api_key

    env_api_key = os.getenv("OPENAI_API_KEY")
    if env_api_key:
        return env_api_key

    if _is_local_endpoint(api_base):
        return "EMPTY"

    return None


def _is_local_endpoint(api_base: str) -> bool:
    """Return whether the configured base URL targets a localhost-style endpoint."""
    hostname = urlparse(api_base).hostname
    return hostname in {"127.0.0.1", "0.0.0.0", "::1", "localhost"}


def _extract_json_payload(text: str) -> str:
    """Remove simple Markdown code fences around a JSON payload."""
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    return stripped
