import os
import unittest
from unittest.mock import patch

from hypoevolve.core.config import LLMConfig
from hypoevolve.runtime.llm_client import (
    LLMClient,
    LLMError,
    _extract_json_payload,
    _is_local_endpoint,
    _resolve_api_key,
)


class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]


class _FakeCompletions:
    def __init__(self, content):
        self._content = content

    def create(self, **kwargs):
        return _FakeResponse(self._content)


class _FakeChat:
    def __init__(self, content):
        self.completions = _FakeCompletions(content)


class _FakeClient:
    def __init__(self, content):
        self.chat = _FakeChat(content)


class TestHypoEvolveLLM(unittest.TestCase):
    def test_extract_json_payload_strips_code_fence(self):
        text = '```json\n{"a": 1}\n```'
        self.assertEqual(_extract_json_payload(text), '{"a": 1}')

    def test_api_key_falls_back_to_environment(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "env-key"}, clear=True):
            client = LLMClient(LLMConfig())
            self.assertEqual(client.api_key, "env-key")

    def test_explicit_api_key_beats_environment(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "env-key"}, clear=True):
            client = LLMClient(
                LLMConfig(
                    api_key="yaml-key",
                    api_base="https://openrouter.ai/api/v1",
                )
            )
            self.assertEqual(client.api_key, "yaml-key")

    def test_local_endpoint_uses_placeholder_api_key(self):
        with patch.dict(os.environ, {}, clear=True):
            client = LLMClient(
                LLMConfig(api_key=None, api_base="http://127.0.0.1:8000/v1")
            )
            self.assertEqual(client.api_key, "EMPTY")

    def test_generate_text_uses_stubbed_client(self):
        client = LLMClient(LLMConfig(api_key="test-key"))
        client._client = _FakeClient("hello")
        self.assertEqual(client.generate_text("sys", "user"), "hello")

    def test_generate_json_parses_payload(self):
        client = LLMClient(LLMConfig(api_key="test-key"))
        client._client = _FakeClient('{"score": 0.9}')
        payload = client.generate_json("sys", "user")
        self.assertEqual(payload["score"], 0.9)

    def test_generate_json_raises_on_invalid_json(self):
        client = LLMClient(LLMConfig(api_key="test-key"))
        client._client = _FakeClient("not-json")
        with self.assertRaises(LLMError):
            client.generate_json("sys", "user")

    def test_generate_text_raises_without_api_key(self):
        with patch.dict(os.environ, {}, clear=True):
            client = LLMClient(
                LLMConfig(api_key=None, api_base="https://example.com/v1")
            )
            with self.assertRaises(LLMError):
                client.generate_text("sys", "user")

    def test_is_local_endpoint_detects_loopback_hosts(self):
        self.assertTrue(_is_local_endpoint("http://localhost:8000/v1"))
        self.assertTrue(_is_local_endpoint("http://127.0.0.1:8000/v1"))
        self.assertFalse(_is_local_endpoint("https://openrouter.ai/api/v1"))


    def test_resolve_api_key_returns_none_for_remote_without_env(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(_resolve_api_key(None, "https://example.com/v1"))

    def test_generate_text_retries_then_raises_llm_error(self):
        client = LLMClient(LLMConfig(api_key="test-key", retries=1, retry_delay=0))
        attempts = []

        def fail_call(messages, **kwargs):
            attempts.append(messages)
            raise RuntimeError("boom")

        client._call_openai = fail_call
        with self.assertRaises(LLMError):
            client.generate_text("sys", "user")
        self.assertEqual(len(attempts), 2)

    def test_call_openai_raises_when_message_content_is_none(self):
        client = LLMClient(LLMConfig(api_key="test-key"))
        client._client = _FakeClient(None)
        with self.assertRaises(LLMError):
            client._call_openai([{"role": "system", "content": "sys"}])

    def test_get_client_raises_clean_error_when_openai_import_fails(self):
        client = LLMClient(LLMConfig(api_key="test-key"))
        with patch.dict('sys.modules', {'openai': None}):
            with patch('builtins.__import__', side_effect=ImportError('no openai')):
                with self.assertRaises(LLMError):
                    client._get_client()


    def test_extract_json_payload_returns_plain_text_when_not_fenced(self):
        self.assertEqual(_extract_json_payload('{"a": 1}'), '{"a": 1}')

    def test_generate_json_retries_after_invalid_json_text(self):
        client = LLMClient(LLMConfig(api_key='test-key', retry_delay=0))
        responses = iter(['not-json', '{"score": 1.0}'])
        client.generate_text = lambda system, user, **kwargs: next(responses)
        payload = client.generate_json('sys', 'user', json_retries=1)
        self.assertEqual(payload, {'score': 1.0})


if __name__ == "__main__":
    unittest.main()
