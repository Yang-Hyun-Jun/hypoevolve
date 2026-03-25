import os
import unittest
from unittest.mock import patch

from hypoevolve.config import LLMConfig
from hypoevolve.llm import LLMClient, LLMError, _extract_json_payload


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
        client = LLMClient(LLMConfig(api_key=None))
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(LLMError):
                client.generate_text("sys", "user")
