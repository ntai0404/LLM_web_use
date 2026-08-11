import asyncio
import importlib
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

import app as app_module
from providers import (
    GeminiWebProvider,
    ProviderAuthRequired,
    ProviderTimeout,
)


class FakeManager:
    def __init__(self, error=None):
        self.error = error

    async def generate(self, prompt, model, files=None, **options):
        if self.error:
            raise self.error
        return SimpleNamespace(
            text="CONTRACT_OK",
            model=model,
            provider="gemini-web",
            metadata={"browser_request": True},
        )


class SlowGeminiClient:
    def __init__(self):
        self.headless = True
        self.stop_calls = 0
        self.last_call_metrics = {}

    async def ask(self, *args, **kwargs):
        await asyncio.sleep(1)
        return "late"

    async def stop(self):
        self.stop_calls += 1


class ApiContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app_module.app)

    def test_openapi_hides_local_ops_and_legacy_alias(self):
        paths = app_module.app.openapi()["paths"]

        self.assertIn("/api/generate", paths)
        self.assertIn("/v1/chat/completions", paths)
        self.assertNotIn("/admin/shutdown", paths)
        self.assertNotIn("/providers/{provider_name}/keepalive", paths)

    def test_invalid_json_has_stable_top_level_error(self):
        response = self.client.post(
            "/api/generate",
            content="{not-json",
            headers={"content-type": "application/json", "x-request-id": "contract-json"},
        )

        self.assertEqual(422, response.status_code)
        self.assertEqual("VALIDATION_ERROR", response.json()["error"])
        self.assertEqual("contract-json", response.json()["request_id"])
        self.assertNotIn("detail", response.json())
        self.assertEqual("contract-json", response.headers["x-request-id"])

    def test_empty_messages_and_invalid_tool_message_are_rejected(self):
        empty = self.client.post(
            "/v1/chat/completions",
            json={"model": "gemini-web", "messages": []},
        )
        invalid_tool = self.client.post(
            "/v1/chat/completions",
            json={
                "model": "gemini-web",
                "messages": [{"role": "tool", "content": "hello"}],
            },
        )

        self.assertEqual(422, empty.status_code)
        self.assertEqual("VALIDATION_ERROR", empty.json()["error"])
        self.assertEqual(422, invalid_tool.status_code)
        self.assertEqual("VALIDATION_ERROR", invalid_tool.json()["error"])

    def test_blank_prompt_is_rejected_without_provider_call(self):
        response = self.client.post(
            "/api/generate", json={"model": "gemini-web", "prompt": "   "}
        )

        self.assertEqual(422, response.status_code)
        self.assertEqual("VALIDATION_ERROR", response.json()["error"])

    def test_openai_optional_parameters_are_accepted(self):
        with patch.object(app_module, "manager", FakeManager()):
            response = self.client.post(
                "/v1/chat/completions",
                json={
                    "model": "gemini-web",
                    "messages": [{"role": "user", "content": "hello"}],
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "frequency_penalty": 0.1,
                    "presence_penalty": 0.2,
                    "seed": 42,
                    "client_optional_extension": "controlled-no-op",
                },
            )

        self.assertEqual(200, response.status_code)

    def test_router_404_uses_stable_error_contract(self):
        response = self.client.get("/does-not-exist")

        self.assertEqual(404, response.status_code)
        self.assertEqual("NOT_FOUND", response.json()["error"])
        self.assertNotIn("detail", response.json())

    def test_unknown_model_returns_404_error_contract(self):
        response = self.client.post(
            "/api/generate", json={"model": "does-not-exist", "prompt": "hello"}
        )

        self.assertEqual(404, response.status_code)
        self.assertEqual("PROVIDER_NOT_FOUND", response.json()["error"])
        self.assertNotIn("detail", response.json())

    def test_openai_response_shape_and_usage_null(self):
        with patch.object(app_module, "manager", FakeManager()):
            response = self.client.post(
                "/v1/chat/completions",
                json={
                    "model": "gemini-web",
                    "messages": [{"role": "user", "content": "hello"}],
                },
            )

        payload = response.json()
        self.assertEqual(200, response.status_code)
        self.assertEqual("chat.completion", payload["object"])
        self.assertTrue(payload["id"].startswith("chatcmpl-"))
        self.assertEqual("gemini-web", payload["model"])
        self.assertEqual("CONTRACT_OK", payload["choices"][0]["message"]["content"])
        self.assertEqual("stop", payload["choices"][0]["finish_reason"])
        self.assertIsNone(payload["usage"])

    def test_auth_required_is_401_without_detail_wrapper(self):
        with patch.object(
            app_module,
            "manager",
            FakeManager(ProviderAuthRequired("Login is required")),
        ):
            response = self.client.post(
                "/api/generate", json={"model": "gemini-web", "prompt": "hello"}
            )

        self.assertEqual(401, response.status_code)
        self.assertEqual("AUTH_REQUIRED", response.json()["error"])
        self.assertNotIn("detail", response.json())


class ProviderTimeoutTests(unittest.IsolatedAsyncioTestCase):
    async def test_timeout_closes_browser_before_lock_is_released(self):
        client = SlowGeminiClient()
        provider = GeminiWebProvider(client, operation_timeout_seconds=0.01)

        with self.assertRaises(ProviderTimeout):
            await provider.generate("hello", "gemini-web")

        self.assertEqual(0, client.stop_calls)
        self.assertFalse(provider._operation_lock.locked())
        self.assertEqual("GENERATION_TIMEOUT", provider.last_error)


if __name__ == "__main__":
    unittest.main()
