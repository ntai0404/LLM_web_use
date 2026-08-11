import asyncio
import unittest
from datetime import datetime, timedelta, timezone

from provider_manager import ProviderManager, ProviderNotFound, ProviderRegistry
from providers.base import (
    GenerationResult,
    KeepalivePolicy,
    KeepaliveResult,
    LLMProvider,
    ProviderStatus,
)
from providers.gemini import GeminiWebProvider
from scheduler import ProviderScheduler


class FakeProvider(LLMProvider):
    name = "fake-web"
    model_aliases = frozenset({"fake-web", "fake-model"})

    def __init__(self, policy=None):
        self.keepalive_policy = policy or KeepalivePolicy(enabled=False)
        self.keepalive_calls = 0
        self.shutdown_calls = 0

    async def generate(self, prompt, model=None, files=None, **options):
        marker = "Reply exactly with this token and nothing else: "
        if marker in prompt:
            return GenerationResult(prompt.split(marker, 1)[1], model or self.name, self.name)
        return GenerationResult(prompt.upper(), model or self.name, self.name)

    async def health_check(self):
        return {"status": ProviderStatus.READY.value, "auth": "ok"}

    async def auth_status(self):
        return "ok"

    async def keepalive(self):
        self.keepalive_calls += 1
        now = datetime.now(timezone.utc).isoformat()
        return KeepaliveResult(True, ProviderStatus.READY, now, now, True)

    async def shutdown(self):
        self.shutdown_calls += 1


class FastPolicy:
    enabled = True

    def next_run(self):
        return datetime.now(timezone.utc) + timedelta(milliseconds=10)


class FakeGeminiClient:
    def __init__(self, auth_required=False):
        self.headless = True
        self.auth_is_required = auth_required
        self.prompts = []
        self.active = 0
        self.max_active = 0
        self.last_request_host = "gemini.google.com"
        self.last_request_endpoint = "https://gemini.google.com/_/BardChatUi/data/assistant.lamda.BardFrontendService/StreamGenerate"
        self.last_http_status = 200
        self.last_call_metrics = {"duration_seconds": 0.01}
        self.stopped = False

    async def ask(self, prompt, model=None, files=None, **options):
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        self.prompts.append(prompt)
        await asyncio.sleep(0.02)
        self.active -= 1
        marker = "Reply exactly with this token and nothing else: "
        return prompt.split(marker, 1)[1] if marker in prompt else "USER_OK"

    async def auth_required(self):
        return self.auth_is_required

    async def health(self):
        return {
            "ok": not self.auth_is_required,
            "auth": "required" if self.auth_is_required else "ok",
            "transport": "internal-stream-generate",
        }

    async def list_models(self, refresh=False):
        return [{"id": "fake", "display_name": "Fake"}]

    async def estimate_request(self, prompt, model=None, files=None, **options):
        return {"input_tokens_estimated": len(prompt)}

    async def stop(self):
        self.stopped = True


class ProviderSystemTests(unittest.IsolatedAsyncioTestCase):
    async def test_registry_routes_model_without_core_provider_logic(self):
        registry = ProviderRegistry()
        provider = FakeProvider()
        registry.register(provider)
        manager = ProviderManager(registry)

        result = await manager.generate("hello", "fake-model")

        self.assertEqual("HELLO", result.text)
        self.assertEqual("fake-web", result.provider)
        with self.assertRaises(ProviderNotFound):
            registry.resolve("unknown-model")

    async def test_generic_scheduler_only_calls_provider_keepalive(self):
        provider = FakeProvider(policy=FastPolicy())
        scheduler = ProviderScheduler([provider])

        await scheduler.start()
        await asyncio.sleep(0.04)
        await scheduler.stop()

        self.assertGreaterEqual(provider.keepalive_calls, 1)

    async def test_gemini_keepalive_is_serialized_and_context_free(self):
        client = FakeGeminiClient()
        provider = GeminiWebProvider(
            client,
            nonce_factory=lambda: "KEEPALIVE_TEST_NONCE",
        )

        keepalive_task = asyncio.create_task(provider.keepalive())
        await asyncio.sleep(0)
        user_task = asyncio.create_task(provider.generate("USER_PROMPT", "gemini-web"))
        keepalive, user_result = await asyncio.gather(keepalive_task, user_task)

        self.assertTrue(keepalive.success)
        self.assertTrue(keepalive.verified)
        self.assertEqual("USER_OK", user_result.text)
        self.assertEqual(1, client.max_active)
        self.assertEqual("USER_PROMPT", client.prompts[-1])
        self.assertNotIn("KEEPALIVE_TEST_NONCE", client.prompts[-1])

    async def test_auth_required_keepalive_does_not_generate(self):
        client = FakeGeminiClient(auth_required=True)
        provider = GeminiWebProvider(client)

        result = await provider.keepalive()

        self.assertFalse(result.success)
        self.assertEqual(ProviderStatus.AUTH_REQUIRED, result.status)
        self.assertEqual([], client.prompts)

    async def test_manager_shutdown_closes_provider_once(self):
        registry = ProviderRegistry()
        provider = FakeProvider()
        registry.register(provider)
        manager = ProviderManager(registry)

        await manager.shutdown()

        self.assertEqual(1, provider.shutdown_calls)

    async def test_provider_admin_test_uses_registered_provider(self):
        registry = ProviderRegistry()
        registry.register(FakeProvider())
        manager = ProviderManager(registry)

        result = await manager.test_provider("fake-web")

        self.assertTrue(result["success"])
        self.assertTrue(result["exact_match"])
        self.assertEqual("fake-web", result["provider"])


if __name__ == "__main__":
    unittest.main()
