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
    ProviderBusy,
    ProviderAuthRequired,
    ProviderError,
    ProviderRateLimited,
)
from providers.gemini import GeminiWebProvider
from gemini_web import AuthRequired, GeminiWebError
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
        self.last_retry_after = None
        self.last_observed_model_id = None
        self.last_call_metrics = {"duration_seconds": 0.01}
        self.stopped = False
        self.scripted_outcomes = []

    async def ask(self, prompt, model=None, files=None, **options):
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        self.prompts.append(prompt)
        await asyncio.sleep(0.02)
        self.active -= 1
        if self.scripted_outcomes:
            outcome = self.scripted_outcomes.pop(0)
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome
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
    async def test_unclassified_stream_error_is_retried_then_succeeds(self):
        client = FakeGeminiClient()
        client.scripted_outcomes = [GeminiWebError("Gemini stream error 1095"), "RECOVERED"]
        provider = GeminiWebProvider(client, upstream_retry_base_seconds=0)

        result = await provider.generate("PROMPT", "gemini-web")

        self.assertEqual("RECOVERED", result.text)
        self.assertEqual(["PROMPT", "PROMPT"], client.prompts)
        self.assertEqual(ProviderStatus.READY, provider.status)

    async def test_stream_1095_uses_recovery_budget_beyond_generic_attempt_limit(self):
        client = FakeGeminiClient()
        client.scripted_outcomes = [
            GeminiWebError("Gemini stream error 1095"),
            GeminiWebError("Gemini stream error 1095"),
            GeminiWebError("Gemini stream error 1095"),
            "RECOVERED_AFTER_THREE_FAILURES",
        ]
        provider = GeminiWebProvider(
            client,
            upstream_max_attempts=3,
            upstream_retry_base_seconds=0,
            stream_1095_recovery_budget_seconds=1,
        )

        result = await provider.generate("PROMPT", "gemini-web")

        self.assertEqual("RECOVERED_AFTER_THREE_FAILURES", result.text)
        self.assertEqual(4, len(client.prompts))

    async def test_other_stream_code_keeps_generic_attempt_limit(self):
        client = FakeGeminiClient()
        client.scripted_outcomes = [
            GeminiWebError("Gemini stream error 1096"),
            GeminiWebError("Gemini stream error 1096"),
            GeminiWebError("Gemini stream error 1096"),
        ]
        provider = GeminiWebProvider(client, upstream_retry_base_seconds=0)

        with self.assertRaisesRegex(ProviderError, "Gemini stream error 1096"):
            await provider.generate("PROMPT", "gemini-web")

        self.assertEqual(3, len(client.prompts))

    async def test_stream_1095_budget_exhaustion_keeps_provider_error_mapping(self):
        client = FakeGeminiClient()
        client.scripted_outcomes = [GeminiWebError("Gemini stream error 1095")]
        provider = GeminiWebProvider(
            client,
            upstream_retry_base_seconds=0.02,
            stream_1095_recovery_budget_seconds=0.01,
        )

        with self.assertRaisesRegex(ProviderError, "Gemini stream error 1095"):
            await provider.generate("PROMPT", "gemini-web")

        self.assertEqual(1, len(client.prompts))
        self.assertEqual("GEMINI_WEB_ERROR", provider.last_error)

    async def test_other_unclassified_upstream_errors_are_bounded(self):
        client = FakeGeminiClient()
        client.scripted_outcomes = [
            GeminiWebError("unknown upstream A"),
            GeminiWebError("unknown upstream B"),
            GeminiWebError("unknown upstream C"),
        ]
        provider = GeminiWebProvider(client, upstream_retry_base_seconds=0)

        with self.assertRaisesRegex(ProviderError, "unknown upstream C"):
            await provider.generate("PROMPT", "gemini-web")

        self.assertEqual(3, len(client.prompts))
        self.assertEqual("GEMINI_WEB_ERROR", provider.last_error)

    async def test_rate_limit_keeps_existing_mapping_without_retry(self):
        client = FakeGeminiClient()
        client.scripted_outcomes = [GeminiWebError("StreamGenerate returned HTTP 429")]
        provider = GeminiWebProvider(client, upstream_retry_base_seconds=0)

        with self.assertRaises(ProviderRateLimited):
            await provider.generate("PROMPT", "gemini-web")

        self.assertEqual(1, len(client.prompts))
        self.assertEqual("RATE_LIMITED", provider.last_error)

    async def test_auth_error_keeps_existing_mapping_without_retry(self):
        client = FakeGeminiClient()
        client.scripted_outcomes = [AuthRequired("login required")]
        provider = GeminiWebProvider(client, upstream_retry_base_seconds=0)

        with self.assertRaises(ProviderAuthRequired):
            await provider.generate("PROMPT", "gemini-web")

        self.assertEqual(1, len(client.prompts))
        self.assertEqual("AUTH_REQUIRED", provider.last_error)

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

    async def test_health_is_non_blocking_while_generation_lock_is_busy(self):
        client = FakeGeminiClient()
        provider = GeminiWebProvider(client)
        provider.status = ProviderStatus.READY
        provider.auth_state = "ok"
        await provider._operation_lock.acquire()
        try:
            health = await asyncio.wait_for(provider.health_check(), timeout=0.1)
        finally:
            provider._operation_lock.release()

        self.assertTrue(health["busy"])
        self.assertEqual("READY", health["status"])
        self.assertEqual("ok", health["auth"])

    async def test_generation_queue_has_bounded_busy_timeout(self):
        client = FakeGeminiClient()
        provider = GeminiWebProvider(client, queue_timeout_seconds=0.01)
        await provider._operation_lock.acquire()
        try:
            with self.assertRaises(ProviderBusy):
                await provider.generate("queued", "gemini-web")
        finally:
            provider._operation_lock.release()

    async def test_generation_session_keeps_repair_before_next_request(self):
        client = FakeGeminiClient()
        provider = GeminiWebProvider(client)

        async with provider.generation_session() as generate:
            first = await generate("FIRST", "gemini-web")
            queued = asyncio.create_task(provider.generate("NEXT", "gemini-web"))
            await asyncio.sleep(0)
            repaired = await generate("REPAIR", "gemini-web")

        next_result = await queued

        self.assertEqual("USER_OK", first.text)
        self.assertEqual("USER_OK", repaired.text)
        self.assertEqual("USER_OK", next_result.text)
        self.assertEqual(["FIRST", "REPAIR", "NEXT"], client.prompts)
        self.assertEqual(1, client.max_active)

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
