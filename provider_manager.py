from __future__ import annotations

import secrets
import time
from typing import Any

from providers.base import GenerationResult, LLMProvider


class ProviderNotFound(ValueError):
    pass


class ProviderRegistry:
    def __init__(self) -> None:
        self._providers: dict[str, LLMProvider] = {}
        self._model_routes: dict[str, str] = {}

    def register(self, provider: LLMProvider) -> None:
        key = provider.name.lower()
        if key in self._providers:
            raise ValueError(f"Provider already registered: {provider.name}")
        self._providers[key] = provider
        for alias in provider.model_aliases | {provider.name}:
            normalized = alias.strip().lower()
            owner = self._model_routes.get(normalized)
            if owner and owner != key:
                raise ValueError(f"Model alias already registered: {alias}")
            self._model_routes[normalized] = key

    def provider(self, name: str) -> LLMProvider:
        provider = self._providers.get(name.strip().lower())
        if provider is None:
            raise ProviderNotFound(f"Unknown provider: {name}")
        return provider

    def resolve(self, model: str | None) -> LLMProvider:
        normalized = (model or "").strip().lower()
        if not normalized and len(self._providers) == 1:
            return next(iter(self._providers.values()))
        provider_name = self._model_routes.get(normalized)
        if provider_name is None:
            raise ProviderNotFound(f"No provider registered for model: {model}")
        return self._providers[provider_name]

    def all(self) -> list[LLMProvider]:
        return list(self._providers.values())


class ProviderManager:
    def __init__(self, registry: ProviderRegistry) -> None:
        self.registry = registry

    async def generate(
        self,
        prompt: str,
        model: str | None,
        files: list[str] | None = None,
        **options: Any,
    ) -> GenerationResult:
        provider = self.registry.resolve(model)
        return await provider.generate(prompt, model, files=files, **options)

    async def estimate(
        self,
        prompt: str,
        model: str | None,
        files: list[str] | None = None,
        **options: Any,
    ) -> dict[str, Any]:
        provider = self.registry.resolve(model)
        return await provider.estimate(prompt, model, files=files, **options)

    async def health_check(self) -> dict[str, Any]:
        providers: dict[str, Any] = {}
        for provider in self.registry.all():
            try:
                providers[provider.name] = await provider.health_check()
            except Exception as exc:
                providers[provider.name] = {
                    "status": "OFFLINE",
                    "auth": "unknown",
                    "last_error": type(exc).__name__,
                }
        statuses = {item.get("status") for item in providers.values()}
        service = "ok" if statuses and statuses <= {"READY"} else "degraded"
        return {"service": service, "providers": providers}

    async def list_models(self, refresh: bool = False) -> list[dict[str, Any]]:
        models: list[dict[str, Any]] = []
        for provider in self.registry.all():
            for model in await provider.list_models(refresh=refresh):
                item = dict(model)
                item.setdefault("provider", provider.name)
                models.append(item)
        return models

    def model_profiles(self) -> list[dict[str, Any]]:
        profiles: list[dict[str, Any]] = []
        for provider in self.registry.all():
            for profile in provider.model_profiles():
                item = dict(profile)
                item.setdefault("provider", provider.name)
                profiles.append(item)
        return profiles

    async def trigger_keepalive(self, provider_name: str) -> dict[str, Any]:
        result = await self.registry.provider(provider_name).keepalive()
        return result.as_dict()

    async def provider_list(self, refresh: bool = False) -> list[dict[str, Any]]:
        providers: list[dict[str, Any]] = []
        for provider in self.registry.all():
            try:
                providers.append(await provider.management_info(refresh=refresh))
            except Exception as exc:
                providers.append(
                    {
                        "id": provider.name,
                        "display_name": provider.name,
                        "provider_type": "provider",
                        "status": "OFFLINE",
                        "auth": "unknown",
                        "model_aliases": sorted(provider.model_aliases),
                        "last_error": type(exc).__name__,
                    }
                )
        return providers

    async def provider_detail(self, provider_name: str, refresh: bool = False) -> dict[str, Any]:
        return await self.registry.provider(provider_name).management_info(refresh=refresh)

    async def test_provider(self, provider_name: str) -> dict[str, Any]:
        provider = self.registry.provider(provider_name)
        nonce = f"PROVIDER_TEST_{secrets.token_hex(8).upper()}"
        prompt = f"Reply exactly with this token and nothing else: {nonce}"
        started = time.perf_counter()
        result = await provider.generate(prompt, provider.name, max_output_tokens=64)
        latency_seconds = round(time.perf_counter() - started, 3)
        exact_match = result.text.strip() == nonce
        return {
            "success": exact_match,
            "provider": provider.name,
            "model": result.model,
            "response": result.text,
            "exact_match": exact_match,
            "latency_seconds": latency_seconds,
            "metadata": result.metadata,
        }

    async def shutdown(self) -> None:
        for provider in reversed(self.registry.all()):
            try:
                await provider.shutdown()
            except Exception:
                continue
