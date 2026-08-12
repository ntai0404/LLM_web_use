from __future__ import annotations

import asyncio
import re
import secrets
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Awaitable, Callable
from zoneinfo import ZoneInfo

from gemini_web import AuthRequired, BrowserUnavailable, GeminiWebClient, GeminiWebError
from providers.base import (
    GenerationResult,
    KeepalivePolicy,
    KeepaliveResult,
    LLMProvider,
    ProviderAuthRequired,
    ProviderBusy,
    ProviderError,
    ProviderRateLimited,
    ProviderStatus,
    ProviderTimeout,
    ProviderUnavailable,
)


ZONE_BANGKOK = ZoneInfo("Asia/Bangkok")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class GeminiWebProvider(LLMProvider):
    name = "gemini-web"
    model_aliases = frozenset(
        {
            "gemini-web",
            "3.5 flash-lite",
            "3.6 flash",
            "3.1 pro",
            "gemini-3.5-flash-lite",
            "gemini-3.6-flash",
            "gemini-3.1-pro-preview",
            "flash-lite",
            "flash",
            "pro",
            "default",
            "auto",
            "8c46e95b1a07cecc",
            "56fdd199312815e2",
            "e6fa609c3fa255c0",
        }
    )

    def __init__(
        self,
        client: GeminiWebClient,
        keepalive_policy: KeepalivePolicy | None = None,
        nonce_factory: Callable[[], str] | None = None,
        operation_timeout_seconds: float = 120.0,
        queue_timeout_seconds: float = 30.0,
    ) -> None:
        self.client = client
        self.keepalive_policy = keepalive_policy or KeepalivePolicy(
            enabled=True,
            timezone="Asia/Bangkok",
            hour=0,
            minute=0,
        )
        self._nonce_factory = nonce_factory or (
            lambda: f"KEEPALIVE_{datetime.now(ZONE_BANGKOK).date().isoformat()}_{secrets.token_hex(6).upper()}"
        )
        self.operation_timeout_seconds = max(0.001, operation_timeout_seconds)
        self.queue_timeout_seconds = max(0.001, queue_timeout_seconds)
        self._operation_lock = asyncio.Lock()
        self.status = ProviderStatus.OFFLINE
        self.auth_state = "unknown"
        self.last_success: str | None = None
        self.last_keepalive: str | None = None
        self.last_keepalive_result: dict[str, Any] | None = None
        self.last_error: str | None = None

    async def _generate_locked(
        self,
        prompt: str,
        model: str | None,
        files: list[str] | None,
        options: dict[str, Any],
    ) -> GenerationResult:
        try:
            text = await asyncio.wait_for(
                self.client.ask(
                    prompt,
                    model,
                    files=files,
                    max_input_tokens=options.get("max_input_tokens"),
                    max_output_tokens=options.get("max_output_tokens"),
                ),
                timeout=self.operation_timeout_seconds,
            )
        except asyncio.TimeoutError as exc:
            self.status = ProviderStatus.DEGRADED
            self.last_error = "GENERATION_TIMEOUT"
            raise ProviderTimeout(
                f"Gemini Web generation exceeded {self.operation_timeout_seconds:g} seconds"
            ) from exc
        except AuthRequired as exc:
            self.status = ProviderStatus.AUTH_REQUIRED
            self.auth_state = "required"
            self.last_error = "AUTH_REQUIRED"
            raise ProviderAuthRequired("Gemini Web login is required") from exc
        except BrowserUnavailable as exc:
            self.status = ProviderStatus.OFFLINE
            self.last_error = "BROWSER_UNAVAILABLE"
            raise ProviderUnavailable("Google Chrome CDP runtime is unavailable") from exc
        except GeminiWebError as exc:
            message = str(exc)
            if re.search(r"(?:HTTP|status)\s*429\b", message, re.IGNORECASE) or re.search(
                r"\b(rate.?limit|quota)\b", message, re.IGNORECASE
            ):
                self.status = ProviderStatus.DEGRADED
                self.last_error = "RATE_LIMITED"
                raise ProviderRateLimited(
                    "Gemini Web rate limit reached",
                    retry_after=self.client.last_retry_after,
                ) from exc
            self.status = ProviderStatus.DEGRADED
            self.last_error = "GEMINI_WEB_ERROR"
            raise ProviderError(message) from exc
        except ValueError:
            raise
        except Exception as exc:
            self.status = ProviderStatus.OFFLINE
            self.last_error = "BROWSER_UNAVAILABLE"
            raise ProviderUnavailable("Gemini Web browser runtime is unavailable") from exc

        self.status = ProviderStatus.READY
        self.auth_state = "ok"
        self.last_success = utc_now()
        self.last_error = None
        return GenerationResult(
            text=text,
            model=model or self.name,
            provider=self.name,
            metadata=dict(self.client.last_call_metrics),
        )

    async def generate(
        self,
        prompt: str,
        model: str | None = None,
        files: list[str] | None = None,
        **options: Any,
    ) -> GenerationResult:
        async with self.generation_session() as generate:
            return await generate(prompt, model, files=files, **options)

    @asynccontextmanager
    async def generation_session(
        self,
    ) -> AsyncIterator[Callable[..., Awaitable[GenerationResult]]]:
        """Hold the browser lock across first-pass and bounded repair calls."""
        try:
            await asyncio.wait_for(
                self._operation_lock.acquire(),
                timeout=self.queue_timeout_seconds,
            )
        except asyncio.TimeoutError as exc:
            raise ProviderBusy(
                f"Gemini browser queue remained busy for {self.queue_timeout_seconds:g} seconds"
            ) from exc

        async def generate_in_session(
            prompt: str,
            model: str | None = None,
            files: list[str] | None = None,
            **options: Any,
        ) -> GenerationResult:
            return await self._generate_locked(prompt, model, files, options)

        try:
            yield generate_in_session
        finally:
            self._operation_lock.release()

    async def auth_status(self) -> str:
        async with self._operation_lock:
            try:
                required = await self.client.auth_required()
            except Exception:
                self.status = ProviderStatus.OFFLINE
                self.auth_state = "unknown"
                self.last_error = "BROWSER_UNAVAILABLE"
                return "unknown"
            if required:
                self.status = ProviderStatus.AUTH_REQUIRED
                self.auth_state = "required"
                self.last_error = "AUTH_REQUIRED"
                return "required"
            if self.status in {ProviderStatus.OFFLINE, ProviderStatus.AUTH_REQUIRED}:
                self.status = ProviderStatus.READY
            self.last_error = None
            self.auth_state = "ok"
            return "ok"

    async def health_check(self) -> dict[str, Any]:
        if self._operation_lock.locked():
            return self._health_payload(busy=True)
        try:
            await asyncio.wait_for(self._operation_lock.acquire(), timeout=0.1)
        except asyncio.TimeoutError:
            return self._health_payload(busy=True)
        try:
            raw = await self.client.health()
            auth = str(raw.get("auth", "unknown"))
            if raw.get("ok") and auth == "ok":
                self.status = ProviderStatus.READY
                self.auth_state = "ok"
                self.last_error = None
            elif auth == "required":
                self.status = ProviderStatus.AUTH_REQUIRED
                self.auth_state = "required"
                self.last_error = "AUTH_REQUIRED"
            elif raw.get("url"):
                self.status = ProviderStatus.DEGRADED
                self.last_error = str(raw.get("error") or "HEALTH_CHECK_FAILED")
            else:
                self.status = ProviderStatus.OFFLINE
                self.auth_state = "unknown"
                self.last_error = str(raw.get("error") or "BROWSER_UNAVAILABLE")
            return self._health_payload(raw=raw, busy=False, auth=auth)
        finally:
            self._operation_lock.release()

    def _health_payload(
        self,
        raw: dict[str, Any] | None = None,
        busy: bool = False,
        auth: str | None = None,
    ) -> dict[str, Any]:
        raw = raw or {}
        return {
            "status": self.status.value,
            "auth": auth or self.auth_state,
            "busy": busy,
            "last_success": self.last_success,
            "last_keepalive": self.last_keepalive,
            "last_keepalive_result": self.last_keepalive_result,
            "last_error": self.last_error,
            "browser_headless": self.client.headless,
            "browser_visible": not self.client.headless,
            "transport": raw.get("transport") or "internal-stream-generate",
            "last_request_host": raw.get("last_request_host") or self.client.last_request_host,
            "last_request_endpoint": raw.get("last_request_endpoint") or self.client.last_request_endpoint,
            "last_http_status": raw.get("last_http_status") or self.client.last_http_status,
            "last_retry_after": raw.get("last_retry_after") or self.client.last_retry_after,
            "last_observed_model_id": raw.get("last_observed_model_id") or self.client.last_observed_model_id,
            "keepalive_policy": {
                "enabled": self.keepalive_policy.enabled,
                "timezone": self.keepalive_policy.timezone,
                "hour": self.keepalive_policy.hour,
                "minute": self.keepalive_policy.minute,
            },
        }

    async def keepalive(self) -> KeepaliveResult:
        attempted_at = utc_now()
        async with self._operation_lock:
            try:
                if await self.client.auth_required():
                    self.status = ProviderStatus.AUTH_REQUIRED
                    self.auth_state = "required"
                    self.last_error = "AUTH_REQUIRED"
                    result = KeepaliveResult(
                        success=False,
                        status=self.status,
                        attempted_at=attempted_at,
                        completed_at=utc_now(),
                        detail="AUTH_REQUIRED",
                    )
                else:
                    nonce = self._nonce_factory()
                    prompt = f"Reply exactly with this token and nothing else: {nonce}"
                    generated = await self._generate_locked(
                        prompt,
                        self.name,
                        None,
                        {"max_output_tokens": 64},
                    )
                    exact_match = generated.text.strip() == nonce
                    network_verified = (
                        self.client.last_request_host == "gemini.google.com"
                        and self.client.last_http_status == 200
                    )
                    success = exact_match and network_verified
                    self.status = ProviderStatus.READY if success else ProviderStatus.DEGRADED
                    self.auth_state = "ok"
                    if success:
                        self.last_success = utc_now()
                        self.last_error = None
                    else:
                        self.last_error = "KEEPALIVE_VERIFICATION_FAILED"
                    result = KeepaliveResult(
                        success=success,
                        status=self.status,
                        attempted_at=attempted_at,
                        completed_at=utc_now(),
                        verified=success,
                        detail="ok" if success else self.last_error,
                        metadata={
                            "exact_match": exact_match,
                            "request_host": self.client.last_request_host,
                            "request_endpoint": self.client.last_request_endpoint,
                            "http_status": self.client.last_http_status,
                        },
                    )
            except ProviderAuthRequired:
                self.status = ProviderStatus.AUTH_REQUIRED
                self.auth_state = "required"
                self.last_error = "AUTH_REQUIRED"
                result = KeepaliveResult(
                    success=False,
                    status=self.status,
                    attempted_at=attempted_at,
                    completed_at=utc_now(),
                    detail="AUTH_REQUIRED",
                )
            except Exception as exc:
                self.status = ProviderStatus.DEGRADED
                self.last_error = type(exc).__name__
                result = KeepaliveResult(
                    success=False,
                    status=self.status,
                    attempted_at=attempted_at,
                    completed_at=utc_now(),
                    detail=self.last_error,
                )
            self.last_keepalive = result.completed_at
            self.last_keepalive_result = result.as_dict()
            return result

    async def list_models(self, refresh: bool = False) -> list[dict[str, Any]]:
        async with self._operation_lock:
            return await self.client.list_models(refresh=refresh)

    async def estimate(
        self,
        prompt: str,
        model: str | None = None,
        files: list[str] | None = None,
        **options: Any,
    ) -> dict[str, Any]:
        return await self.client.estimate_request(
            prompt,
            model,
            files=files,
            max_input_tokens=options.get("max_input_tokens"),
            max_output_tokens=options.get("max_output_tokens"),
        )

    def model_profiles(self) -> list[dict[str, Any]]:
        return self.client.reference_profiles()

    async def management_info(self, refresh: bool = False) -> dict[str, Any]:
        health = await self.health_check() if refresh else None
        if health is not None:
            status = health["status"]
            auth = health["auth"]
        else:
            status = self.status.value
            auth = self.auth_state
        schedule = self.keepalive_policy
        return {
            "id": self.name,
            "display_name": "Gemini Web",
            "provider_type": "browser-backed",
            "status": status,
            "auth": auth,
            "browser_runtime": "Google Chrome",
            "headless": self.client.headless,
            "profile": Path(self.client.profile_dir).name,
            "model_aliases": sorted(self.model_aliases),
            "keepalive": {
                "enabled": schedule.enabled,
                "strategy": "authenticated_generation",
                "timezone": schedule.timezone,
                "hour": schedule.hour,
                "minute": schedule.minute,
            },
            "last_keepalive": self.last_keepalive,
            "last_success": self.last_success,
            "last_error": self.last_error,
            "transport": "internal-stream-generate",
        }

    async def shutdown(self) -> None:
        async with self._operation_lock:
            await self.client.stop()
            self.status = ProviderStatus.OFFLINE
