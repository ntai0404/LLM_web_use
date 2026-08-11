from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any
from zoneinfo import ZoneInfo


class ProviderStatus(str, Enum):
    READY = "READY"
    DEGRADED = "DEGRADED"
    AUTH_REQUIRED = "AUTH_REQUIRED"
    OFFLINE = "OFFLINE"


class ProviderError(RuntimeError):
    pass


class ProviderAuthRequired(ProviderError):
    pass


class ProviderTimeout(ProviderError):
    pass


class ProviderUnavailable(ProviderError):
    pass


@dataclass(frozen=True)
class KeepalivePolicy:
    enabled: bool = False
    timezone: str = "UTC"
    hour: int = 0
    minute: int = 0

    def __post_init__(self) -> None:
        if not 0 <= self.hour <= 23:
            raise ValueError("Keepalive hour must be between 0 and 23")
        if not 0 <= self.minute <= 59:
            raise ValueError("Keepalive minute must be between 0 and 59")
        ZoneInfo(self.timezone)

    def next_run(self, now: datetime | None = None) -> datetime:
        zone = ZoneInfo(self.timezone)
        local_now = (now or datetime.now(zone)).astimezone(zone)
        target = local_now.replace(
            hour=self.hour,
            minute=self.minute,
            second=0,
            microsecond=0,
        )
        if target <= local_now:
            target += timedelta(days=1)
        return target


@dataclass
class GenerationResult:
    text: str
    model: str
    provider: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class KeepaliveResult:
    success: bool
    status: ProviderStatus
    attempted_at: str
    completed_at: str | None = None
    verified: bool = False
    detail: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "status": self.status.value,
            "attempted_at": self.attempted_at,
            "completed_at": self.completed_at,
            "verified": self.verified,
            "detail": self.detail,
            "metadata": self.metadata,
        }


class LLMProvider(ABC):
    name: str
    model_aliases: frozenset[str]
    keepalive_policy: KeepalivePolicy

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        model: str | None = None,
        files: list[str] | None = None,
        **options: Any,
    ) -> GenerationResult:
        raise NotImplementedError

    @abstractmethod
    async def health_check(self) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    async def auth_status(self) -> str:
        raise NotImplementedError

    @abstractmethod
    async def keepalive(self) -> KeepaliveResult:
        raise NotImplementedError

    async def list_models(self, refresh: bool = False) -> list[dict[str, Any]]:
        return [{"id": alias, "provider": self.name} for alias in sorted(self.model_aliases)]

    def model_profiles(self) -> list[dict[str, Any]]:
        return []

    async def estimate(self, prompt: str, model: str | None = None, files: list[str] | None = None, **options: Any) -> dict[str, Any]:
        return {"provider": self.name, "available": False}

    async def management_info(self, refresh: bool = False) -> dict[str, Any]:
        health = await self.health_check() if refresh else {}
        return {
            "id": self.name,
            "display_name": self.name,
            "provider_type": "provider",
            "model_aliases": sorted(self.model_aliases),
            "status": health.get("status", ProviderStatus.OFFLINE.value),
            "auth": health.get("auth", "unknown"),
            "keepalive": {
                "enabled": self.keepalive_policy.enabled,
                "timezone": self.keepalive_policy.timezone,
                "hour": self.keepalive_policy.hour,
                "minute": self.keepalive_policy.minute,
            },
        }

    async def shutdown(self) -> None:
        return None
