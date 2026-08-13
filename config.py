from __future__ import annotations

import os
from dataclasses import dataclass


def env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    host: str
    port: int
    gemini_profile_dir: str
    gemini_timeout_ms: int
    gemini_queue_timeout_ms: int
    gemini_upstream_max_attempts: int
    gemini_upstream_retry_base_seconds: float
    headless: bool
    keepalive_enabled: bool
    keepalive_timezone: str
    keepalive_hour: int
    keepalive_minute: int

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            host=os.getenv("HOST", "0.0.0.0"),
            port=int(os.getenv("PORT", "4444")),
            gemini_profile_dir=os.getenv(
                "GEMINI_PROFILE_DIR",
                "var/profiles/gemini-main",
            ),
            gemini_timeout_ms=int(os.getenv("GEMINI_TIMEOUT_MS", "120000")),
            gemini_queue_timeout_ms=int(
                os.getenv("GEMINI_QUEUE_TIMEOUT_MS", "30000")
            ),
            gemini_upstream_max_attempts=max(
                1,
                int(os.getenv("GEMINI_UPSTREAM_MAX_ATTEMPTS", "3")),
            ),
            gemini_upstream_retry_base_seconds=max(
                0.0,
                float(os.getenv("GEMINI_UPSTREAM_RETRY_BASE_SECONDS", "1")),
            ),
            headless=env_bool("HEADLESS", True),
            keepalive_enabled=env_bool("GEMINI_KEEPALIVE_ENABLED", True),
            keepalive_timezone=os.getenv("GEMINI_KEEPALIVE_TIMEZONE", "Asia/Bangkok"),
            keepalive_hour=int(os.getenv("GEMINI_KEEPALIVE_HOUR", "0")),
            keepalive_minute=int(os.getenv("GEMINI_KEEPALIVE_MINUTE", "0")),
        )
