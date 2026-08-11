from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any

from providers.base import LLMProvider


class ProviderScheduler:
    def __init__(self, providers: list[LLMProvider]) -> None:
        self.providers = providers
        self._tasks: list[asyncio.Task[Any]] = []
        self._stop = asyncio.Event()

    async def start(self) -> None:
        if self._tasks:
            return
        self._stop.clear()
        for provider in self.providers:
            if provider.keepalive_policy.enabled:
                self._tasks.append(
                    asyncio.create_task(
                        self._run_provider(provider),
                        name=f"keepalive:{provider.name}",
                    )
                )

    async def _run_provider(self, provider: LLMProvider) -> None:
        while not self._stop.is_set():
            next_run_at = provider.keepalive_policy.next_run()
            current_time = datetime.now(next_run_at.tzinfo)
            delay = max(0.0, (next_run_at - current_time).total_seconds())
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=delay)
                return
            except asyncio.TimeoutError:
                try:
                    await provider.keepalive()
                except Exception:
                    # A provider failure must not terminate the generic schedule.
                    # The provider owns its status/error reporting and retry policy.
                    continue

    async def stop(self) -> None:
        self._stop.set()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
