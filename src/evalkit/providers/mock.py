"""A deterministic provider for offline demos, CI dogfooding and tests."""

from __future__ import annotations

import asyncio
from typing import Any

from evalkit.config import ProviderConfig
from evalkit.providers.base import Completion, Request


class MockProvider:
    """Returns ``responses[case_id]``, else ``default_response``, else echoes the last user turn."""

    def __init__(self, config: ProviderConfig) -> None:
        self.config = config
        self.label = config.label
        self.calls = 0

    def cache_identity(self) -> dict[str, Any]:
        return {
            "type": "mock",
            "responses": self.config.responses,
            "default": self.config.default_response,
        }

    async def complete(self, request: Request) -> Completion:
        self.calls += 1
        if self.config.latency_ms:
            await asyncio.sleep(self.config.latency_ms / 1000)
        if request.case_id is not None and request.case_id in self.config.responses:
            text = self.config.responses[request.case_id]
        elif self.config.default_response is not None:
            text = self.config.default_response
        else:
            text = request.messages[-1]["content"]
        return Completion(text=text, input_tokens=0, output_tokens=0, stop_reason="end_turn")

    async def aclose(self) -> None:
        return None
