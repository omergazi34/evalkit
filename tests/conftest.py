from __future__ import annotations

import asyncio
from typing import Any

import pytest

from evalkit.config import ProviderConfig, Suite
from evalkit.providers.base import Completion, ProviderError, Request


def make_suite(**overrides: Any) -> Suite:
    data: dict[str, Any] = {
        "name": "test",
        "provider": {"type": "mock"},
        "cases": [{"id": "echo", "input": "hello"}],
    }
    for key, value in overrides.items():
        if value is None:
            data.pop(key, None)
        else:
            data[key] = value
    return Suite.model_validate(data)


class ScriptedProvider:
    """Test double: returns queued outputs (or a function of the request) and records calls."""

    def __init__(
        self, label: str = "scripted", outputs: Any = None, fail: bool = False, delay: float = 0.0
    ):
        self.label = label
        self.outputs = outputs if outputs is not None else (lambda req: req.messages[-1]["content"])
        self.fail = fail
        self.delay = delay
        self.calls: list[Request] = []
        self.in_flight = 0
        self.max_in_flight = 0

    def cache_identity(self) -> dict[str, Any]:
        return {"type": "scripted", "label": self.label}

    async def complete(self, request: Request) -> Completion:
        self.calls.append(request)
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            if self.delay:
                await asyncio.sleep(self.delay)
            if self.fail:
                raise ProviderError(f"{self.label}: boom")
            text = self.outputs(request) if callable(self.outputs) else self.outputs.pop(0)
            return Completion(text=text, input_tokens=10, output_tokens=5, stop_reason="end_turn")
        finally:
            self.in_flight -= 1

    async def aclose(self) -> None:
        return None


@pytest.fixture
def mock_config() -> ProviderConfig:
    return ProviderConfig(type="mock")
