"""Any server that speaks the OpenAI Chat Completions protocol: Ollama, LM Studio, vLLM, OpenAI."""

from __future__ import annotations

import asyncio
import os
import random
from typing import Any

import httpx

from evalkit.config import ProviderConfig
from evalkit.providers.base import Completion, ProviderError, Request

DEFAULT_BASE_URL = "https://api.openai.com/v1"
_RETRYABLE = {408, 409, 429, 500, 502, 503, 504}


class OpenAICompatProvider:
    def __init__(
        self,
        config: ProviderConfig,
        transport: httpx.AsyncBaseTransport | None = None,
        max_retries: int = 4,
        backoff_s: float = 0.5,
    ) -> None:
        self.config = config
        self.label = config.label
        self.max_retries = max_retries
        self.backoff_s = backoff_s
        headers = {"Content-Type": "application/json"}
        if config.api_key_env:
            key = os.environ.get(config.api_key_env)
            if not key:
                raise ProviderError(
                    f"{self.label}: environment variable {config.api_key_env} is not set"
                )
            headers["Authorization"] = f"Bearer {key}"
        self._client = httpx.AsyncClient(
            base_url=(config.base_url or DEFAULT_BASE_URL).rstrip("/"),
            headers=headers,
            timeout=config.timeout_s,
            transport=transport,
        )

    def cache_identity(self) -> dict[str, Any]:
        c = self.config
        return {
            "type": "openai",
            "base_url": c.base_url or DEFAULT_BASE_URL,
            "model": c.model,
            "max_tokens": c.max_tokens,
            "temperature": c.temperature,
            "params": c.params,
        }

    def _payload(self, request: Request) -> dict[str, Any]:
        c = self.config
        messages: list[dict[str, str]] = []
        if request.system:
            messages.append({"role": "system", "content": request.system})
        messages.extend(request.messages)
        payload: dict[str, Any] = {
            "model": c.model,
            "messages": messages,
            "max_tokens": c.max_tokens,
        }
        if c.temperature is not None:
            payload["temperature"] = c.temperature
        payload.update(c.params)
        return payload

    async def complete(self, request: Request) -> Completion:
        payload = self._payload(request)
        for attempt in range(self.max_retries + 1):
            try:
                response = await self._client.post("/chat/completions", json=payload)
            except httpx.TransportError as exc:
                error: str = f"connection error: {exc}"
            else:
                if response.status_code < 400:
                    return self._parse(response.json())
                error = f"HTTP {response.status_code}: {response.text[:300]}"
                if response.status_code not in _RETRYABLE:
                    raise ProviderError(f"{self.label}: {error}")
            if attempt < self.max_retries:
                # Exponential backoff with full jitter.
                await asyncio.sleep(random.uniform(0, self.backoff_s * 2**attempt))
        raise ProviderError(f"{self.label}: {error} (after {self.max_retries + 1} attempts)")

    def _parse(self, body: dict[str, Any]) -> Completion:
        try:
            choice = body["choices"][0]
            text = choice["message"].get("content") or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(f"{self.label}: unexpected response shape: {body!r:.300}") from exc
        usage = body.get("usage") or {}
        finish = choice.get("finish_reason")
        return Completion(
            text=text,
            input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
            stop_reason=finish,
            refused=finish == "content_filter",
        )

    async def aclose(self) -> None:
        await self._client.aclose()
