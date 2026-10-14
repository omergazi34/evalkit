"""Claude via the official Anthropic SDK."""

from __future__ import annotations

from typing import Any

from evalkit.config import ProviderConfig
from evalkit.providers.base import Completion, ProviderError, Request


class AnthropicProvider:
    """Calls ``messages.create`` and returns the concatenated text blocks.

    Thinking blocks are ignored: evals grade what the user would see. A refusal
    (``stop_reason == "refusal"``) is returned as data so it shows up in the report instead of
    being retried or silently rerouted. That is also why server-side model fallbacks are not
    enabled here: an eval must measure the model it names.
    """

    def __init__(self, config: ProviderConfig, client: Any | None = None) -> None:
        self.config = config
        self.label = config.label
        if client is None:
            import anthropic

            client = anthropic.AsyncAnthropic(timeout=config.timeout_s, max_retries=4)
        self._client = client

    def cache_identity(self) -> dict[str, Any]:
        c = self.config
        return {
            "type": "anthropic",
            "model": c.model,
            "max_tokens": c.max_tokens,
            "temperature": c.temperature,
            "effort": c.effort,
            "params": c.params,
        }

    def _request_kwargs(self, request: Request) -> dict[str, Any]:
        c = self.config
        kwargs: dict[str, Any] = {
            "model": c.model,
            "max_tokens": c.max_tokens,
            "messages": request.messages,
        }
        if request.system:
            kwargs["system"] = request.system
        # Only send sampling/effort when configured: newer models reject sampling params, and
        # leaving them out keeps each model on its own documented defaults.
        if c.temperature is not None:
            kwargs["temperature"] = c.temperature
        if c.effort is not None:
            kwargs["output_config"] = {"effort": c.effort}
        kwargs.update(c.params)
        return kwargs

    async def complete(self, request: Request) -> Completion:
        import anthropic

        try:
            response = await self._client.messages.create(**self._request_kwargs(request))
        except anthropic.APIStatusError as exc:
            raise ProviderError(f"{self.label}: HTTP {exc.status_code}: {exc.message}") from exc
        except anthropic.APIConnectionError as exc:
            raise ProviderError(f"{self.label}: connection error: {exc}") from exc

        text = "".join(block.text for block in response.content if block.type == "text")
        metadata: dict[str, Any] = {}
        refused = response.stop_reason == "refusal"
        details = getattr(response, "stop_details", None)
        if refused and details is not None:
            metadata["refusal_category"] = getattr(details, "category", None)
        request_id = getattr(response, "_request_id", None)
        if request_id:
            metadata["request_id"] = request_id
        return Completion(
            text=text,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            stop_reason=response.stop_reason,
            refused=refused,
            metadata=metadata,
        )

    async def aclose(self) -> None:
        close = getattr(self._client, "close", None)
        if close is not None:
            await close()
