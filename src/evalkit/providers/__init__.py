from __future__ import annotations

from evalkit.config import ProviderConfig
from evalkit.providers.base import Completion, Provider, ProviderError, Request


def build_provider(config: ProviderConfig) -> Provider:
    if config.type == "anthropic":
        from evalkit.providers.claude import AnthropicProvider

        return AnthropicProvider(config)
    if config.type == "openai":
        from evalkit.providers.openai_compat import OpenAICompatProvider

        return OpenAICompatProvider(config)
    from evalkit.providers.mock import MockProvider

    return MockProvider(config)


__all__ = ["Completion", "Provider", "ProviderError", "Request", "build_provider"]
