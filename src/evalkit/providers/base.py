from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class Completion:
    """What a provider returns for one request. Providers never raise for model-level outcomes
    such as refusals; those are data. They raise ``ProviderError`` only for transport failures."""

    text: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    stop_reason: str | None = None
    refused: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Request:
    messages: list[dict[str, str]]
    system: str | None = None
    case_id: str | None = None


class ProviderError(Exception):
    """A request failed after the provider's own retries were exhausted."""


class Provider(Protocol):
    label: str

    def cache_identity(self) -> dict[str, Any]:
        """Everything that changes the output for a given request (model, params, ...)."""
        ...

    async def complete(self, request: Request) -> Completion: ...

    async def aclose(self) -> None: ...
