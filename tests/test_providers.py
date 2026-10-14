from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from evalkit.config import ProviderConfig
from evalkit.providers import build_provider
from evalkit.providers.base import ProviderError, Request
from evalkit.providers.claude import AnthropicProvider
from evalkit.providers.mock import MockProvider
from evalkit.providers.openai_compat import OpenAICompatProvider

REQUEST = Request(messages=[{"role": "user", "content": "hi"}], system="be brief", case_id="c1")


def chat_response(text: str = "hello", finish: str = "stop") -> dict[str, Any]:
    return {
        "choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": finish}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 3},
    }


def openai_provider(handler: Any, **config: Any) -> OpenAICompatProvider:
    cfg = ProviderConfig(type="openai", model="llama3", base_url="http://local/v1", **config)
    return OpenAICompatProvider(cfg, transport=httpx.MockTransport(handler), backoff_s=0)


def test_openai_compat_sends_system_first_and_parses_usage() -> None:
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=chat_response())

    completion = asyncio.run(
        openai_provider(handler, temperature=0, params={"seed": 7}).complete(REQUEST)
    )
    assert completion.text == "hello"
    assert (completion.input_tokens, completion.output_tokens) == (12, 3)
    body = seen[0]
    assert body["messages"][0] == {"role": "system", "content": "be brief"}
    assert body["temperature"] == 0
    assert body["seed"] == 7


def test_openai_compat_retries_transient_errors() -> None:
    statuses = iter([429, 503, 200])

    def handler(_: httpx.Request) -> httpx.Response:
        status = next(statuses)
        return httpx.Response(
            status, json=chat_response() if status == 200 else {"error": "slow down"}
        )

    assert asyncio.run(openai_provider(handler).complete(REQUEST)).text == "hello"


def test_openai_compat_does_not_retry_client_errors() -> None:
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(400, text="bad model")

    with pytest.raises(ProviderError, match="HTTP 400"):
        asyncio.run(openai_provider(handler).complete(REQUEST))
    assert calls == 1


def test_openai_compat_gives_up_after_max_retries() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(ProviderError, match="after 5 attempts"):
        asyncio.run(openai_provider(handler).complete(REQUEST))


def test_openai_compat_rejects_unexpected_shapes() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"oops": True})

    with pytest.raises(ProviderError, match="unexpected response shape"):
        asyncio.run(openai_provider(handler).complete(REQUEST))


def test_openai_compat_requires_configured_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EVALKIT_TEST_KEY", raising=False)
    with pytest.raises(ProviderError, match="EVALKIT_TEST_KEY"):
        openai_provider(lambda r: httpx.Response(200), api_key_env="EVALKIT_TEST_KEY")


def test_openai_compat_sends_bearer_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EVALKIT_TEST_KEY", "sk-test")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer sk-test"
        return httpx.Response(200, json=chat_response())

    asyncio.run(openai_provider(handler, api_key_env="EVALKIT_TEST_KEY").complete(REQUEST))


class FakeMessages:
    def __init__(self, response: Any) -> None:
        self.response = response
        self.kwargs: dict[str, Any] = {}

    async def create(self, **kwargs: Any) -> Any:
        self.kwargs = kwargs
        return self.response


def fake_message(*blocks: Any, stop_reason: str = "end_turn", stop_details: Any = None) -> Any:
    return SimpleNamespace(
        content=list(blocks),
        stop_reason=stop_reason,
        stop_details=stop_details,
        usage=SimpleNamespace(input_tokens=20, output_tokens=4),
        _request_id="req_123",
    )


def test_anthropic_request_only_sends_configured_params() -> None:
    messages = FakeMessages(fake_message(SimpleNamespace(type="text", text="ok")))
    cfg = ProviderConfig(type="anthropic", model="claude-opus-5-5", effort="low", max_tokens=256)
    provider = AnthropicProvider(cfg, client=SimpleNamespace(messages=messages))
    completion = asyncio.run(provider.complete(REQUEST))
    assert completion.text == "ok"
    assert completion.metadata["request_id"] == "req_123"
    assert messages.kwargs == {
        "model": "claude-opus-5-5",
        "max_tokens": 256,
        "messages": REQUEST.messages,
        "system": "be brief",
        "output_config": {"effort": "low"},
    }


def test_anthropic_ignores_thinking_blocks_and_joins_text() -> None:
    response = fake_message(
        SimpleNamespace(type="thinking", thinking=""),
        SimpleNamespace(type="text", text="Hello, "),
        SimpleNamespace(type="text", text="world"),
    )
    provider = AnthropicProvider(
        ProviderConfig(type="anthropic", model="m"),
        client=SimpleNamespace(messages=FakeMessages(response)),
    )
    assert asyncio.run(provider.complete(REQUEST)).text == "Hello, world"


def test_anthropic_refusal_is_returned_as_data() -> None:
    response = fake_message(stop_reason="refusal", stop_details=SimpleNamespace(category="cyber"))
    provider = AnthropicProvider(
        ProviderConfig(type="anthropic", model="m"),
        client=SimpleNamespace(messages=FakeMessages(response)),
    )
    completion = asyncio.run(provider.complete(REQUEST))
    assert completion.refused
    assert completion.metadata["refusal_category"] == "cyber"


def test_mock_provider_resolution_order() -> None:
    cfg = ProviderConfig(type="mock", responses={"c1": "scripted"}, default_response="default")
    mock = MockProvider(cfg)
    assert asyncio.run(mock.complete(REQUEST)).text == "scripted"
    other = Request(messages=[{"role": "user", "content": "x"}], case_id="c2")
    assert asyncio.run(mock.complete(other)).text == "default"
    echo = MockProvider(ProviderConfig(type="mock"))
    assert asyncio.run(echo.complete(other)).text == "x"


def test_build_provider_dispatches_on_type() -> None:
    assert isinstance(build_provider(ProviderConfig(type="mock")), MockProvider)
    provider = build_provider(ProviderConfig(type="openai", model="m", base_url="http://x/v1"))
    assert isinstance(provider, OpenAICompatProvider)
    asyncio.run(provider.aclose())
