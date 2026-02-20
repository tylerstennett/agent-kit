from __future__ import annotations

import os
from typing import Any

import pytest
from conduit import Conduit, SyncConduit
from conduit.config import OpenRouterConfig
from conduit.models.messages import Message, RequestContext
from conduit.tools.schema import ToolDefinition
from dotenv import load_dotenv

load_dotenv()


def _api_key() -> str | None:
    key = os.environ.get("OPENROUTER_API_KEY")
    if key and key != "your-openrouter-api-key-here":
        return key
    return None


def _model() -> str:
    return os.environ.get("INTEGRATION_TEST_MODEL", "google/gemini-2.5-flash-lite")


requires_api_key = pytest.mark.skipif(
    _api_key() is None,
    reason="OPENROUTER_API_KEY not set; set it in .env to run integration tests",
)


class ConduitCallRecorder:
    """Wraps a SyncConduit to record arguments passed to ``chat()`` while
    still forwarding every call to the real implementation."""

    def __init__(self, client: SyncConduit) -> None:
        self.calls: list[dict[str, Any]] = []
        self._original_chat = client.chat

        def _recording_chat(
            messages: list[Message],
            tools: list[ToolDefinition] | None = None,
            tool_choice: str | dict[str, Any] | None = None,
            stream: bool = False,
            config_overrides: dict[str, Any] | None = None,
            context: RequestContext | None = None,
            runtime_overrides: dict[str, Any] | None = None,
        ) -> Any:
            self.calls.append(
                {
                    "messages": messages,
                    "tools": tools,
                    "tool_choice": tool_choice,
                    "stream": stream,
                    "config_overrides": config_overrides,
                    "context": context,
                    "runtime_overrides": runtime_overrides,
                }
            )
            return self._original_chat(
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                stream=stream,
                config_overrides=config_overrides,
                context=context,
                runtime_overrides=runtime_overrides,
            )

        client.chat = _recording_chat  # type: ignore[method-assign]


class AsyncConduitCallRecorder:
    """Wraps an async Conduit to record arguments passed to ``chat()`` while
    still forwarding every call to the real implementation."""

    def __init__(self, client: Conduit) -> None:
        self.calls: list[dict[str, Any]] = []
        self._original_chat = client.chat

        async def _recording_chat(
            messages: list[Message],
            tools: list[ToolDefinition] | None = None,
            tool_choice: str | dict[str, Any] | None = None,
            stream: bool = False,
            config_overrides: dict[str, Any] | None = None,
            context: RequestContext | None = None,
            runtime_overrides: dict[str, Any] | None = None,
        ) -> Any:
            self.calls.append(
                {
                    "messages": messages,
                    "tools": tools,
                    "tool_choice": tool_choice,
                    "stream": stream,
                    "config_overrides": config_overrides,
                    "context": context,
                    "runtime_overrides": runtime_overrides,
                }
            )
            return await self._original_chat(
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                stream=stream,
                config_overrides=config_overrides,
                context=context,
                runtime_overrides=runtime_overrides,
            )

        client.chat = _recording_chat  # type: ignore[method-assign]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def openrouter_api_key() -> str:
    key = _api_key()
    if not key:
        pytest.skip("OPENROUTER_API_KEY not set")
    return key


@pytest.fixture
def model_name() -> str:
    return _model()


@pytest.fixture
def openrouter_config(openrouter_api_key: str, model_name: str) -> OpenRouterConfig:
    return OpenRouterConfig(model=model_name, api_key=openrouter_api_key)


@pytest.fixture
def sync_conduit(openrouter_config: OpenRouterConfig) -> SyncConduit:
    client = SyncConduit(openrouter_config)
    yield client  # type: ignore[misc]
    client.close()


@pytest.fixture
async def async_conduit(openrouter_config: OpenRouterConfig) -> Conduit:
    client = Conduit(openrouter_config)
    yield client  # type: ignore[misc]
    await client.aclose()


@pytest.fixture
def sync_conduit_factory(openrouter_api_key: str, model_name: str):
    """Factory that creates fresh SyncConduit instances, cleaning them all up
    at the end of the test."""
    clients: list[SyncConduit] = []

    def _create() -> SyncConduit:
        config = OpenRouterConfig(model=model_name, api_key=openrouter_api_key)
        client = SyncConduit(config)
        clients.append(client)
        return client

    yield _create
    for client in clients:
        client.close()


@pytest.fixture
async def async_conduit_factory(openrouter_api_key: str, model_name: str):
    """Factory that creates fresh async Conduit instances, cleaning them all
    up at the end of the test."""
    clients: list[Conduit] = []

    def _create() -> Conduit:
        config = OpenRouterConfig(model=model_name, api_key=openrouter_api_key)
        client = Conduit(config)
        clients.append(client)
        return client

    yield _create
    for client in clients:
        await client.aclose()
