from __future__ import annotations

import asyncio
from typing import Any

import pytest
from conduit import Conduit, SyncConduit
from conduit.config import VLLMConfig
from conduit.models.messages import ChatResponse, Message, Role
from conduit.tools.schema import ToolCall as ConduitToolCall
from conduit.tools.schema import ToolDefinition
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from agent_kit import Agent, AgentConfig, tool
from agent_kit.config import InvocationConfig
from agent_kit.llm.conduit_adapter import ConduitModelAdapter, tool_schemas_to_conduit_tools
from agent_kit.state import AgentState


@tool(name="echo")
def echo_tool(state: AgentState, text: str) -> dict[str, str]:
    del state
    return {"echo": text}


@pytest.mark.asyncio
async def test_conduit_model_adapter_maps_messages_response_and_config_overrides() -> None:
    model = Conduit(VLLMConfig(model="m"))
    seen: dict[str, Any] = {}

    async def fake_chat(
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        stream: bool = False,
        config_overrides: dict[str, Any] | None = None,
    ) -> ChatResponse:
        del stream
        seen["messages"] = messages
        seen["tools"] = tools
        seen["tool_choice"] = tool_choice
        seen["config_overrides"] = config_overrides
        return ChatResponse(
            content="done",
            tool_calls=[
                ConduitToolCall(
                    id="c1",
                    name="echo",
                    arguments={"text": "hello"},
                )
            ],
        )

    model.chat = fake_chat  # type: ignore[method-assign]

    schemas = [
        {
            "name": "echo",
            "description": "Echo text.",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        }
    ]
    adapter = ConduitModelAdapter(model, tools=tool_schemas_to_conduit_tools(schemas))

    response = await adapter.acomplete(
        [
            SystemMessage(content="System"),
            HumanMessage(content="Hello"),
            AIMessage(
                content="",
                tool_calls=[{"id": "call-1", "name": "echo", "args": {"text": "a"}}],
            ),
            ToolMessage(content='{"status":"ok"}', tool_call_id="call-1", name="echo"),
        ],
        InvocationConfig(
            configurable={"temperature": 0.2},
            thread_id="thread-1",
            tags=["tag-1"],
        ),
    )

    await model.aclose()

    assert seen["config_overrides"] == {"temperature": 0.2}
    assert seen["tool_choice"] == "auto"
    assert seen["tools"] is not None
    seen_tools = seen["tools"]
    assert isinstance(seen_tools, list)
    assert seen_tools[0].name == "echo"

    seen_messages = seen["messages"]
    assert isinstance(seen_messages, list)
    assert seen_messages[0].role == Role.SYSTEM
    assert seen_messages[1].role == Role.USER
    assert seen_messages[2].role == Role.ASSISTANT
    assert seen_messages[2].tool_calls is not None
    assert seen_messages[2].tool_calls[0].name == "echo"
    assert seen_messages[3].role == Role.TOOL
    assert seen_messages[3].tool_call_id == "call-1"

    assert response.tokens == ["done"]
    assert response.tool_calls == [{"id": "c1", "name": "echo", "args": {"text": "hello"}}]


@pytest.mark.asyncio
async def test_agent_auto_wraps_async_conduit_for_async_calls() -> None:
    model = Conduit(VLLMConfig(model="m"))

    async def fake_chat(
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        stream: bool = False,
        config_overrides: dict[str, Any] | None = None,
    ) -> ChatResponse:
        del messages, tools, tool_choice, stream, config_overrides
        return ChatResponse(content="done")

    model.chat = fake_chat  # type: ignore[method-assign]
    agent = Agent(model=model, tools=[])

    state = await agent.arun("hi")
    assert state["termination"]["reason"] == "completed"

    await model.aclose()


def test_agent_with_async_conduit_blocks_sync_calls() -> None:
    model = Conduit(VLLMConfig(model="m"))
    agent = Agent(model=model, tools=[])

    with pytest.raises(TypeError, match="async-only"):
        agent.run("hi")

    asyncio.run(model.aclose())


def test_agent_auto_wraps_sync_conduit_for_sync_calls() -> None:
    model = SyncConduit(VLLMConfig(model="m"))

    def fake_chat(
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        stream: bool = False,
        config_overrides: dict[str, Any] | None = None,
    ) -> ChatResponse:
        del messages, tools, tool_choice, stream, config_overrides
        return ChatResponse(content="done")

    model.chat = fake_chat  # type: ignore[method-assign]
    agent = Agent(model=model, tools=[])

    state = agent.run("hi")
    assert state["termination"]["reason"] == "completed"

    model.close()


@pytest.mark.asyncio
async def test_agent_with_sync_conduit_blocks_async_calls() -> None:
    model = SyncConduit(VLLMConfig(model="m"))
    agent = Agent(model=model, tools=[])

    with pytest.raises(TypeError, match="sync-only"):
        await agent.arun("hi")

    await asyncio.to_thread(model.close)


def test_agent_conduit_auto_mode_sends_tool_schemas_and_records_signature() -> None:
    model = SyncConduit(VLLMConfig(model="m"))
    seen_tools: list[ToolDefinition] | None = None

    def fake_chat(
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        stream: bool = False,
        config_overrides: dict[str, Any] | None = None,
    ) -> ChatResponse:
        nonlocal seen_tools
        del messages, tool_choice, stream, config_overrides
        seen_tools = tools
        return ChatResponse(content="done")

    model.chat = fake_chat  # type: ignore[method-assign]
    agent = Agent(model=model, tools=[echo_tool])

    state = agent.run("hi")
    model.close()

    assert state["termination"]["reason"] == "completed"
    assert seen_tools is not None
    assert seen_tools[0].name == "echo"
    assert agent.last_tool_schema_signature is not None


def test_agent_conduit_manual_mode_skips_tool_schemas() -> None:
    model = SyncConduit(VLLMConfig(model="m"))
    seen_tools: list[ToolDefinition] | None = None

    def fake_chat(
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        stream: bool = False,
        config_overrides: dict[str, Any] | None = None,
    ) -> ChatResponse:
        nonlocal seen_tools
        del messages, tool_choice, stream, config_overrides
        seen_tools = tools
        return ChatResponse(content="done")

    model.chat = fake_chat  # type: ignore[method-assign]
    agent = Agent(
        model=model,
        tools=[echo_tool],
        config=AgentConfig(model_tool_binding_mode="manual"),
    )

    state = agent.run("hi")
    model.close()

    assert state["termination"]["reason"] == "completed"
    assert seen_tools is None
    assert agent.last_tool_schema_signature is None


def test_agent_conduit_off_mode_skips_tool_schemas() -> None:
    model = SyncConduit(VLLMConfig(model="m"))
    seen_tools: list[ToolDefinition] | None = None

    def fake_chat(
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        stream: bool = False,
        config_overrides: dict[str, Any] | None = None,
    ) -> ChatResponse:
        nonlocal seen_tools
        del messages, tool_choice, stream, config_overrides
        seen_tools = tools
        return ChatResponse(content="done")

    model.chat = fake_chat  # type: ignore[method-assign]
    agent = Agent(
        model=model,
        tools=[echo_tool],
        config=AgentConfig(model_tool_binding_mode="off"),
    )

    state = agent.run("hi")
    model.close()

    assert state["termination"]["reason"] == "completed"
    assert seen_tools is None
    assert agent.last_tool_schema_signature is None


def test_agent_conduit_uses_only_configurable_as_overrides() -> None:
    model = SyncConduit(VLLMConfig(model="m"))
    seen_config_overrides: dict[str, Any] | None = None

    def fake_chat(
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        stream: bool = False,
        config_overrides: dict[str, Any] | None = None,
    ) -> ChatResponse:
        nonlocal seen_config_overrides
        del messages, tools, tool_choice, stream
        seen_config_overrides = config_overrides
        return ChatResponse(content="done")

    model.chat = fake_chat  # type: ignore[method-assign]
    agent = Agent(model=model, tools=[])

    state = agent.run(
        "hi",
        config=InvocationConfig(
            configurable={"temperature": 0.1},
            thread_id="thread-1",
            tags=["tag-1"],
        ),
    )
    model.close()

    assert state["termination"]["reason"] == "completed"
    assert seen_config_overrides == {"temperature": 0.1}


def test_sync_conduit_blocks_acomplete_directly() -> None:
    model = SyncConduit(VLLMConfig(model="m"))
    adapter = ConduitModelAdapter(model)

    with pytest.raises(TypeError, match="sync-only"):
        asyncio.run(adapter.acomplete([HumanMessage(content="hi")], InvocationConfig()))

    model.close()


def test_async_conduit_blocks_complete_directly() -> None:
    model = Conduit(VLLMConfig(model="m"))
    adapter = ConduitModelAdapter(model)

    with pytest.raises(TypeError, match="async-only"):
        adapter.complete([HumanMessage(content="hi")], InvocationConfig())

    asyncio.run(model.aclose())
