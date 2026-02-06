from __future__ import annotations

from collections.abc import Sequence

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from agent_kit import Agent, tool
from agent_kit.config import AgentConfig
from agent_kit.errors import ModelToolBindingError, ToolSchemaConversionError
from agent_kit.state import AgentState
from agent_kit.tools.schema import tools_to_model_schemas
from tests.fakes import FakeModelAdapter, make_response


@tool(name="echo")
def echo_tool(state: AgentState, text: str) -> dict[str, str]:
    return {"echo": text}


class UnsupportedPayload:
    pass


@tool(name="unsupported")
def unsupported_tool(state: AgentState, payload: UnsupportedPayload) -> dict[str, str]:
    return {"ok": "true"}


class RecordingChatModel(BaseChatModel):
    responses: list[AIMessage] = Field(default_factory=lambda: [AIMessage(content="done")])
    bind_calls: int = 0
    fail_bind: bool = False
    response_index: int = 0
    bound_tools: list[dict[str, object]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "recording-chat-model"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: object | None = None,
        **kwargs: object,
    ) -> ChatResult:
        del messages, stop, run_manager, kwargs
        index = min(self.response_index, len(self.responses) - 1)
        self.response_index += 1
        response = self.responses[index]
        return ChatResult(generations=[ChatGeneration(message=response)])

    def bind_tools(
        self,
        tools: Sequence[object],
        *,
        tool_choice: str | None = None,
        **kwargs: object,
    ) -> object:
        del tool_choice, kwargs
        if self.fail_bind:
            raise RuntimeError("bind failure")
        schemas: list[dict[str, object]] = []
        for item in tools:
            if isinstance(item, dict):
                schemas.append(dict(item))
        self.bind_calls += 1
        self.bound_tools = schemas
        return self


def test_auto_binding_uses_runtime_tools_for_base_chat_model() -> None:
    model = RecordingChatModel(responses=[AIMessage(content="done")])
    agent = Agent(model=model, tools=[echo_tool])

    state = agent.run("hi")

    assert state["termination"]["reason"] == "completed"
    assert model.bind_calls == 1
    assert model.bound_tools
    assert model.bound_tools[0]["name"] == "echo"
    assert agent.last_tool_schema_signature is not None


def test_auto_binding_skips_when_no_tools_for_bindable_model() -> None:
    model = RecordingChatModel(responses=[AIMessage(content="done")])
    agent = Agent(model=model, tools=[])

    state = agent.run("hi")

    assert state["termination"]["reason"] == "completed"
    assert model.bind_calls == 0
    assert agent.last_tool_schema_signature is None


def test_auto_mode_allows_no_tool_run_for_non_bindable_model() -> None:
    model = FakeListChatModel(responses=["done"])
    agent = Agent(model=model, tools=[])

    state = agent.run("hi")

    assert state["termination"]["reason"] == "completed"
    assert agent.last_tool_schema_signature is None


def test_auto_binding_fails_when_base_model_cannot_bind_tools() -> None:
    model = FakeListChatModel(responses=["done"])
    agent = Agent(model=model, tools=[echo_tool])

    with pytest.raises(ModelToolBindingError):
        agent.run("hi")


def test_auto_binding_treats_custom_model_adapter_as_pre_bound() -> None:
    adapter = FakeModelAdapter(responses=[make_response("done")])
    agent = Agent(model=adapter, tools=[echo_tool], config=AgentConfig())

    state = agent.run("hi")

    assert state["termination"]["reason"] == "completed"
    assert agent.last_tool_schema_signature is None


def test_strict_policy_raises_for_unsupported_annotations() -> None:
    model = RecordingChatModel(responses=[AIMessage(content="done")])
    agent = Agent(model=model, tools=[unsupported_tool], config=AgentConfig())

    with pytest.raises(ToolSchemaConversionError):
        agent.run("hi")


def test_warn_policy_falls_back_to_permissive_schema_with_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level("WARNING")
    schemas = tools_to_model_schemas([unsupported_tool], policy="warn")

    parameters = schemas[0]["parameters"]
    assert isinstance(parameters, dict)
    properties = parameters["properties"]
    assert isinstance(properties, dict)
    assert properties["payload"] == {}
    assert any("Unsupported annotation" in record.message for record in caplog.records)


def test_ignore_policy_falls_back_to_permissive_schema_without_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level("WARNING")
    caplog.clear()
    schemas = tools_to_model_schemas([unsupported_tool], policy="ignore")

    parameters = schemas[0]["parameters"]
    assert isinstance(parameters, dict)
    properties = parameters["properties"]
    assert isinstance(properties, dict)
    assert properties["payload"] == {}
    assert caplog.records == []
