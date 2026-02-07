from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Protocol, TypedDict, cast
from uuid import uuid4

from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.runnables.config import RunnableConfig

from agent_kit.config import InvocationConfig


class ToolCall(TypedDict):
    id: str
    name: str
    args: dict[str, Any]


@dataclass(slots=True)
class ModelResponse:
    message: AIMessage
    tool_calls: list[ToolCall] = field(default_factory=list)
    tokens: list[str] = field(default_factory=list)


class ModelAdapter(Protocol):
    def complete(self, messages: list[BaseMessage], config: InvocationConfig) -> ModelResponse: ...

    async def acomplete(
        self, messages: list[BaseMessage], config: InvocationConfig
    ) -> ModelResponse: ...


class RunnableLike(Protocol):
    def invoke(
        self,
        input: object,
        config: RunnableConfig | None = None,
        **kwargs: object,
    ) -> BaseMessage: ...

    async def ainvoke(
        self,
        input: object,
        config: RunnableConfig | None = None,
        **kwargs: object,
    ) -> BaseMessage: ...


class LangChainModelAdapter:
    def __init__(self, model: object) -> None:
        if not hasattr(model, "invoke") or not hasattr(model, "ainvoke"):
            raise TypeError("LangChainModelAdapter requires a model with invoke and ainvoke.")
        self._model = cast(RunnableLike, model)

    def complete(self, messages: list[BaseMessage], config: InvocationConfig) -> ModelResponse:
        config_map = self._build_config(config)
        response = self._model.invoke(messages, config=config_map)
        message = self._to_ai_message(response)
        return ModelResponse(
            message=message,
            tool_calls=self._extract_tool_calls(message),
            tokens=self._extract_tokens(message),
        )

    async def acomplete(
        self, messages: list[BaseMessage], config: InvocationConfig
    ) -> ModelResponse:
        config_map = self._build_config(config)
        response = await self._model.ainvoke(messages, config=config_map)
        message = self._to_ai_message(response)
        return ModelResponse(
            message=message,
            tool_calls=self._extract_tool_calls(message),
            tokens=self._extract_tokens(message),
        )

    def _build_config(self, config: InvocationConfig) -> RunnableConfig:
        built: dict[str, Any] = {}
        if config.configurable:
            built["configurable"] = dict(config.configurable)
        if config.tags:
            built["tags"] = list(config.tags)
        if config.thread_id:
            configurable = built.setdefault("configurable", {})
            configurable["thread_id"] = config.thread_id
        return cast(RunnableConfig, built)

    def _to_ai_message(self, response: object) -> AIMessage:
        if isinstance(response, AIMessage):
            return response
        if isinstance(response, BaseMessage):
            return AIMessage(content=str(response.content))
        content = getattr(response, "content", response)
        return AIMessage(content=str(content))

    def _extract_tokens(self, message: AIMessage) -> list[str]:
        if isinstance(message.content, str) and message.content:
            return [message.content]
        return []

    def _extract_tool_calls(self, message: AIMessage) -> list[ToolCall]:
        tool_calls_raw = getattr(message, "tool_calls", [])
        if not tool_calls_raw and isinstance(message.additional_kwargs, dict):
            tool_calls_raw = message.additional_kwargs.get("tool_calls", [])
        parsed: list[ToolCall] = []
        for call in tool_calls_raw:
            call_id = str(call.get("id") or f"call-{uuid4().hex[:8]}")
            name = str(call.get("name") or "")
            args_value = call.get("args") or {}
            if isinstance(args_value, str):
                try:
                    parsed_args = json.loads(args_value)
                except json.JSONDecodeError:
                    parsed_args = {}
            elif isinstance(args_value, dict):
                parsed_args = dict(args_value)
            else:
                parsed_args = {}
            parsed.append({"id": call_id, "name": name, "args": parsed_args})
        return parsed
