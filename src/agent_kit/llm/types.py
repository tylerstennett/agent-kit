from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, TypedDict

from langchain_core.messages import AIMessage, BaseMessage

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
