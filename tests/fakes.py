from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage

from agent_kit.llm.types import ModelResponse, ToolCall
from agent_kit.state import AgentState
from agent_kit.tools.base import BaseTool


@dataclass(slots=True)
class FakeModelAdapter:
    responses: list[ModelResponse]
    cursor: int = 0
    seen_messages: list[list[BaseMessage]] = field(default_factory=list)

    def complete(self, messages: list[BaseMessage], config: Any) -> ModelResponse:
        self.seen_messages.append(list(messages))
        response = self.responses[min(self.cursor, len(self.responses) - 1)]
        self.cursor += 1
        return response

    async def acomplete(self, messages: list[BaseMessage], config: Any) -> ModelResponse:
        return self.complete(messages, config)


class EchoTool(BaseTool):
    def execute(self, state: AgentState, text: str) -> Any:
        return {"echo": text}


class MetadataTool(BaseTool):
    def execute(self, state: AgentState, key: str, value: str) -> Any:
        metadata = dict(state.get("metadata", {}))
        metadata[key] = value
        state["metadata"] = metadata
        return {"key": key, "value": value}


class SlowTool(BaseTool):
    async def aexecute(self, state: AgentState, delay: float) -> Any:
        import asyncio

        await asyncio.sleep(delay)
        return {"delay": delay}

    def execute(self, state: AgentState, delay: float) -> Any:
        return {"delay": delay}


def make_response(
    content: str,
    *,
    tool_calls: list[ToolCall] | None = None,
    tokens: list[str] | None = None,
) -> ModelResponse:
    message = AIMessage(content=content, tool_calls=tool_calls or [])
    return ModelResponse(
        message=message, tool_calls=list(tool_calls or []), tokens=list(tokens or [])
    )
