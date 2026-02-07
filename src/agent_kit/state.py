from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage

from agent_kit.config import ToolResult


def append_messages(current: list[BaseMessage], update: Sequence[BaseMessage]) -> list[BaseMessage]:
    return [*current, *list(update)]


def append_tool_outputs(
    current: list[ToolResult], update: Sequence[ToolResult]
) -> list[ToolResult]:
    return [*current, *list(update)]


def merge_dict(current: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    merged = dict(current)
    merged.update(update)
    return merged


class AgentState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], append_messages]
    tool_outputs: Annotated[list[ToolResult], append_tool_outputs]
    metadata: Annotated[dict[str, Any], merge_dict]
    routing_hints: Annotated[dict[str, Any], merge_dict]
    termination: Annotated[dict[str, Any], merge_dict]


def make_initial_state(
    messages: Sequence[BaseMessage] | None = None,
    metadata: dict[str, Any] | None = None,
) -> AgentState:
    return {
        "messages": list(messages or []),
        "tool_outputs": [],
        "metadata": dict(metadata or {}),
        "routing_hints": {},
        "termination": {},
    }
