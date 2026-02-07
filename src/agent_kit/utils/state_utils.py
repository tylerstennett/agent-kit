from __future__ import annotations

from collections.abc import Iterable
from copy import deepcopy
from typing import Any

from langchain_core.messages import BaseMessage

from agent_kit.config import ToolResult
from agent_kit.state import AgentState


def merge_metadata(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    merged.update(update)
    return merged


def clone_state(state: AgentState) -> AgentState:
    return deepcopy(state)


def _message_delta(before: list[BaseMessage], after: list[BaseMessage]) -> list[BaseMessage]:
    if len(after) <= len(before):
        return []
    return list(after[len(before) :])


def _tool_output_delta(before: list[ToolResult], after: list[ToolResult]) -> list[ToolResult]:
    if len(after) <= len(before):
        return []
    return list(after[len(before) :])


def _dict_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    delta: dict[str, Any] = {}
    for key, value in after.items():
        if key not in before or before[key] != value:
            delta[key] = value
    return delta


def state_delta(before: AgentState, after: AgentState) -> dict[str, Any]:
    before_messages = list(before.get("messages", []))
    after_messages = list(after.get("messages", []))
    before_outputs = list(before.get("tool_outputs", []))
    after_outputs = list(after.get("tool_outputs", []))
    before_metadata = dict(before.get("metadata", {}))
    after_metadata = dict(after.get("metadata", {}))
    before_routing = dict(before.get("routing_hints", {}))
    after_routing = dict(after.get("routing_hints", {}))
    before_termination = dict(before.get("termination", {}))
    after_termination = dict(after.get("termination", {}))

    delta: dict[str, Any] = {}
    message_delta = _message_delta(before_messages, after_messages)
    if message_delta:
        delta["messages"] = message_delta
    output_delta = _tool_output_delta(before_outputs, after_outputs)
    if output_delta:
        delta["tool_outputs"] = output_delta

    metadata_delta = _dict_delta(before_metadata, after_metadata)
    if metadata_delta:
        delta["metadata"] = metadata_delta
    routing_delta = _dict_delta(before_routing, after_routing)
    if routing_delta:
        delta["routing_hints"] = routing_delta
    termination_delta = _dict_delta(before_termination, after_termination)
    if termination_delta:
        delta["termination"] = termination_delta
    return delta


def apply_delta(state: AgentState, delta: dict[str, Any]) -> AgentState:
    messages = list(state.get("messages", []))
    tool_outputs = list(state.get("tool_outputs", []))
    metadata = dict(state.get("metadata", {}))
    routing_hints = dict(state.get("routing_hints", {}))
    termination = dict(state.get("termination", {}))

    messages.extend(delta.get("messages", []))
    tool_outputs.extend(delta.get("tool_outputs", []))
    metadata.update(delta.get("metadata", {}))
    routing_hints.update(delta.get("routing_hints", {}))
    termination.update(delta.get("termination", {}))

    state["messages"] = messages
    state["tool_outputs"] = tool_outputs
    state["metadata"] = metadata
    state["routing_hints"] = routing_hints
    state["termination"] = termination
    return state


def merge_parallel_state_deltas(
    base_state: AgentState,
    deltas: Iterable[dict[str, Any]],
) -> AgentState:
    merged = clone_state(base_state)
    for delta in deltas:
        apply_delta(merged, delta)
    return merged
