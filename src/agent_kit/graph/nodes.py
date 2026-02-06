from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any, cast

from langchain_core.messages import AIMessage, ToolMessage

from ..config import InvocationConfig, ToolResult
from ..events import LLMTokenEvent, StateUpdateEvent
from ..model_adapter import ModelAdapter, ToolCall
from ..state import AgentState
from ..tools.executor import ToolExecutor
from ..utils.state_utils import clone_state, state_delta

EventSink = Callable[[object], None]


def _serialize_tool_result(result: ToolResult) -> str:
    payload = {
        "status": result.status,
        "data": result.data,
        "error": {
            "code": result.error.code,
            "message": result.error.message,
            "retriable": result.error.retriable,
            "details": result.error.details,
        }
        if result.error
        else None,
    }
    try:
        return json.dumps(payload)
    except TypeError:
        payload["data"] = str(payload["data"])
        return json.dumps(payload)


def _to_tool_message(result: ToolResult) -> ToolMessage:
    return ToolMessage(
        content=_serialize_tool_result(result),
        name=result.tool_name,
        tool_call_id=result.call_id,
    )


def reasoning_step(
    state: AgentState,
    model: ModelAdapter,
    config: InvocationConfig,
    *,
    event_sink: EventSink | None = None,
) -> tuple[AgentState, list[ToolCall]]:
    before = clone_state(state)
    response = model.complete(list(state.get("messages", [])), config)
    messages = list(state.get("messages", []))
    messages.append(response.message)
    state["messages"] = messages
    if event_sink and config.emit_llm_tokens:
        for token in response.tokens:
            event_sink(LLMTokenEvent(run_id="", token=token))
    if event_sink:
        event_sink(StateUpdateEvent(run_id="", delta=state_delta(before, state)))
    return state, response.tool_calls


async def areasoning_step(
    state: AgentState,
    model: ModelAdapter,
    config: InvocationConfig,
    *,
    event_sink: EventSink | None = None,
) -> tuple[AgentState, list[ToolCall]]:
    before = clone_state(state)
    response = await model.acomplete(list(state.get("messages", [])), config)
    messages = list(state.get("messages", []))
    messages.append(response.message)
    state["messages"] = messages
    if event_sink and config.emit_llm_tokens:
        for token in response.tokens:
            event_sink(LLMTokenEvent(run_id="", token=token))
    if event_sink:
        event_sink(StateUpdateEvent(run_id="", delta=state_delta(before, state)))
    return state, response.tool_calls


def action_step(
    state: AgentState,
    tool_calls: list[ToolCall],
    executor: ToolExecutor,
    config: InvocationConfig,
    *,
    event_sink: EventSink | None = None,
) -> tuple[AgentState, list[ToolResult]]:
    before = clone_state(state)
    tool_call_dicts = cast(list[dict[str, Any]], list(tool_calls))
    state, results = executor.execute_calls(
        state, tool_call_dicts, config, event_sink=event_sink
    )
    messages = list(state.get("messages", []))
    tool_outputs = list(state.get("tool_outputs", []))
    for result in results:
        messages.append(_to_tool_message(result))
        tool_outputs.append(result)
    state["messages"] = messages
    state["tool_outputs"] = tool_outputs
    if event_sink:
        event_sink(StateUpdateEvent(run_id="", delta=state_delta(before, state)))
    return state, results


async def aaction_step(
    state: AgentState,
    tool_calls: list[ToolCall],
    executor: ToolExecutor,
    config: InvocationConfig,
    *,
    event_sink: EventSink | None = None,
) -> tuple[AgentState, list[ToolResult]]:
    before = clone_state(state)
    tool_call_dicts = cast(list[dict[str, Any]], list(tool_calls))
    state, results = await executor.aexecute_calls(
        state, tool_call_dicts, config, event_sink=event_sink
    )
    messages = list(state.get("messages", []))
    tool_outputs = list(state.get("tool_outputs", []))
    for result in results:
        messages.append(_to_tool_message(result))
        tool_outputs.append(result)
    state["messages"] = messages
    state["tool_outputs"] = tool_outputs
    if event_sink:
        event_sink(StateUpdateEvent(run_id="", delta=state_delta(before, state)))
    return state, results


def has_tool_calls(message: AIMessage | None) -> bool:
    if message is None:
        return False
    tool_calls = getattr(message, "tool_calls", [])
    return bool(tool_calls)
