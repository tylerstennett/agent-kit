from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from agent_kit.config import InvocationConfig, InvocationRequest
from agent_kit.errors import InvocationValidationError
from agent_kit.state import AgentState, make_initial_state


def normalize_request(
    request: InvocationRequest | str,
    *,
    metadata: dict[str, Any] | None = None,
    system_prompt: str | None = None,
    config: InvocationConfig | None = None,
) -> InvocationRequest:
    if isinstance(request, str):
        normalized = InvocationRequest(input_text=request)
    else:
        normalized = deepcopy(request)
    if metadata:
        normalized.metadata.update(metadata)
    if system_prompt is not None:
        normalized.system_prompt = system_prompt
    if config is not None:
        normalized.config = config
    if not normalized.input_text and not normalized.messages:
        raise InvocationValidationError("Invocation must include input_text or messages.")
    return normalized


def build_messages(request: InvocationRequest) -> list[BaseMessage]:
    messages = [msg for msg in request.messages if isinstance(msg, BaseMessage)]
    if request.input_text:
        messages.append(HumanMessage(content=request.input_text))
    if request.system_prompt is not None:
        system_message = SystemMessage(content=request.system_prompt)
        if messages and isinstance(messages[0], SystemMessage):
            messages[0] = system_message
        else:
            messages = [system_message, *messages]
    return messages


def build_initial_state(
    request: InvocationRequest,
    *,
    default_metadata: dict[str, Any],
    metadata_merge: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]],
) -> AgentState:
    merged_metadata = metadata_merge(default_metadata, request.metadata)
    state = make_initial_state(messages=build_messages(request), metadata=merged_metadata)
    if request.state_patch:
        for key, value in request.state_patch.items():
            if key == "messages" and isinstance(value, list):
                state["messages"] = value
            elif key == "tool_outputs" and isinstance(value, list):
                state["tool_outputs"] = value
            elif key == "metadata" and isinstance(value, dict):
                state["metadata"] = value
            elif key == "routing_hints" and isinstance(value, dict):
                state["routing_hints"] = value
            elif key == "termination" and isinstance(value, dict):
                state["termination"] = value
    return state
