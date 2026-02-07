from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast

from langchain_core.messages import AIMessage, BaseMessage

from agent_kit.config import InvocationConfig, InvocationRequest, ToolResult
from agent_kit.state import AgentState
from agent_kit.tools.base import BaseTool
from agent_kit.utils.state_utils import merge_metadata

if TYPE_CHECKING:
    from agent_kit.agent import Agent

StateBridgeMode = Literal["copy_merge", "shared", "isolated"]
MessageMergeMode = Literal["final_with_transcript", "inline_all", "none"]
FailureMode = Literal["continue", "abort"]


class InvocableAgent(Protocol):
    def invoke(
        self,
        request: InvocationRequest | str,
        *,
        metadata: dict[str, Any] | None = None,
        system_prompt: str | None = None,
        config: InvocationConfig | None = None,
    ) -> AgentState: ...

    async def ainvoke(
        self,
        request: InvocationRequest | str,
        *,
        metadata: dict[str, Any] | None = None,
        system_prompt: str | None = None,
        config: InvocationConfig | None = None,
    ) -> AgentState: ...


MapInputCallback = Callable[[AgentState, dict[str, Any]], InvocationRequest]
MergeOutputCallback = Callable[[AgentState, AgentState], AgentState]
MessageMergeCallback = Callable[[AgentState, AgentState, ToolResult], list[BaseMessage]]


@dataclass(slots=True)
class NestedAgentPolicy:
    state_bridge: StateBridgeMode = "copy_merge"
    message_merge: MessageMergeMode = "final_with_transcript"
    failure_mode: FailureMode = "continue"
    max_depth: int = 3
    map_input: MapInputCallback | None = None
    merge_output: MergeOutputCallback | None = None
    merge_messages: MessageMergeCallback | None = None


DEFAULT_NESTED_POLICY = NestedAgentPolicy()


def _get_depth(state: AgentState) -> int:
    metadata = state.get("metadata", {})
    value = metadata.get("_agent_depth", 0)
    if isinstance(value, int):
        return value
    return 0


def _last_ai_message(messages: list[BaseMessage]) -> AIMessage | None:
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            return message
    return None


class NestedAgentTool(BaseTool):
    def __init__(
        self,
        agent: InvocableAgent,
        *,
        name: str | None = None,
        description: str = "Delegate to a child agent.",
        policy: NestedAgentPolicy | None = None,
    ) -> None:
        super().__init__(name=name or "nested_agent", description=description)
        self._agent = agent
        self.policy = policy or DEFAULT_NESTED_POLICY

    def execute(self, state: AgentState, **kwargs: Any) -> Any:
        request = self._build_request(state, kwargs)
        child_state = self._agent.invoke(request)
        return {"child_state": child_state}

    async def aexecute(self, state: AgentState, **kwargs: Any) -> Any:
        request = self._build_request(state, kwargs)
        child_state = await self._agent.ainvoke(request)
        return {"child_state": child_state}

    def post_execute(self, state: AgentState, result: ToolResult) -> tuple[AgentState, ToolResult]:
        if result.status != "success" or not isinstance(result.data, dict):
            return state, result
        child_state = result.data.get("child_state")
        if not isinstance(child_state, dict):
            return state, result
        child_agent_state = cast(AgentState, child_state)

        merged_state = self._merge_state(state, child_agent_state)
        result.metadata = dict(result.metadata)
        result.metadata["child_termination"] = dict(child_agent_state.get("termination", {}))
        result.metadata["child_transcript"] = list(child_agent_state.get("messages", []))

        if self.policy.message_merge == "inline_all":
            messages = list(merged_state.get("messages", []))
            messages.extend(list(child_agent_state.get("messages", [])))
            merged_state["messages"] = messages
        elif self.policy.message_merge == "final_with_transcript":
            last_ai = _last_ai_message(list(child_agent_state.get("messages", [])))
            if last_ai is not None:
                result.metadata["child_final_message"] = last_ai.content
        return merged_state, result

    def _merge_state(self, parent: AgentState, child: AgentState) -> AgentState:
        if self.policy.merge_output is not None:
            return self.policy.merge_output(parent, child)
        mode = self.policy.state_bridge
        if mode == "isolated":
            return parent

        merged = cast(AgentState, dict(parent))
        parent_metadata = dict(parent.get("metadata", {}))
        child_metadata = dict(child.get("metadata", {}))
        child_metadata.pop("_agent_depth", None)
        merged_metadata = merge_metadata(
            parent_metadata,
            child_metadata,
        )
        if "_agent_depth" in parent_metadata:
            merged_metadata["_agent_depth"] = parent_metadata["_agent_depth"]
        else:
            merged_metadata.pop("_agent_depth", None)
        merged["metadata"] = merged_metadata

        if mode == "shared":
            merged_messages = list(parent.get("messages", []))
            child_messages = list(child.get("messages", []))
            if len(child_messages) > len(merged_messages):
                merged_messages.extend(child_messages[len(merged_messages) :])
            merged["messages"] = merged_messages
            parent_outputs = list(parent.get("tool_outputs", []))
            child_outputs = list(child.get("tool_outputs", []))
            if len(child_outputs) > len(parent_outputs):
                parent_outputs.extend(child_outputs[len(parent_outputs) :])
            merged["tool_outputs"] = parent_outputs
        return merged  # copy_merge and shared both merge metadata by default

    def _build_request(self, state: AgentState, kwargs: dict[str, Any]) -> InvocationRequest:
        parent_depth = _get_depth(state)
        if parent_depth >= self.policy.max_depth:
            raise RuntimeError(f"Nested depth limit reached ({self.policy.max_depth}).")

        if self.policy.map_input is not None:
            request = self.policy.map_input(state, kwargs)
        else:
            request = self._default_request(state, kwargs)

        merged_metadata = merge_metadata(dict(state.get("metadata", {})), dict(request.metadata))
        merged_metadata["_agent_depth"] = parent_depth + 1
        request.metadata = merged_metadata
        return request

    def _default_request(self, state: AgentState, kwargs: dict[str, Any]) -> InvocationRequest:
        mode = self.policy.state_bridge
        if mode == "isolated":
            if "input" in kwargs:
                return InvocationRequest(input_text=str(kwargs["input"]))
            if "message" in kwargs:
                return InvocationRequest(input_text=str(kwargs["message"]))
            return InvocationRequest(input_text="")

        messages = list(state.get("messages", []))
        if "input" in kwargs and kwargs["input"]:
            return InvocationRequest(input_text=str(kwargs["input"]), messages=messages)
        return InvocationRequest(messages=messages)


def agent_as_tool(
    agent: Agent,
    *,
    name: str | None = None,
    description: str = "Delegate to a child agent.",
    policy: NestedAgentPolicy | None = None,
) -> NestedAgentTool:
    return NestedAgentTool(agent, name=name, description=description, policy=policy)
