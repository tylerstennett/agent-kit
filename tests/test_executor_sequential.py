from __future__ import annotations

from typing import Any

from agent_kit.config import InvocationConfig
from agent_kit.state import AgentState, make_initial_state
from agent_kit.tools.base import BaseTool
from agent_kit.tools.executor import ToolExecutor
from agent_kit.tools.registry import ToolRegistry


class TraceTool(BaseTool):
    def pre_execute(self, state: AgentState):
        log = list(state.get("metadata", {}).get("order", []))
        log.append("tool_pre")
        metadata = dict(state.get("metadata", {}))
        metadata["order"] = log
        state["metadata"] = metadata
        return super().pre_execute(state)

    def execute(self, state: AgentState, value: int) -> Any:
        log = list(state.get("metadata", {}).get("order", []))
        log.append("tool_execute")
        metadata = dict(state.get("metadata", {}))
        metadata["order"] = log
        state["metadata"] = metadata
        return {"value": value}

    def post_execute(self, state: AgentState, result):
        log = list(state.get("metadata", {}).get("order", []))
        log.append("tool_post")
        metadata = dict(state.get("metadata", {}))
        metadata["order"] = log
        state["metadata"] = metadata
        return state, result


def test_sequential_executor_respects_hook_order() -> None:
    order: list[str] = []

    def before_hook(state: AgentState, tool_name: str, tool_args: dict[str, Any]) -> AgentState:
        order.append("before")
        return state

    def after_hook(state: AgentState, tool_name: str, result):
        order.append("after")
        return state, result

    executor = ToolExecutor(
        ToolRegistry([TraceTool(name="trace")]),
        before_hooks=[before_hook],
        after_hooks=[after_hook],
    )
    state = make_initial_state(metadata={"order": []})

    out_state, results = executor.execute_calls(
        state,
        [{"id": "1", "name": "trace", "args": {"value": 5}}],
        InvocationConfig(execution_mode="sequential"),
    )

    assert results[0].status == "success"
    assert order == ["before", "after"]
    assert out_state["metadata"]["order"] == ["tool_pre", "tool_execute", "tool_post"]
