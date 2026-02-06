from __future__ import annotations

from typing import Any

from agent_kit.config import InvocationConfig, ToolDecision, ToolResult
from agent_kit.state import AgentState, make_initial_state
from agent_kit.tools import ToolExecutor, ToolRegistry, tool


def test_decorator_tool_infers_signature_and_hooks() -> None:
    order: list[str] = []

    def pre(state: AgentState) -> ToolDecision:
        order.append("pre")
        return ToolDecision(action="continue")

    def post(state: AgentState, result: ToolResult) -> tuple[AgentState, ToolResult]:
        order.append("post")
        result.metadata["tag"] = "ok"
        return state, result

    @tool(name="concat", pre_execute=pre, post_execute=post)
    def concat(state: AgentState, left: str, right: str) -> Any:
        order.append("execute")
        return left + right

    executor = ToolExecutor(ToolRegistry([concat]))
    state = make_initial_state()
    _, results = executor.execute_calls(
        state,
        [{"id": "call1", "name": "concat", "args": {"left": "a", "right": "b"}}],
        InvocationConfig(),
    )

    assert order == ["pre", "execute", "post"]
    assert results[0].data == "ab"
    assert results[0].metadata["tag"] == "ok"
