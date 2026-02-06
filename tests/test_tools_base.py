from __future__ import annotations

from typing import Any

from agent_kit.config import InvocationConfig, ToolDecision
from agent_kit.state import make_initial_state
from agent_kit.tools.base import BaseTool
from agent_kit.tools.executor import ToolExecutor
from agent_kit.tools.registry import ToolRegistry


class AddTool(BaseTool):
    def execute(self, state: Any, a: int, b: int) -> Any:
        return a + b


class SkipTool(BaseTool):
    def pre_execute(self, state: Any) -> ToolDecision:
        return ToolDecision(action="skip", reason="disabled")

    def execute(self, state: Any, **kwargs: Any) -> Any:
        raise AssertionError("execute should not run")


def test_base_tool_validation_failure_becomes_failed_result() -> None:
    tool = AddTool(name="add")
    executor = ToolExecutor(ToolRegistry([tool]))
    state = make_initial_state()

    _, result = executor.execute_calls(
        state,
        [{"id": "c1", "name": "add", "args": {"a": "bad", "b": 2}}],
        InvocationConfig(),
    )

    assert result[0].status == "failed"
    assert result[0].error is not None
    assert result[0].error.code == "validation_error"


def test_pre_execute_skip_short_circuits_tool() -> None:
    tool = SkipTool(name="skip")
    executor = ToolExecutor(ToolRegistry([tool]))
    state = make_initial_state()

    _, result = executor.execute_calls(
        state,
        [{"id": "c1", "name": "skip", "args": {}}],
        InvocationConfig(),
    )

    assert result[0].status == "skipped"
