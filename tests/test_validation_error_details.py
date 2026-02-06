from __future__ import annotations

from typing import Any

from agent_kit.config import InvocationConfig
from agent_kit.state import make_initial_state
from agent_kit.tools.base import BaseTool
from agent_kit.tools.executor import ToolExecutor
from agent_kit.tools.registry import ToolRegistry


class AddTool(BaseTool):
    def execute(self, state: Any, a: int, b: int) -> Any:
        return a + b


def test_validation_error_details_for_unexpected_arguments() -> None:
    tool = AddTool(name="add")
    executor = ToolExecutor(ToolRegistry([tool]))
    state = make_initial_state()

    _, results = executor.execute_calls(
        state,
        [{"id": "c1", "name": "add", "args": {"a": 1, "b": 2, "extra": "x"}}],
        InvocationConfig(),
    )

    error = results[0].error
    assert error is not None
    assert error.code == "validation_error"
    assert error.details["kind"] == "unexpected_args"
    assert error.details["unexpected"] == ["extra"]


def test_validation_error_details_for_missing_arguments() -> None:
    tool = AddTool(name="add")
    executor = ToolExecutor(ToolRegistry([tool]))
    state = make_initial_state()

    _, results = executor.execute_calls(
        state,
        [{"id": "c1", "name": "add", "args": {"a": 1}}],
        InvocationConfig(),
    )

    error = results[0].error
    assert error is not None
    assert error.code == "validation_error"
    assert error.details["kind"] == "missing_args"
    assert error.details["missing"] == ["b"]


def test_validation_error_details_for_invalid_type() -> None:
    tool = AddTool(name="add")
    executor = ToolExecutor(ToolRegistry([tool]))
    state = make_initial_state()

    _, results = executor.execute_calls(
        state,
        [{"id": "c1", "name": "add", "args": {"a": "bad", "b": 2}}],
        InvocationConfig(),
    )

    error = results[0].error
    assert error is not None
    assert error.code == "validation_error"
    assert error.details["kind"] == "invalid_type"
    assert error.details["arg_name"] == "a"
    assert error.details["expected"] == "int"
    assert error.details["actual_type"] == "str"
