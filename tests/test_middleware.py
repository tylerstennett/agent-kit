from __future__ import annotations

import asyncio
from typing import Any

import pytest

from agent_kit.config import InvocationConfig, ToolError, ToolResult
from agent_kit.middleware import MetricsMiddleware, RetryMiddleware, TimeoutMiddleware
from agent_kit.state import AgentState, make_initial_state
from agent_kit.tools.base import BaseTool
from agent_kit.tools.executor import ToolExecutor
from agent_kit.tools.registry import ToolRegistry


class FlakyTool(BaseTool):
    def __init__(self) -> None:
        super().__init__(name="flaky")
        self.calls = 0

    def execute(self, state: AgentState) -> Any:
        self.calls += 1
        if self.calls == 1:
            return ToolResult(
                tool_name=self.name,
                call_id="",
                status="failed",
                error=ToolError(code="transient", message="try again", retriable=True),
            )
        return {"ok": True}


class AsyncSlowTool(BaseTool):
    def execute(self, state: AgentState, delay: float) -> Any:
        return {"delay": delay}

    async def aexecute(self, state: AgentState, delay: float) -> Any:
        await asyncio.sleep(delay)
        return {"delay": delay}


def test_retry_and_metrics_middleware() -> None:
    tool = FlakyTool()
    executor = ToolExecutor(
        ToolRegistry([tool]),
        middlewares=[RetryMiddleware(max_retries=1), MetricsMiddleware()],
    )

    state = make_initial_state()
    out_state, results = executor.execute_calls(
        state,
        [{"id": "x", "name": "flaky", "args": {}}],
        InvocationConfig(),
    )

    assert results[0].status == "success"
    assert tool.calls == 2
    metrics = out_state["metadata"]["metrics"]["flaky"]
    assert metrics["count"] == 2


@pytest.mark.asyncio
async def test_timeout_middleware_on_async_tool() -> None:
    tool = AsyncSlowTool(name="slow")
    executor = ToolExecutor(
        ToolRegistry([tool]), middlewares=[TimeoutMiddleware(timeout_seconds=0.01)]
    )
    state = make_initial_state()

    _, results = await executor.aexecute_calls(
        state,
        [{"id": "s1", "name": "slow", "args": {"delay": 0.05}}],
        InvocationConfig(),
    )

    assert results[0].status == "failed"
    assert results[0].error is not None
    assert results[0].error.code == "timeout"
