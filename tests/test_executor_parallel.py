from __future__ import annotations

import time
from typing import Any

from agent_kit.config import InvocationConfig
from agent_kit.state import AgentState, make_initial_state
from agent_kit.tools.base import BaseTool
from agent_kit.tools.executor import ToolExecutor
from agent_kit.tools.registry import ToolRegistry


class ParallelMetadataTool(BaseTool):
    def execute(self, state: AgentState, key: str, value: str, delay: float) -> Any:
        time.sleep(delay)
        metadata = dict(state.get("metadata", {}))
        metadata[key] = value
        state["metadata"] = metadata
        return {"value": value}


def test_parallel_snapshot_merge_is_deterministic() -> None:
    tool = ParallelMetadataTool(name="meta")
    executor = ToolExecutor(ToolRegistry([tool]))
    state = make_initial_state(metadata={"key": "seed"})

    out_state, results = executor.execute_calls(
        state,
        [
            {"id": "1", "name": "meta", "args": {"key": "key", "value": "first", "delay": 0.03}},
            {"id": "2", "name": "meta", "args": {"key": "key", "value": "second", "delay": 0.0}},
        ],
        InvocationConfig(execution_mode="parallel", max_parallel_workers=2),
    )

    assert [result.status for result in results] == ["success", "success"]
    assert out_state["metadata"]["key"] == "second"
