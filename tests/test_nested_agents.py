from __future__ import annotations

from agent_kit.agent import Agent
from agent_kit.config import InvocationConfig
from agent_kit.nested import NestedAgentPolicy
from agent_kit.state import make_initial_state
from agent_kit.tools.executor import ToolExecutor
from agent_kit.tools.registry import ToolRegistry
from tests.fakes import FakeModelAdapter, make_response


def build_child_agent() -> Agent:
    model = FakeModelAdapter(responses=[make_response("child done")])
    return Agent(model=model, tools=[])


def test_nested_agent_tool_merges_child_metadata() -> None:
    child = build_child_agent()
    nested_tool = child.as_tool(name="delegate")

    parent_model = FakeModelAdapter(
        responses=[
            make_response(
                "delegate",
                tool_calls=[{"id": "n1", "name": "delegate", "args": {"input": "hello"}}],
            ),
            make_response("parent done"),
        ]
    )
    parent = Agent(model=parent_model, tools=[nested_tool])

    out = parent.run("go")

    assert out["termination"]["reason"] == "completed"
    assert len(out["tool_outputs"]) == 1
    assert "child_transcript" in out["tool_outputs"][0].metadata


def test_nested_agent_depth_limit_enforced() -> None:
    child = build_child_agent()
    tool = child.as_tool(name="delegate", policy=NestedAgentPolicy(max_depth=0))
    executor = ToolExecutor(ToolRegistry([tool]))

    state = make_initial_state(metadata={"_agent_depth": 0})
    _, results = executor.execute_calls(
        state,
        [{"id": "d1", "name": "delegate", "args": {"input": "x"}}],
        InvocationConfig(),
    )

    assert results[0].status == "failed"
    assert results[0].error is not None
