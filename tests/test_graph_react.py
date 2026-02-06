from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage

from agent_kit.config import GraphBuildConfig, InvocationConfig
from agent_kit.graph.react import ReActGraphBuilder
from agent_kit.state import AgentState, make_initial_state
from agent_kit.tools.base import BaseTool
from tests.fakes import EchoTool, FakeModelAdapter, make_response


class CounterTool(BaseTool):
    def execute(self, state: AgentState) -> Any:
        metadata = dict(state.get("metadata", {}))
        seen = int(metadata.get("count", 0))
        metadata["count"] = seen + 1
        state["metadata"] = metadata
        return {"seen": seen}


def test_react_graph_stops_when_no_tool_calls() -> None:
    model = FakeModelAdapter(
        responses=[
            make_response(
                "thinking",
                tool_calls=[{"id": "t1", "name": "echo", "args": {"text": "hello"}}],
                tokens=["thinking"],
            ),
            make_response("done", tokens=["done"]),
        ]
    )
    builder = ReActGraphBuilder()
    compiled = builder.build(model, [EchoTool(name="echo")], GraphBuildConfig(max_steps=5))

    state = make_initial_state(messages=[HumanMessage(content="hi")])
    out = compiled.invoke(state, InvocationConfig())

    assert out["termination"]["reason"] == "completed"
    assert len(out["tool_outputs"]) == 1


def test_react_graph_sets_max_steps_termination() -> None:
    model = FakeModelAdapter(
        responses=[
            make_response(
                "loop",
                tool_calls=[{"id": "t1", "name": "echo", "args": {"text": "x"}}],
            )
        ]
    )
    builder = ReActGraphBuilder()
    compiled = builder.build(model, [EchoTool(name="echo")], GraphBuildConfig(max_steps=1))

    state = make_initial_state(messages=[HumanMessage(content="hi")])
    out = compiled.invoke(state, InvocationConfig())

    assert out["termination"]["reason"] == "max_steps_reached"


def test_react_graph_honors_parallel_default_mode() -> None:
    model = FakeModelAdapter(
        responses=[
            make_response(
                "parallel",
                tool_calls=[
                    {"id": "c1", "name": "counter", "args": {}},
                    {"id": "c2", "name": "counter", "args": {}},
                ],
            ),
            make_response("done"),
        ]
    )
    builder = ReActGraphBuilder()
    compiled = builder.build(
        model,
        [CounterTool(name="counter")],
        GraphBuildConfig(max_steps=5, default_execution_mode="parallel"),
    )

    state = make_initial_state(messages=[HumanMessage(content="hi")], metadata={"count": 0})
    out = compiled.invoke(state, InvocationConfig())

    assert out["termination"]["reason"] == "completed"
    assert len(out["tool_outputs"]) == 2
    assert out["metadata"]["count"] == 1
    assert [result.data["seen"] for result in out["tool_outputs"]] == [0, 0]
