from __future__ import annotations

from langchain_core.messages import HumanMessage

from agent_kit.config import GraphBuildConfig, InvocationConfig
from agent_kit.graph.react import ReActGraphBuilder
from agent_kit.state import make_initial_state
from tests.fakes import EchoTool, FakeModelAdapter, make_response


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
