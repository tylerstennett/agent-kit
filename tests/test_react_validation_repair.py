from __future__ import annotations

from typing import Any

from agent_kit import Agent
from agent_kit.config import AgentConfig, GraphBuildConfig
from agent_kit.state import AgentState
from agent_kit.tools.base import BaseTool
from tests.fakes import FakeModelAdapter, make_response


class IntTool(BaseTool):
    def execute(self, state: AgentState, value: int) -> Any:
        return {"value": value}


def _agent_with_responses(responses: list[Any]) -> Agent:
    model = FakeModelAdapter(responses=responses)
    graph_config = GraphBuildConfig(max_steps=10, default_validation_repair_turns=1)
    config = AgentConfig(graph_config=graph_config)
    return Agent(model=model, tools=[IntTool(name="intval")], config=config)


def test_validation_repair_allows_one_failing_turn_then_recovers() -> None:
    agent = _agent_with_responses(
        [
            make_response(
                "first",
                tool_calls=[{"id": "c1", "name": "intval", "args": {"value": "bad"}}],
            ),
            make_response(
                "retry",
                tool_calls=[{"id": "c2", "name": "intval", "args": {"value": 7}}],
            ),
            make_response("done"),
        ]
    )

    state = agent.run("go")

    assert state["termination"]["reason"] == "completed"
    assert len(state["tool_outputs"]) == 2
    assert state["tool_outputs"][0].error is not None
    assert state["tool_outputs"][0].error.code == "validation_error"
    assert state["tool_outputs"][1].status == "success"


def test_validation_repair_exhausts_on_two_consecutive_failing_turns() -> None:
    agent = _agent_with_responses(
        [
            make_response(
                "first",
                tool_calls=[{"id": "c1", "name": "intval", "args": {"value": "bad"}}],
            ),
            make_response(
                "second",
                tool_calls=[{"id": "c2", "name": "intval", "args": {"value": "bad-again"}}],
            ),
            make_response("unused"),
        ]
    )

    state = agent.run("go")

    assert state["termination"]["reason"] == "validation_repair_exhausted"
    assert len(state["tool_outputs"]) == 2
    assert state["tool_outputs"][0].error is not None
    assert state["tool_outputs"][0].error.code == "validation_error"
    assert state["tool_outputs"][1].error is not None
    assert state["tool_outputs"][1].error.code == "validation_error"


def test_validation_repair_counter_resets_after_successful_turn() -> None:
    agent = _agent_with_responses(
        [
            make_response(
                "fail-1",
                tool_calls=[{"id": "c1", "name": "intval", "args": {"value": "bad"}}],
            ),
            make_response(
                "ok-1",
                tool_calls=[{"id": "c2", "name": "intval", "args": {"value": 2}}],
            ),
            make_response(
                "fail-2",
                tool_calls=[{"id": "c3", "name": "intval", "args": {"value": "bad"}}],
            ),
            make_response(
                "ok-2",
                tool_calls=[{"id": "c4", "name": "intval", "args": {"value": 4}}],
            ),
            make_response("done"),
        ]
    )

    state = agent.run("go")

    assert state["termination"]["reason"] == "completed"
    assert len(state["tool_outputs"]) == 4
    assert state["tool_outputs"][0].error is not None
    assert state["tool_outputs"][0].error.code == "validation_error"
    assert state["tool_outputs"][1].status == "success"
    assert state["tool_outputs"][2].error is not None
    assert state["tool_outputs"][2].error.code == "validation_error"
    assert state["tool_outputs"][3].status == "success"
