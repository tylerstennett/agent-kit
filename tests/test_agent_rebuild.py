from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from typing import Any

from agent_kit.agent import Agent
from agent_kit.config import GraphBuildConfig, InvocationConfig
from agent_kit.events import RunEndEvent, RunStartEvent
from agent_kit.state import AgentState
from agent_kit.tools.base import BaseTool
from tests.fakes import EchoTool, FakeModelAdapter, make_response


@dataclass
class StubCompiledGraph:
    def invoke(self, state: AgentState, config: InvocationConfig) -> AgentState:
        state["termination"] = {"reason": "completed", "steps": 0}
        return state

    async def ainvoke(self, state: AgentState, config: InvocationConfig) -> AgentState:
        return self.invoke(state, config)

    def stream(self, state: AgentState, config: InvocationConfig) -> Iterator[Any]:
        yield RunStartEvent(run_id="r")
        yield RunEndEvent(run_id="r", termination={"reason": "completed"})

    async def astream(self, state: AgentState, config: InvocationConfig) -> AsyncIterator[Any]:
        for event in self.stream(state, config):
            yield event


class CountingBuilder:
    def __init__(self) -> None:
        self.build_count = 0

    def build(
        self,
        model: Any,
        tools: list[BaseTool],
        config: GraphBuildConfig,
        *,
        before_hooks: list[Any] | None = None,
        after_hooks: list[Any] | None = None,
        middlewares: list[Any] | None = None,
    ) -> StubCompiledGraph:
        self.build_count += 1
        return StubCompiledGraph()


def test_rebuild_marks_graph_dirty_and_lazy_recompiles() -> None:
    model = FakeModelAdapter(responses=[make_response("done")])
    builder = CountingBuilder()
    agent = Agent(model=model, tools=[EchoTool(name="echo")], graph_builder=builder)

    agent.run("first")
    assert builder.build_count == 1

    agent.run("second")
    assert builder.build_count == 1

    agent.add_tool(EchoTool(name="echo2"))
    agent.run("third")
    assert builder.build_count == 2
