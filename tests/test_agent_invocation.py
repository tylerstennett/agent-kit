from __future__ import annotations

import asyncio

import pytest

from agent_kit.agent import Agent
from agent_kit.errors import SyncInAsyncContextError
from agent_kit.events import RunEndEvent, RunStartEvent, ToolEndEvent, ToolStartEvent
from tests.fakes import EchoTool, FakeModelAdapter, make_response


def build_agent() -> Agent:
    model = FakeModelAdapter(
        responses=[
            make_response(
                "use tool",
                tool_calls=[{"id": "c1", "name": "echo", "args": {"text": "hello"}}],
                tokens=["use", "tool"],
            ),
            make_response("done", tokens=["done"]),
        ]
    )
    return Agent(model=model, tools=[EchoTool(name="echo")])


def test_run_returns_state_and_merges_metadata() -> None:
    agent = build_agent()
    out = agent.run("hi", metadata={"session": "s1"})

    assert out["metadata"]["session"] == "s1"
    assert out["termination"]["reason"] == "completed"


@pytest.mark.asyncio
async def test_arun_returns_state() -> None:
    agent = build_agent()
    out = await agent.arun("hi")

    assert out["termination"]["reason"] == "completed"


def test_stream_emits_lifecycle_events() -> None:
    agent = build_agent()
    events = list(agent.stream("hi"))

    assert any(isinstance(event, RunStartEvent) for event in events)
    assert any(isinstance(event, ToolStartEvent) for event in events)
    assert any(isinstance(event, ToolEndEvent) for event in events)
    assert any(isinstance(event, RunEndEvent) for event in events)


def test_run_inside_event_loop_raises_guidance() -> None:
    agent = build_agent()

    async def call_run() -> None:
        with pytest.raises(SyncInAsyncContextError):
            agent.run("hi")

    asyncio.run(call_run())
