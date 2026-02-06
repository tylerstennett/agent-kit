from __future__ import annotations

from agent_kit.agent import Agent
from agent_kit.config import InvocationConfig
from agent_kit.events import LLMTokenEvent, RunEndEvent, RunStartEvent, StateUpdateEvent
from tests.fakes import EchoTool, FakeModelAdapter, make_response


def test_stream_emits_core_event_types_and_state_snapshot_option() -> None:
    model = FakeModelAdapter(
        responses=[
            make_response(
                "use",
                tool_calls=[{"id": "c1", "name": "echo", "args": {"text": "hello"}}],
                tokens=["u", "se"],
            ),
            make_response("done", tokens=["done"]),
        ]
    )
    agent = Agent(model=model, tools=[EchoTool(name="echo")])

    events = list(agent.stream("hi", config=InvocationConfig(include_state_snapshots=True)))

    assert isinstance(events[0], RunStartEvent)
    assert isinstance(events[-1], RunEndEvent)
    assert any(isinstance(event, LLMTokenEvent) for event in events)
    state_events = [event for event in events if isinstance(event, StateUpdateEvent)]
    assert state_events
    assert all(event.snapshot is not None for event in state_events)
    assert all(event.delta for event in state_events)
    assert any("messages" in event.delta for event in state_events)
    assert any("tool_outputs" in event.delta for event in state_events)
