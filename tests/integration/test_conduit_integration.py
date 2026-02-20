"""Comprehensive integration tests for Agent Kit with Conduit (OpenRouter).

These tests exercise the full stack: Agent → ConduitModelAdapter → SyncConduit/Conduit
→ OpenRouter API → real LLM.  They require a valid ``OPENROUTER_API_KEY`` in the
``.env`` file at the project root.

Run with::

    pytest tests/integration/ -v
"""

from __future__ import annotations

from typing import Any

import pytest
from conduit import Conduit, SyncConduit
from conduit.config import OpenRouterConfig
from conduit.exceptions import ProviderError
from conduit.models.messages import RequestContext
from dotenv import load_dotenv
from langchain_core.messages import AIMessage

from agent_kit import (
    Agent,
    AgentConfig,
    GraphBuildConfig,
    InvocationConfig,
    InvocationRequest,
    MetricsMiddleware,
    RunEndEvent,
    RunStartEvent,
    StateUpdateEvent,
    ToolEndEvent,
    ToolStartEvent,
    logging_middleware,
    tool,
)
from agent_kit.config import ToolResult
from agent_kit.llm.conduit_adapter import (
    CONDUIT_CONTEXT_METADATA_KEY,
    CONDUIT_RUNTIME_OVERRIDES_KEY,
)
from agent_kit.state import AgentState

from .conftest import AsyncConduitCallRecorder, ConduitCallRecorder, requires_api_key

load_dotenv()

# ---------------------------------------------------------------------------
# Module-level markers: every test in this file is an integration test and
# requires a valid API key.
# ---------------------------------------------------------------------------

pytestmark = [pytest.mark.integration, requires_api_key]


# ---------------------------------------------------------------------------
# Reusable tools
# ---------------------------------------------------------------------------


@tool(name="add")
def add_tool(state: AgentState, a: int, b: int) -> int:
    """Add two integers and return the sum."""
    return a + b


@tool(name="multiply")
def multiply_tool(state: AgentState, a: int, b: int) -> int:
    """Multiply two integers and return the product."""
    return a * b


@tool(name="reverse_string")
def reverse_string_tool(state: AgentState, text: str) -> str:
    """Reverse a string and return it."""
    return text[::-1]


@tool(name="get_weather")
def get_weather_tool(state: AgentState, city: str) -> str:
    """Look up the current weather for a given city."""
    weather_data: dict[str, str] = {
        "new york": "Sunny, 72F",
        "london": "Cloudy, 55F",
        "tokyo": "Rainy, 65F",
        "paris": "Clear, 68F",
    }
    return weather_data.get(city.lower(), f"Weather data unavailable for {city}")


@tool(name="concat")
def concat_tool(state: AgentState, left: str, right: str) -> str:
    """Concatenate two strings."""
    return left + right


@tool(name="always_fails")
def failing_tool(state: AgentState) -> str:
    """This tool always raises an error. Do not call it."""
    raise RuntimeError("Intentional test failure")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default_agent_config(max_steps: int = 5) -> AgentConfig:
    return AgentConfig(graph_config=GraphBuildConfig(max_steps=max_steps))


# ===========================================================================
# 1. Basic Chat Completion
# ===========================================================================


class TestBasicCompletion:
    """Simple prompt-in / text-out tests — no tools involved."""

    def test_sync_conduit_simple_completion(self, sync_conduit: SyncConduit) -> None:
        """Agent.run with SyncConduit answers a factual question."""
        agent = Agent(model=sync_conduit, tools=[], config=_default_agent_config())
        state = agent.run("What is the capital of Japan? Answer in one word.")

        assert state["termination"]["reason"] == "completed"
        messages = state["messages"]
        last_ai = [m for m in messages if isinstance(m, AIMessage)]
        assert len(last_ai) >= 1
        assert "tokyo" in last_ai[-1].content.lower()

    async def test_async_conduit_simple_completion(self, async_conduit: Conduit) -> None:
        """Agent.arun with async Conduit answers a factual question."""
        agent = Agent(model=async_conduit, tools=[], config=_default_agent_config())
        state = await agent.arun("What is the capital of France? Answer in one word.")

        assert state["termination"]["reason"] == "completed"
        messages = state["messages"]
        last_ai = [m for m in messages if isinstance(m, AIMessage)]
        assert len(last_ai) >= 1
        assert "paris" in last_ai[-1].content.lower()

    def test_agent_run_returns_messages_and_termination(self, sync_conduit: SyncConduit) -> None:
        """State contains messages list and termination dict after a simple run."""
        agent = Agent(model=sync_conduit, tools=[], config=_default_agent_config())
        state = agent.run("Say hello.")

        assert "messages" in state
        assert len(state["messages"]) >= 2  # HumanMessage + AIMessage
        assert state["termination"]["reason"] == "completed"
        assert state["termination"]["steps"] >= 1


# ===========================================================================
# 2. System Prompts
# ===========================================================================


class TestSystemPrompts:
    """Verify that system prompts are forwarded and influence responses."""

    def test_system_prompt_steers_response(self, sync_conduit: SyncConduit) -> None:
        """A system prompt telling the model to respond in French should
        produce a response containing French text."""
        agent = Agent(model=sync_conduit, tools=[], config=_default_agent_config())
        state = agent.run(
            "What is 1+1?",
            system_prompt="You must always respond in French. No English allowed.",
        )

        assert state["termination"]["reason"] == "completed"
        last_ai = [m for m in state["messages"] if isinstance(m, AIMessage)][-1]
        content = last_ai.content.lower()
        # The model should respond with French words — "deux", "égale", "est", etc.
        assert any(
            word in content
            for word in ["deux", "réponse", "est", "résultat", "la", "égale", "font"]
        ), f"Expected French response, got: {last_ai.content}"

    def test_system_prompt_persona(self, sync_conduit: SyncConduit) -> None:
        """System prompt creates a specific persona that shows in the response."""
        agent = Agent(model=sync_conduit, tools=[], config=_default_agent_config())
        state = agent.run(
            "Who are you?",
            system_prompt=(
                "You are CaptainBot, a pirate AI. You always talk like a pirate. "
                "Include 'Arrr' in every response."
            ),
        )

        assert state["termination"]["reason"] == "completed"
        last_ai = [m for m in state["messages"] if isinstance(m, AIMessage)][-1]
        assert "arr" in last_ai.content.lower()


# ===========================================================================
# 3. Single Tool Calling
# ===========================================================================


class TestSingleToolCalling:
    """Verify the model calls a single tool and the result flows correctly."""

    def test_single_tool_call_sync(self, sync_conduit: SyncConduit) -> None:
        """Agent calls the add tool and produces a correct final answer."""
        agent = Agent(
            model=sync_conduit,
            tools=[add_tool],
            config=_default_agent_config(),
        )
        state = agent.run("Use the add tool to compute 15 + 27. Report the exact result.")

        assert state["termination"]["reason"] == "completed"
        assert len(state["tool_outputs"]) >= 1
        # The add tool should have been called and returned 42
        add_results = [r for r in state["tool_outputs"] if r.tool_name == "add"]
        assert len(add_results) >= 1
        assert add_results[0].status == "success"
        assert add_results[0].data == 42

    async def test_single_tool_call_async(self, async_conduit: Conduit) -> None:
        """Async agent calls the add tool correctly."""
        agent = Agent(
            model=async_conduit,
            tools=[add_tool],
            config=_default_agent_config(),
        )
        state = await agent.arun("Use the add tool to compute 100 + 200. Report the result.")

        assert state["termination"]["reason"] == "completed"
        add_results = [r for r in state["tool_outputs"] if r.tool_name == "add"]
        assert len(add_results) >= 1
        assert add_results[0].status == "success"
        assert add_results[0].data == 300

    def test_tool_outputs_recorded_in_state(self, sync_conduit: SyncConduit) -> None:
        """tool_outputs in the final state contain the tool result metadata."""
        agent = Agent(
            model=sync_conduit,
            tools=[add_tool],
            config=_default_agent_config(),
        )
        state = agent.run("Use the add tool to compute 7 + 8.")

        assert len(state["tool_outputs"]) >= 1
        result = state["tool_outputs"][0]
        assert isinstance(result, ToolResult)
        assert result.tool_name == "add"
        assert result.call_id  # non-empty string
        assert result.status == "success"
        assert result.data == 15
        assert result.started_at is not None
        assert result.ended_at is not None
        assert result.duration_ms is not None
        assert result.duration_ms >= 0

    def test_tool_result_incorporated_in_final_response(self, sync_conduit: SyncConduit) -> None:
        """The model's final text response references the computed tool result."""
        agent = Agent(
            model=sync_conduit,
            tools=[add_tool],
            config=_default_agent_config(),
        )
        state = agent.run(
            "Use the add tool to compute 50 + 70. "
            "Then tell me the result in your final response."
        )

        assert state["termination"]["reason"] == "completed"
        last_ai = [m for m in state["messages"] if isinstance(m, AIMessage)][-1]
        assert "120" in last_ai.content


# ===========================================================================
# 4. Multiple Tools
# ===========================================================================


class TestMultipleTools:
    """Agent selects the correct tool from several, chains calls, etc."""

    def test_agent_selects_correct_tool(self, sync_conduit: SyncConduit) -> None:
        """Given both add and reverse_string, the model picks reverse_string
        for a string reversal task."""
        agent = Agent(
            model=sync_conduit,
            tools=[add_tool, reverse_string_tool],
            config=_default_agent_config(),
        )
        state = agent.run(
            "Use the reverse_string tool to reverse the text 'hello'. " "Report the result."
        )

        assert state["termination"]["reason"] == "completed"
        reverse_results = [r for r in state["tool_outputs"] if r.tool_name == "reverse_string"]
        assert len(reverse_results) >= 1
        assert reverse_results[0].status == "success"
        assert reverse_results[0].data == "olleh"

    def test_tool_with_multiple_parameter_types(self, sync_conduit: SyncConduit) -> None:
        """The get_weather tool receives a string argument and returns data."""
        agent = Agent(
            model=sync_conduit,
            tools=[get_weather_tool],
            config=_default_agent_config(),
        )
        state = agent.run("Use the get_weather tool to check the weather in Tokyo.")

        assert state["termination"]["reason"] == "completed"
        weather_results = [r for r in state["tool_outputs"] if r.tool_name == "get_weather"]
        assert len(weather_results) >= 1
        assert weather_results[0].status == "success"
        assert "65" in str(weather_results[0].data)  # "Rainy, 65F"

    def test_multi_step_react_loop(self, sync_conduit: SyncConduit) -> None:
        """Agent performs two sequential tool calls in separate ReAct steps.

        Step 1: add(15, 27) → 42
        Step 2: multiply(42, 3) → 126
        """
        agent = Agent(
            model=sync_conduit,
            tools=[add_tool, multiply_tool],
            config=_default_agent_config(max_steps=8),
        )
        state = agent.run(
            "First, use the add tool to compute 15 + 27. "
            "Then, use the multiply tool to multiply that result by 3. "
            "Report both intermediate and final results."
        )

        assert state["termination"]["reason"] == "completed"
        add_results = [r for r in state["tool_outputs"] if r.tool_name == "add"]
        mul_results = [r for r in state["tool_outputs"] if r.tool_name == "multiply"]
        assert len(add_results) >= 1
        assert len(mul_results) >= 1
        assert add_results[0].data == 42
        assert mul_results[0].data == 126

    def test_multiple_tools_all_available(self, sync_conduit: SyncConduit) -> None:
        """All four tools are registered; the model uses the correct ones."""
        agent = Agent(
            model=sync_conduit,
            tools=[add_tool, multiply_tool, reverse_string_tool, get_weather_tool],
            config=_default_agent_config(),
        )
        state = agent.run("Use the multiply tool to compute 6 * 9. Report the result.")

        assert state["termination"]["reason"] == "completed"
        mul_results = [r for r in state["tool_outputs"] if r.tool_name == "multiply"]
        assert len(mul_results) >= 1
        assert mul_results[0].data == 54


# ===========================================================================
# 5. Configuration
# ===========================================================================


class TestConfiguration:
    """InvocationConfig and AgentConfig are respected end-to-end."""

    def test_max_steps_limits_execution(self, sync_conduit: SyncConduit) -> None:
        """With max_steps=1, the agent terminates after one step even if
        the model wants to call more tools."""
        agent = Agent(
            model=sync_conduit,
            tools=[add_tool, multiply_tool],
            config=_default_agent_config(max_steps=1),
        )
        state = agent.run(
            "Use the add tool to compute 1 + 2. Then use multiply to multiply "
            "the result by 10. You MUST call add first."
        )

        # With max_steps=1 the loop runs once: reason → act → done.
        # The model gets one reasoning step. It may or may not call a tool.
        # Either way, termination is "max_steps_reached" or "completed".
        reason = state["termination"]["reason"]
        assert reason in ("max_steps_reached", "completed")
        assert state["termination"]["steps"] <= 1

    def test_config_forwarding_thread_and_tags(self, sync_conduit: SyncConduit) -> None:
        """thread_id and tags from InvocationConfig are forwarded to
        Conduit's RequestContext."""
        recorder = ConduitCallRecorder(sync_conduit)
        agent = Agent(model=sync_conduit, tools=[], config=_default_agent_config())

        state = agent.run(
            "Say hello.",
            config=InvocationConfig(
                thread_id="test-thread-42",
                tags=["integration", "test"],
            ),
        )

        assert state["termination"]["reason"] == "completed"
        assert len(recorder.calls) >= 1
        context = recorder.calls[0]["context"]
        assert isinstance(context, RequestContext)
        assert context.thread_id == "test-thread-42"
        assert context.tags == ["integration", "test"]

    def test_config_overrides_forwarded(self, sync_conduit: SyncConduit) -> None:
        """Configurable dict entries (other than Conduit-specific keys) are
        forwarded as config_overrides."""
        recorder = ConduitCallRecorder(sync_conduit)
        agent = Agent(model=sync_conduit, tools=[], config=_default_agent_config())

        agent.run(
            "Say hi.",
            config=InvocationConfig(configurable={"temperature": 0.1}),
        )

        assert len(recorder.calls) >= 1
        assert recorder.calls[0]["config_overrides"] == {"temperature": 0.1}

    def test_auto_tool_binding_sends_schemas(self, sync_conduit: SyncConduit) -> None:
        """In auto mode, tool schemas are sent to the API via the tools parameter."""
        recorder = ConduitCallRecorder(sync_conduit)
        agent = Agent(
            model=sync_conduit,
            tools=[add_tool],
            config=AgentConfig(
                model_tool_binding_mode="auto",
                graph_config=GraphBuildConfig(max_steps=3),
            ),
        )

        agent.run("What is 2 + 3? Use the add tool.")

        assert len(recorder.calls) >= 1
        sent_tools = recorder.calls[0]["tools"]
        assert sent_tools is not None
        tool_names = [t.name for t in sent_tools]
        assert "add" in tool_names
        assert recorder.calls[0]["tool_choice"] == "auto"
        assert agent.last_tool_schema_signature is not None

    def test_manual_tool_binding_skips_schemas(self, sync_conduit: SyncConduit) -> None:
        """In manual mode, no tool schemas are sent to the API."""
        recorder = ConduitCallRecorder(sync_conduit)
        agent = Agent(
            model=sync_conduit,
            tools=[add_tool],
            config=AgentConfig(
                model_tool_binding_mode="manual",
                graph_config=GraphBuildConfig(max_steps=3),
            ),
        )

        state = agent.run("Hello.")

        assert state["termination"]["reason"] == "completed"
        assert len(recorder.calls) >= 1
        assert recorder.calls[0]["tools"] is None
        assert recorder.calls[0]["tool_choice"] is None
        assert agent.last_tool_schema_signature is None

    def test_invocation_request_with_metadata(self, sync_conduit: SyncConduit) -> None:
        """InvocationRequest metadata is merged into the agent state."""
        agent = Agent(model=sync_conduit, tools=[], config=_default_agent_config())

        request = InvocationRequest(
            input_text="Say hi.",
            metadata={"session_id": "sess-123", "user": "testbot"},
            config=InvocationConfig(),
        )
        state = agent.invoke(request)

        assert state["termination"]["reason"] == "completed"
        assert state["metadata"]["session_id"] == "sess-123"
        assert state["metadata"]["user"] == "testbot"


# ===========================================================================
# 6. Streaming
# ===========================================================================


class TestStreaming:
    """Agent streaming produces correct lifecycle events."""

    def test_sync_stream_lifecycle_events(self, sync_conduit: SyncConduit) -> None:
        """Sync streaming yields at least RunStartEvent and RunEndEvent."""
        agent = Agent(model=sync_conduit, tools=[], config=_default_agent_config())
        events = list(agent.stream("Say hello."))

        event_types = [type(e) for e in events]
        assert RunStartEvent in event_types
        assert RunEndEvent in event_types
        # RunStartEvent should be first, RunEndEvent should be last
        assert isinstance(events[0], RunStartEvent)
        assert isinstance(events[-1], RunEndEvent)

    async def test_async_stream_lifecycle_events(self, async_conduit: Conduit) -> None:
        """Async streaming yields at least RunStartEvent and RunEndEvent."""
        agent = Agent(model=async_conduit, tools=[], config=_default_agent_config())
        events = [event async for event in agent.astream("Say hello.")]

        event_types = [type(e) for e in events]
        assert RunStartEvent in event_types
        assert RunEndEvent in event_types
        assert isinstance(events[0], RunStartEvent)
        assert isinstance(events[-1], RunEndEvent)

    def test_stream_with_tool_events(self, sync_conduit: SyncConduit) -> None:
        """Stream includes ToolStartEvent and ToolEndEvent when a tool is called."""
        agent = Agent(
            model=sync_conduit,
            tools=[add_tool],
            config=_default_agent_config(),
        )
        events = list(agent.stream("Use the add tool to compute 3 + 4. Report the result."))

        event_types = [type(e) for e in events]
        assert ToolStartEvent in event_types
        assert ToolEndEvent in event_types

        tool_starts = [e for e in events if isinstance(e, ToolStartEvent)]
        assert any(e.tool_name == "add" for e in tool_starts)

        tool_ends = [e for e in events if isinstance(e, ToolEndEvent)]
        assert any(e.result is not None and e.result.status == "success" for e in tool_ends)

    def test_stream_state_snapshots(self, sync_conduit: SyncConduit) -> None:
        """When include_state_snapshots=True, StateUpdateEvents carry snapshots."""
        agent = Agent(
            model=sync_conduit,
            tools=[add_tool],
            config=_default_agent_config(),
        )
        events = list(
            agent.stream(
                "Use the add tool to compute 5 + 5.",
                config=InvocationConfig(include_state_snapshots=True),
            )
        )

        state_updates = [e for e in events if isinstance(e, StateUpdateEvent)]
        if state_updates:
            # At least one snapshot should be present
            snapshots_with_data = [e for e in state_updates if e.snapshot is not None]
            assert len(snapshots_with_data) >= 1

    def test_stream_run_end_has_termination(self, sync_conduit: SyncConduit) -> None:
        """The final RunEndEvent contains the termination reason."""
        agent = Agent(model=sync_conduit, tools=[], config=_default_agent_config())
        events = list(agent.stream("Say hello."))

        run_end = [e for e in events if isinstance(e, RunEndEvent)]
        assert len(run_end) == 1
        assert run_end[0].termination["reason"] == "completed"


# ===========================================================================
# 7. Middleware
# ===========================================================================


class TestMiddleware:
    """Built-in middleware works correctly with real Conduit calls."""

    def test_metrics_middleware_records_tool_stats(self, sync_conduit: SyncConduit) -> None:
        """MetricsMiddleware records call count and duration for each tool."""
        agent = Agent(
            model=sync_conduit,
            tools=[add_tool],
            config=_default_agent_config(),
        )
        agent.add_middleware(MetricsMiddleware())

        state = agent.run("Use the add tool to compute 10 + 20.")

        assert state["termination"]["reason"] == "completed"
        metrics = state["metadata"].get("metrics", {})
        assert "add" in metrics
        assert metrics["add"]["count"] >= 1
        assert metrics["add"]["last_duration_ms"] >= 0

    def test_logging_middleware_does_not_interfere(self, sync_conduit: SyncConduit) -> None:
        """logging_middleware does not alter tool results."""
        agent = Agent(
            model=sync_conduit,
            tools=[add_tool],
            config=_default_agent_config(),
        )
        agent.add_middleware(logging_middleware())

        state = agent.run("Use the add tool to compute 1 + 1.")

        assert state["termination"]["reason"] == "completed"
        add_results = [r for r in state["tool_outputs"] if r.tool_name == "add"]
        assert len(add_results) >= 1
        assert add_results[0].status == "success"
        assert add_results[0].data == 2

    def test_multiple_middlewares_stacked(self, sync_conduit: SyncConduit) -> None:
        """Stacking logging + metrics middleware together works without conflict."""
        agent = Agent(
            model=sync_conduit,
            tools=[add_tool],
            config=_default_agent_config(),
        )
        agent.add_middleware(logging_middleware())
        agent.add_middleware(MetricsMiddleware())

        state = agent.run("Use the add tool to compute 99 + 1.")

        assert state["termination"]["reason"] == "completed"
        add_results = [r for r in state["tool_outputs"] if r.tool_name == "add"]
        assert add_results[0].data == 100
        assert state["metadata"]["metrics"]["add"]["count"] >= 1


# ===========================================================================
# 8. Hooks
# ===========================================================================


class TestHooks:
    """Before/after tool hooks fire during real Conduit-backed invocations."""

    def test_before_tool_hook_fires(self, sync_conduit: SyncConduit) -> None:
        """A before-tool hook captures the tool name and args in metadata."""
        captured: list[dict[str, Any]] = []

        def before_hook(state: AgentState, tool_name: str, args: dict[str, Any]) -> AgentState:
            captured.append({"tool_name": tool_name, "args": args})
            return state

        agent = Agent(
            model=sync_conduit,
            tools=[add_tool],
            config=_default_agent_config(),
        )
        agent.add_before_tool_hook(before_hook)

        state = agent.run("Use the add tool to compute 3 + 7.")

        assert state["termination"]["reason"] == "completed"
        assert len(captured) >= 1
        assert captured[0]["tool_name"] == "add"
        assert captured[0]["args"]["a"] == 3
        assert captured[0]["args"]["b"] == 7

    def test_after_tool_hook_fires(self, sync_conduit: SyncConduit) -> None:
        """An after-tool hook sees the successful ToolResult."""
        captured_results: list[ToolResult] = []

        def after_hook(
            state: AgentState, tool_name: str, result: ToolResult
        ) -> tuple[AgentState, ToolResult]:
            captured_results.append(result)
            return state, result

        agent = Agent(
            model=sync_conduit,
            tools=[add_tool],
            config=_default_agent_config(),
        )
        agent.add_after_tool_hook(after_hook)

        state = agent.run("Use the add tool to compute 11 + 22.")

        assert state["termination"]["reason"] == "completed"
        assert len(captured_results) >= 1
        assert captured_results[0].status == "success"
        assert captured_results[0].data == 33

    def test_before_hook_can_mutate_state_metadata(self, sync_conduit: SyncConduit) -> None:
        """Before hooks can add metadata that persists in the final state."""

        def before_hook(state: AgentState, tool_name: str, args: dict[str, Any]) -> AgentState:
            metadata = dict(state.get("metadata", {}))
            metadata["hook_fired"] = True
            metadata["intercepted_tool"] = tool_name
            state["metadata"] = metadata
            return state

        agent = Agent(
            model=sync_conduit,
            tools=[add_tool],
            config=_default_agent_config(),
        )
        agent.add_before_tool_hook(before_hook)

        state = agent.run("Use the add tool to compute 1 + 1.")

        assert state["metadata"]["hook_fired"] is True
        assert state["metadata"]["intercepted_tool"] == "add"


# ===========================================================================
# 9. State Management
# ===========================================================================


class TestStateManagement:
    """State (metadata, messages, termination) is correct after invocation."""

    def test_metadata_propagated_through_invocation(self, sync_conduit: SyncConduit) -> None:
        """Metadata passed via agent.run(metadata=...) appears in the final state."""
        agent = Agent(model=sync_conduit, tools=[], config=_default_agent_config())
        state = agent.run("Hello", metadata={"session": "s1", "user_id": "u42"})

        assert state["metadata"]["session"] == "s1"
        assert state["metadata"]["user_id"] == "u42"

    def test_default_metadata_merged(self, sync_conduit: SyncConduit) -> None:
        """AgentConfig.default_metadata is merged into every invocation."""
        config = AgentConfig(
            default_metadata={"environment": "test", "version": "1.0"},
            graph_config=GraphBuildConfig(max_steps=3),
        )
        agent = Agent(model=sync_conduit, tools=[], config=config)
        state = agent.run("Hello", metadata={"session": "s2"})

        assert state["metadata"]["environment"] == "test"
        assert state["metadata"]["version"] == "1.0"
        assert state["metadata"]["session"] == "s2"

    def test_messages_accumulated_in_state(self, sync_conduit: SyncConduit) -> None:
        """After a simple run, messages contain at least the user message and
        one AI response."""
        agent = Agent(model=sync_conduit, tools=[], config=_default_agent_config())
        state = agent.run("Hello, world!")

        messages = state["messages"]
        assert len(messages) >= 2
        # First message is the HumanMessage
        assert messages[0].content == "Hello, world!"
        # Last message should be an AIMessage
        assert isinstance(messages[-1], AIMessage)
        assert len(messages[-1].content) > 0

    def test_tool_call_messages_in_history(self, sync_conduit: SyncConduit) -> None:
        """When tools are called, the message history includes ToolMessages."""
        agent = Agent(
            model=sync_conduit,
            tools=[add_tool],
            config=_default_agent_config(),
        )
        state = agent.run("Use the add tool to compute 2 + 3.")

        messages = state["messages"]
        # Should contain: Human → AI (with tool_call) → Tool → AI (final)
        assert len(messages) >= 4
        # Find the ToolMessage
        from langchain_core.messages import ToolMessage

        tool_msgs = [m for m in messages if isinstance(m, ToolMessage)]
        assert len(tool_msgs) >= 1


# ===========================================================================
# 10. Nested Agents
# ===========================================================================


class TestNestedAgents:
    """A parent agent delegates to a child agent via a tool."""

    def test_nested_agent_via_custom_tool(self, sync_conduit_factory: Any) -> None:
        """A parent agent uses a tool that internally invokes a child agent."""
        child_conduit = sync_conduit_factory()
        child_agent = Agent(
            model=child_conduit,
            tools=[],
            config=_default_agent_config(max_steps=3),
        )

        @tool(name="ask_expert")
        def ask_expert(state: AgentState, question: str) -> str:
            """Ask an expert agent a question and get their answer."""
            child_state = child_agent.run(question)
            child_messages = child_state.get("messages", [])
            for msg in reversed(child_messages):
                if isinstance(msg, AIMessage):
                    return str(msg.content)
            return "No answer received."

        parent_conduit = sync_conduit_factory()
        parent_agent = Agent(
            model=parent_conduit,
            tools=[ask_expert],
            config=_default_agent_config(max_steps=5),
        )

        state = parent_agent.run(
            "Use the ask_expert tool with the question: 'What is the capital of Germany?'"
        )

        assert state["termination"]["reason"] == "completed"
        assert len(state["tool_outputs"]) >= 1
        expert_results = [r for r in state["tool_outputs"] if r.tool_name == "ask_expert"]
        assert len(expert_results) >= 1
        assert expert_results[0].status == "success"
        # The child agent should have mentioned Berlin
        assert "berlin" in str(expert_results[0].data).lower()

    def test_nested_agent_with_as_tool(self, sync_conduit_factory: Any) -> None:
        """agent.as_tool() creates a NestedAgentTool that works end-to-end."""
        child_conduit = sync_conduit_factory()
        child_agent = Agent(
            model=child_conduit,
            tools=[],
            config=_default_agent_config(max_steps=3),
        )
        nested_tool = child_agent.as_tool(
            name="delegate",
            description=(
                "Delegate a question to a child agent. "
                "Pass the question as the 'input' keyword argument."
            ),
        )

        parent_conduit = sync_conduit_factory()
        parent_agent = Agent(
            model=parent_conduit,
            tools=[nested_tool],
            config=_default_agent_config(max_steps=5),
        )

        state = parent_agent.run(
            "Use the delegate tool with input='What is 2+2?' to get an answer."
        )

        assert state["termination"]["reason"] == "completed"
        assert len(state["tool_outputs"]) >= 1
        delegate_results = [r for r in state["tool_outputs"] if r.tool_name == "delegate"]
        assert len(delegate_results) >= 1
        assert delegate_results[0].status == "success"


# ===========================================================================
# 11. Error Handling
# ===========================================================================


class TestErrorHandling:
    """Errors are handled gracefully — bad keys, tool failures, etc."""

    def test_invalid_api_key_raises_error(self, model_name: str) -> None:
        """An invalid API key produces an authentication error."""
        bad_config = OpenRouterConfig(model=model_name, api_key="sk-or-invalid-key-12345")
        client = SyncConduit(bad_config)
        agent = Agent(model=client, tools=[], config=_default_agent_config())

        with pytest.raises(ProviderError):
            agent.run("Hello")

        client.close()

    def test_failing_tool_error_captured(self, sync_conduit: SyncConduit) -> None:
        """When a tool raises, the error is captured in ToolResult, not propagated."""
        agent = Agent(
            model=sync_conduit,
            tools=[failing_tool],
            config=_default_agent_config(max_steps=3),
        )

        # The model may or may not call the tool (it says "do not call" in desc).
        # We use a system prompt to force it.
        state = agent.run(
            "Call the always_fails tool right now.",
            system_prompt="You must call the always_fails tool. Do it immediately.",
        )

        # If the model called the tool, verify the error was captured
        fail_results = [r for r in state["tool_outputs"] if r.tool_name == "always_fails"]
        if fail_results:
            assert fail_results[0].status == "failed"
            assert fail_results[0].error is not None
            assert "Intentional test failure" in fail_results[0].error.message

    def test_tool_error_does_not_crash_agent(self, sync_conduit: SyncConduit) -> None:
        """Agent completes even when a tool errors — the model sees the error
        and produces a final response."""
        agent = Agent(
            model=sync_conduit,
            tools=[add_tool, failing_tool],
            config=_default_agent_config(max_steps=5),
        )

        state = agent.run(
            "Try to call always_fails. If it errors, use the add tool " "to compute 1 + 1 instead.",
            system_prompt="Call always_fails first. When it fails, call add(1, 1).",
        )

        # Agent should terminate normally even if tools error
        assert state["termination"]["reason"] in ("completed", "max_steps_reached")


# ===========================================================================
# 12. Conduit-Specific Configuration Keys
# ===========================================================================


class TestConduitSpecificConfig:
    """Conduit-specific keys in InvocationConfig.configurable are forwarded."""

    def test_conduit_context_metadata_forwarded(self, sync_conduit: SyncConduit) -> None:
        """CONDUIT_CONTEXT_METADATA_KEY entries appear in RequestContext.metadata."""
        recorder = ConduitCallRecorder(sync_conduit)
        agent = Agent(model=sync_conduit, tools=[], config=_default_agent_config())

        agent.run(
            "Hello.",
            config=InvocationConfig(
                configurable={
                    CONDUIT_CONTEXT_METADATA_KEY: {
                        "trace_id": "trace-abc",
                        "request_source": "integration_test",
                    },
                },
                thread_id="ctx-thread",
            ),
        )

        assert len(recorder.calls) >= 1
        context = recorder.calls[0]["context"]
        assert isinstance(context, RequestContext)
        assert context.metadata["trace_id"] == "trace-abc"
        assert context.metadata["request_source"] == "integration_test"
        assert context.thread_id == "ctx-thread"
        # context_metadata should NOT leak into config_overrides
        overrides = recorder.calls[0]["config_overrides"]
        if overrides:
            assert CONDUIT_CONTEXT_METADATA_KEY not in overrides

    def test_conduit_runtime_overrides_forwarded(self, sync_conduit: SyncConduit) -> None:
        """CONDUIT_RUNTIME_OVERRIDES_KEY entries are passed as runtime_overrides."""
        recorder = ConduitCallRecorder(sync_conduit)
        agent = Agent(model=sync_conduit, tools=[], config=_default_agent_config())

        agent.run(
            "Hello.",
            config=InvocationConfig(
                configurable={
                    CONDUIT_RUNTIME_OVERRIDES_KEY: {"max_tokens": 50},
                    "temperature": 0.5,
                },
            ),
        )

        assert len(recorder.calls) >= 1
        assert recorder.calls[0]["runtime_overrides"] == {"max_tokens": 50}
        assert recorder.calls[0]["config_overrides"] == {"temperature": 0.5}

    def test_combined_conduit_config_keys(self, sync_conduit: SyncConduit) -> None:
        """All Conduit-specific keys plus regular overrides work together."""
        recorder = ConduitCallRecorder(sync_conduit)
        agent = Agent(model=sync_conduit, tools=[], config=_default_agent_config())

        agent.run(
            "Hello.",
            config=InvocationConfig(
                thread_id="combo-thread",
                tags=["a", "b"],
                configurable={
                    CONDUIT_CONTEXT_METADATA_KEY: {"env": "test"},
                    CONDUIT_RUNTIME_OVERRIDES_KEY: {"top_p": 0.9},
                    "temperature": 0.3,
                },
            ),
        )

        assert len(recorder.calls) >= 1
        call = recorder.calls[0]
        assert call["config_overrides"] == {"temperature": 0.3}
        assert call["runtime_overrides"] == {"top_p": 0.9}
        ctx = call["context"]
        assert isinstance(ctx, RequestContext)
        assert ctx.thread_id == "combo-thread"
        assert ctx.tags == ["a", "b"]
        assert ctx.metadata == {"env": "test"}


# ===========================================================================
# 13. Agent Rebuild & Tool Mutation
# ===========================================================================


class TestAgentRebuild:
    """Adding/removing tools triggers a graph rebuild that works with Conduit."""

    def test_add_tool_triggers_rebuild(self, sync_conduit: SyncConduit) -> None:
        """After adding a tool dynamically, the agent uses it correctly."""
        agent = Agent(model=sync_conduit, tools=[], config=_default_agent_config())

        # First run: no tools available
        state1 = agent.run("Say hello.")
        assert state1["termination"]["reason"] == "completed"
        assert len(state1["tool_outputs"]) == 0

        # Add a tool dynamically
        agent.add_tool(add_tool)
        state2 = agent.run("Use the add tool to compute 5 + 5.")
        assert state2["termination"]["reason"] == "completed"
        add_results = [r for r in state2["tool_outputs"] if r.tool_name == "add"]
        assert len(add_results) >= 1
        assert add_results[0].data == 10

    def test_remove_tool_triggers_rebuild(self, sync_conduit: SyncConduit) -> None:
        """After removing a tool, the agent no longer sends its schema."""
        recorder = ConduitCallRecorder(sync_conduit)
        agent = Agent(
            model=sync_conduit,
            tools=[add_tool, multiply_tool],
            config=_default_agent_config(),
        )

        # First run with both tools
        agent.run("Hello.")
        assert len(recorder.calls) >= 1
        first_tools = recorder.calls[0]["tools"]
        assert first_tools is not None
        first_names = {t.name for t in first_tools}
        assert "add" in first_names
        assert "multiply" in first_names

        # Remove multiply
        agent.remove_tool("multiply")
        recorder.calls.clear()
        agent.run("Hello again.")

        assert len(recorder.calls) >= 1
        second_tools = recorder.calls[0]["tools"]
        assert second_tools is not None
        second_names = {t.name for t in second_tools}
        assert "add" in second_names
        assert "multiply" not in second_names


# ===========================================================================
# 14. Async with Tools
# ===========================================================================


class TestAsyncWithTools:
    """Async agent paths with tool calling via Conduit (async)."""

    async def test_async_tool_call_and_response(self, async_conduit: Conduit) -> None:
        """Async agent calls a tool and returns a correct result."""
        agent = Agent(
            model=async_conduit,
            tools=[multiply_tool],
            config=_default_agent_config(),
        )
        state = await agent.arun("Use the multiply tool to compute 7 * 8. Report the result.")

        assert state["termination"]["reason"] == "completed"
        mul_results = [r for r in state["tool_outputs"] if r.tool_name == "multiply"]
        assert len(mul_results) >= 1
        assert mul_results[0].data == 56

    async def test_async_stream_with_tools(self, async_conduit: Conduit) -> None:
        """Async streaming with tool calls produces the expected events."""
        agent = Agent(
            model=async_conduit,
            tools=[reverse_string_tool],
            config=_default_agent_config(),
        )
        events = [
            event
            async for event in agent.astream("Use the reverse_string tool to reverse 'world'.")
        ]

        event_types = [type(e) for e in events]
        assert RunStartEvent in event_types
        assert RunEndEvent in event_types
        assert ToolStartEvent in event_types
        assert ToolEndEvent in event_types

    async def test_async_config_forwarding(self, async_conduit: Conduit) -> None:
        """Config forwarding works correctly in async mode."""
        recorder = AsyncConduitCallRecorder(async_conduit)
        agent = Agent(model=async_conduit, tools=[], config=_default_agent_config())

        await agent.arun(
            "Hello.",
            config=InvocationConfig(
                thread_id="async-thread",
                tags=["async-test"],
                configurable={"temperature": 0.2},
            ),
        )

        assert len(recorder.calls) >= 1
        call = recorder.calls[0]
        assert call["config_overrides"] == {"temperature": 0.2}
        context = call["context"]
        assert isinstance(context, RequestContext)
        assert context.thread_id == "async-thread"
        assert context.tags == ["async-test"]

    async def test_async_metrics_middleware(self, async_conduit: Conduit) -> None:
        """MetricsMiddleware works in async mode."""
        agent = Agent(
            model=async_conduit,
            tools=[add_tool],
            config=_default_agent_config(),
        )
        agent.add_middleware(MetricsMiddleware())

        state = await agent.arun("Use the add tool to compute 50 + 50.")

        assert state["termination"]["reason"] == "completed"
        metrics = state["metadata"].get("metrics", {})
        assert "add" in metrics
        assert metrics["add"]["count"] >= 1
