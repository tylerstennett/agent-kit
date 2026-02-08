from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from typing import Any, cast
from uuid import uuid4

from agent_kit.config import GraphBuildConfig, InvocationConfig, ToolResult
from agent_kit.events import ErrorEvent, RunEndEvent, RunStartEvent, StateUpdateEvent, StreamEvent
from agent_kit.graph.base import CompiledGraph, GraphBuilder
from agent_kit.graph.nodes import aaction_step, action_step, areasoning_step, reasoning_step
from agent_kit.llm.types import ModelAdapter
from agent_kit.state import AgentState
from agent_kit.tools.base import BaseTool
from agent_kit.tools.executor import ToolExecutor
from agent_kit.tools.registry import ToolRegistry
from agent_kit.utils.state_utils import clone_state


def _set_termination(state: AgentState, reason: str, steps: int) -> AgentState:
    termination = dict(state.get("termination", {}))
    termination["reason"] = reason
    termination["steps"] = steps
    state["termination"] = termination
    return state


def _has_validation_error(results: list[ToolResult]) -> bool:
    for result in results:
        if result.error is not None and result.error.code == "validation_error":
            return True
    return False


@dataclass(slots=True)
class ReActCompiledGraph:
    model: ModelAdapter
    executor: ToolExecutor
    config: GraphBuildConfig

    def _effective_config(self, invocation_config: InvocationConfig) -> InvocationConfig:
        validation_repair_turns = invocation_config.validation_repair_turns
        if validation_repair_turns is None:
            validation_repair_turns = self.config.default_validation_repair_turns
        validation_repair_turns = max(validation_repair_turns, 0)
        effective = InvocationConfig(
            execution_mode=invocation_config.execution_mode or self.config.default_execution_mode,
            recursion_limit=invocation_config.recursion_limit,
            max_steps=invocation_config.max_steps,
            include_state_snapshots=invocation_config.include_state_snapshots,
            thread_id=invocation_config.thread_id,
            configurable={**self.config.model_config_overrides, **invocation_config.configurable},
            tags=list(invocation_config.tags),
            max_parallel_workers=invocation_config.max_parallel_workers
            or self.config.max_parallel_workers,
            emit_llm_tokens=invocation_config.emit_llm_tokens,
            validation_repair_turns=validation_repair_turns,
        )
        return effective

    def invoke(self, state: AgentState, config: InvocationConfig) -> AgentState:
        effective = self._effective_config(config)
        max_steps = effective.max_steps or self.config.max_steps
        repair_budget = effective.validation_repair_turns or 0
        consecutive_validation_failures = 0
        current = clone_state(state)
        for step in range(max_steps):
            current, tool_calls = reasoning_step(current, self.model, effective)
            if not tool_calls:
                return _set_termination(current, "completed", step + 1)
            current, results = action_step(current, tool_calls, self.executor, effective)
            if _has_validation_error(results):
                consecutive_validation_failures += 1
                if consecutive_validation_failures > repair_budget:
                    return _set_termination(current, "validation_repair_exhausted", step + 1)
            else:
                consecutive_validation_failures = 0
        return _set_termination(current, "max_steps_reached", max_steps)

    async def ainvoke(self, state: AgentState, config: InvocationConfig) -> AgentState:
        effective = self._effective_config(config)
        max_steps = effective.max_steps or self.config.max_steps
        repair_budget = effective.validation_repair_turns or 0
        consecutive_validation_failures = 0
        current = clone_state(state)
        for step in range(max_steps):
            current, tool_calls = await areasoning_step(current, self.model, effective)
            if not tool_calls:
                return _set_termination(current, "completed", step + 1)
            current, results = await aaction_step(current, tool_calls, self.executor, effective)
            if _has_validation_error(results):
                consecutive_validation_failures += 1
                if consecutive_validation_failures > repair_budget:
                    return _set_termination(current, "validation_repair_exhausted", step + 1)
            else:
                consecutive_validation_failures = 0
        return _set_termination(current, "max_steps_reached", max_steps)

    def stream(self, state: AgentState, config: InvocationConfig) -> Iterator[StreamEvent]:
        effective = self._effective_config(config)
        max_steps = effective.max_steps or self.config.max_steps
        repair_budget = effective.validation_repair_turns or 0
        consecutive_validation_failures = 0
        run_id = uuid4().hex
        current = clone_state(state)
        pending: list[StreamEvent] = []

        def sink(event: object) -> None:
            stream_event = cast(StreamEvent, event)
            stream_event.run_id = run_id
            if effective.include_state_snapshots and isinstance(stream_event, StateUpdateEvent):
                stream_event.snapshot = dict(clone_state(current))
            pending.append(stream_event)

        yield RunStartEvent(run_id=run_id)
        try:
            for step in range(max_steps):
                current, tool_calls = reasoning_step(
                    current, self.model, effective, event_sink=sink
                )
                while pending:
                    yield pending.pop(0)
                if not tool_calls:
                    current = _set_termination(current, "completed", step + 1)
                    break
                current, results = action_step(
                    current, tool_calls, self.executor, effective, event_sink=sink
                )
                while pending:
                    yield pending.pop(0)
                if _has_validation_error(results):
                    consecutive_validation_failures += 1
                    if consecutive_validation_failures > repair_budget:
                        current = _set_termination(current, "validation_repair_exhausted", step + 1)
                        break
                else:
                    consecutive_validation_failures = 0
            else:
                current = _set_termination(current, "max_steps_reached", max_steps)
        except Exception as exc:
            current = _set_termination(current, "error", 0)
            yield ErrorEvent(run_id=run_id, message=str(exc))
        yield RunEndEvent(run_id=run_id, termination=dict(current.get("termination", {})))

    async def astream(
        self, state: AgentState, config: InvocationConfig
    ) -> AsyncIterator[StreamEvent]:
        effective = self._effective_config(config)
        max_steps = effective.max_steps or self.config.max_steps
        repair_budget = effective.validation_repair_turns or 0
        consecutive_validation_failures = 0
        run_id = uuid4().hex
        current = clone_state(state)
        pending: list[StreamEvent] = []

        def sink(event: object) -> None:
            stream_event = cast(StreamEvent, event)
            stream_event.run_id = run_id
            if effective.include_state_snapshots and isinstance(stream_event, StateUpdateEvent):
                stream_event.snapshot = dict(clone_state(current))
            pending.append(stream_event)

        yield RunStartEvent(run_id=run_id)
        try:
            for step in range(max_steps):
                current, tool_calls = await areasoning_step(
                    current, self.model, effective, event_sink=sink
                )
                while pending:
                    yield pending.pop(0)
                if not tool_calls:
                    current = _set_termination(current, "completed", step + 1)
                    break
                current, results = await aaction_step(
                    current, tool_calls, self.executor, effective, event_sink=sink
                )
                while pending:
                    yield pending.pop(0)
                if _has_validation_error(results):
                    consecutive_validation_failures += 1
                    if consecutive_validation_failures > repair_budget:
                        current = _set_termination(current, "validation_repair_exhausted", step + 1)
                        break
                else:
                    consecutive_validation_failures = 0
            else:
                current = _set_termination(current, "max_steps_reached", max_steps)
        except Exception as exc:
            current = _set_termination(current, "error", 0)
            yield ErrorEvent(run_id=run_id, message=str(exc))
        yield RunEndEvent(run_id=run_id, termination=dict(current.get("termination", {})))


class ReActGraphBuilder(GraphBuilder):
    def build(
        self,
        model: ModelAdapter,
        tools: list[BaseTool],
        config: GraphBuildConfig,
        *,
        before_hooks: list[Any] | None = None,
        after_hooks: list[Any] | None = None,
        middlewares: list[Any] | None = None,
    ) -> CompiledGraph:
        registry = ToolRegistry(tools)
        executor = ToolExecutor(
            registry,
            before_hooks=list(before_hooks or []),
            after_hooks=list(after_hooks or []),
            middlewares=list(middlewares or []),
            max_parallel_workers=config.max_parallel_workers,
        )
        return ReActCompiledGraph(model=model, executor=executor, config=config)
