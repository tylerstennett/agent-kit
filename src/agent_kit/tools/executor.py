from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from ..config import (
    AfterToolHook,
    BeforeToolHook,
    ExecutionMode,
    InvocationConfig,
    ToolDecision,
    ToolError,
    ToolResult,
)
from ..errors import ToolValidationError
from ..events import ToolEndEvent, ToolStartEvent
from ..middleware import (
    Middleware,
    ToolExecutionOutcome,
    ToolInvocationContext,
    run_middleware_async,
    run_middleware_sync,
)
from ..state import AgentState
from ..utils.async_utils import maybe_await, maybe_await_sync
from ..utils.state_utils import clone_state, merge_parallel_state_deltas, state_delta
from .base import BaseTool, validate_args
from .registry import ToolRegistry

EventSink = Callable[[object], None]


class ToolExecutor:
    def __init__(
        self,
        registry: ToolRegistry,
        *,
        before_hooks: list[BeforeToolHook] | None = None,
        after_hooks: list[AfterToolHook] | None = None,
        middlewares: list[Middleware] | None = None,
        max_parallel_workers: int = 4,
    ) -> None:
        self._registry = registry
        self._before_hooks = list(before_hooks or [])
        self._after_hooks = list(after_hooks or [])
        self._middlewares = list(middlewares or [])
        self._max_parallel_workers = max_parallel_workers

    def execute_calls(
        self,
        state: AgentState,
        tool_calls: list[dict[str, Any]],
        config: InvocationConfig,
        *,
        event_sink: EventSink | None = None,
    ) -> tuple[AgentState, list[ToolResult]]:
        mode: ExecutionMode = config.execution_mode or "sequential"
        if mode == "parallel" and len(tool_calls) > 1:
            return self._execute_parallel_sync(state, tool_calls, config, event_sink=event_sink)
        return self._execute_sequential_sync(state, tool_calls, config, event_sink=event_sink)

    async def aexecute_calls(
        self,
        state: AgentState,
        tool_calls: list[dict[str, Any]],
        config: InvocationConfig,
        *,
        event_sink: EventSink | None = None,
    ) -> tuple[AgentState, list[ToolResult]]:
        mode: ExecutionMode = config.execution_mode or "sequential"
        if mode == "parallel" and len(tool_calls) > 1:
            return await self._execute_parallel_async(
                state, tool_calls, config, event_sink=event_sink
            )
        return await self._execute_sequential_async(
            state, tool_calls, config, event_sink=event_sink
        )

    def _execute_sequential_sync(
        self,
        state: AgentState,
        tool_calls: list[dict[str, Any]],
        config: InvocationConfig,
        *,
        event_sink: EventSink | None,
    ) -> tuple[AgentState, list[ToolResult]]:
        current = state
        results: list[ToolResult] = []
        for call in tool_calls:
            current, result = self._execute_one_sync(current, call, config, event_sink=event_sink)
            results.append(result)
        return current, results

    async def _execute_sequential_async(
        self,
        state: AgentState,
        tool_calls: list[dict[str, Any]],
        config: InvocationConfig,
        *,
        event_sink: EventSink | None,
    ) -> tuple[AgentState, list[ToolResult]]:
        current = state
        results: list[ToolResult] = []
        for call in tool_calls:
            current, result = await self._execute_one_async(
                current, call, config, event_sink=event_sink
            )
            results.append(result)
        return current, results

    def _execute_parallel_sync(
        self,
        state: AgentState,
        tool_calls: list[dict[str, Any]],
        config: InvocationConfig,
        *,
        event_sink: EventSink | None,
    ) -> tuple[AgentState, list[ToolResult]]:
        base_state = clone_state(state)
        max_workers = config.max_parallel_workers or self._max_parallel_workers
        outcomes: dict[int, tuple[AgentState, ToolResult]] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [
                pool.submit(
                    self._execute_one_sync,
                    clone_state(base_state),
                    tool_call,
                    config,
                    event_sink,
                )
                for tool_call in tool_calls
            ]
            for index, future in enumerate(futures):
                outcomes[index] = future.result()
        deltas: list[dict[str, Any]] = []
        results: list[ToolResult] = []
        for index in range(len(tool_calls)):
            outcome_state, result = outcomes[index]
            deltas.append(state_delta(base_state, outcome_state))
            results.append(result)
        merged_state = merge_parallel_state_deltas(base_state, deltas)
        return merged_state, results

    async def _execute_parallel_async(
        self,
        state: AgentState,
        tool_calls: list[dict[str, Any]],
        config: InvocationConfig,
        *,
        event_sink: EventSink | None,
    ) -> tuple[AgentState, list[ToolResult]]:
        base_state = clone_state(state)
        sem = asyncio.Semaphore(config.max_parallel_workers or self._max_parallel_workers)

        async def run_one(call: dict[str, Any]) -> tuple[AgentState, ToolResult]:
            async with sem:
                return await self._execute_one_async(
                    clone_state(base_state),
                    call,
                    config,
                    event_sink=event_sink,
                )

        outcomes_list = await asyncio.gather(*(run_one(call) for call in tool_calls))
        deltas: list[dict[str, Any]] = []
        results: list[ToolResult] = []
        for outcome_state, result in outcomes_list:
            deltas.append(state_delta(base_state, outcome_state))
            results.append(result)
        merged_state = merge_parallel_state_deltas(base_state, deltas)
        return merged_state, results

    def _execute_one_sync(
        self,
        state: AgentState,
        tool_call: dict[str, Any],
        config: InvocationConfig,
        event_sink: EventSink | None,
    ) -> tuple[AgentState, ToolResult]:
        tool_name = str(tool_call.get("name", ""))
        call_id = str(tool_call.get("id", ""))
        tool_args = dict(tool_call.get("args", {}))
        tool = self._registry.get(tool_name)
        if tool is None:
            result = ToolResult(
                tool_name=tool_name,
                call_id=call_id or "",
                status="failed",
                error=ToolError(
                    code="tool_not_found", message=f"Tool '{tool_name}' was not found."
                ),
            )
            return state, result

        call_id = call_id or tool.next_call_id()
        context = ToolInvocationContext(
            state=state,
            tool_name=tool_name,
            call_id=call_id,
            tool_args=tool_args,
            metadata={"mode": config.execution_mode or "sequential"},
        )
        if event_sink:
            event_sink(
                ToolStartEvent(run_id="", tool_name=tool_name, call_id=call_id, tool_args=tool_args)
            )

        def terminal(ctx: ToolInvocationContext) -> ToolExecutionOutcome:
            return self._run_tool_pipeline_sync(tool, ctx)

        try:
            outcome = run_middleware_sync(context, self._middlewares, terminal)
        except Exception as exc:
            started = time.perf_counter()
            result = tool.normalize_failure(call_id, exc, started, code="middleware_error")
            outcome = ToolExecutionOutcome(state=state, result=result)

        if event_sink:
            event_sink(ToolEndEvent(run_id="", result=outcome.result))
        return outcome.state, outcome.result

    async def _execute_one_async(
        self,
        state: AgentState,
        tool_call: dict[str, Any],
        config: InvocationConfig,
        event_sink: EventSink | None,
    ) -> tuple[AgentState, ToolResult]:
        tool_name = str(tool_call.get("name", ""))
        call_id = str(tool_call.get("id", ""))
        tool_args = dict(tool_call.get("args", {}))
        tool = self._registry.get(tool_name)
        if tool is None:
            result = ToolResult(
                tool_name=tool_name,
                call_id=call_id or "",
                status="failed",
                error=ToolError(
                    code="tool_not_found", message=f"Tool '{tool_name}' was not found."
                ),
            )
            return state, result

        call_id = call_id or tool.next_call_id()
        context = ToolInvocationContext(
            state=state,
            tool_name=tool_name,
            call_id=call_id,
            tool_args=tool_args,
            metadata={"mode": config.execution_mode or "sequential"},
        )
        if event_sink:
            event_sink(
                ToolStartEvent(run_id="", tool_name=tool_name, call_id=call_id, tool_args=tool_args)
            )

        async def terminal(ctx: ToolInvocationContext) -> ToolExecutionOutcome:
            return await self._run_tool_pipeline_async(tool, ctx)

        try:
            outcome = await run_middleware_async(context, self._middlewares, terminal)
        except Exception as exc:
            started = time.perf_counter()
            result = tool.normalize_failure(call_id, exc, started, code="middleware_error")
            outcome = ToolExecutionOutcome(state=state, result=result)

        if event_sink:
            event_sink(ToolEndEvent(run_id="", result=outcome.result))
        return outcome.state, outcome.result

    def _run_tool_pipeline_sync(
        self, tool: BaseTool, context: ToolInvocationContext
    ) -> ToolExecutionOutcome:
        state = context.state
        started = time.perf_counter()

        for hook in self._before_hooks:
            try:
                state = maybe_await_sync(hook(state, tool.name, context.tool_args))
            except Exception as exc:
                result = tool.normalize_failure(
                    context.call_id, exc, started, code="before_hook_error"
                )
                return ToolExecutionOutcome(state=state, result=result)

        try:
            decision: ToolDecision = maybe_await_sync(tool.apre_execute(state))
        except Exception as exc:
            result = tool.normalize_failure(context.call_id, exc, started, code="pre_execute_error")
            state, result = self._apply_after_hooks_sync(state, tool.name, result, started)
            return ToolExecutionOutcome(state=state, result=result)

        if decision.action == "skip":
            result = decision.result_override or ToolResult(
                tool_name=tool.name,
                call_id=context.call_id,
                status="skipped",
                error=ToolError(
                    code="skipped", message=decision.reason or "Skipped by pre_execute."
                ),
                started_at=started,
                ended_at=time.perf_counter(),
            )
            if result.duration_ms is None and result.started_at and result.ended_at:
                result.duration_ms = (result.ended_at - result.started_at) * 1000
            state, result = self._apply_after_hooks_sync(state, tool.name, result, started)
            return ToolExecutionOutcome(state=state, result=result)

        try:
            validate_args(tool.arg_specs, context.tool_args, allow_extra=tool.allow_extra_args)
            output = maybe_await_sync(tool.aexecute(state, **context.tool_args))
            ended = time.perf_counter()
            result = tool.normalize_success(context.call_id, output, started, ended)
            state, result = maybe_await_sync(tool.apost_execute(state, result))
        except ToolValidationError as exc:
            result = tool.normalize_failure(context.call_id, exc, started, code="validation_error")
        except Exception as exc:
            result = tool.normalize_failure(context.call_id, exc, started, code="execution_error")

        state, result = self._apply_after_hooks_sync(state, tool.name, result, started)
        return ToolExecutionOutcome(state=state, result=result)

    async def _run_tool_pipeline_async(
        self, tool: BaseTool, context: ToolInvocationContext
    ) -> ToolExecutionOutcome:
        state = context.state
        started = time.perf_counter()

        for hook in self._before_hooks:
            try:
                state = await maybe_await(hook(state, tool.name, context.tool_args))
            except Exception as exc:
                result = tool.normalize_failure(
                    context.call_id, exc, started, code="before_hook_error"
                )
                return ToolExecutionOutcome(state=state, result=result)

        try:
            decision = await tool.apre_execute(state)
        except Exception as exc:
            result = tool.normalize_failure(context.call_id, exc, started, code="pre_execute_error")
            state, result = await self._apply_after_hooks_async(state, tool.name, result, started)
            return ToolExecutionOutcome(state=state, result=result)

        if decision.action == "skip":
            result = decision.result_override or ToolResult(
                tool_name=tool.name,
                call_id=context.call_id,
                status="skipped",
                error=ToolError(
                    code="skipped", message=decision.reason or "Skipped by pre_execute."
                ),
                started_at=started,
                ended_at=time.perf_counter(),
            )
            if result.duration_ms is None and result.started_at and result.ended_at:
                result.duration_ms = (result.ended_at - result.started_at) * 1000
            state, result = await self._apply_after_hooks_async(state, tool.name, result, started)
            return ToolExecutionOutcome(state=state, result=result)

        try:
            validate_args(tool.arg_specs, context.tool_args, allow_extra=tool.allow_extra_args)
            output = await tool.aexecute(state, **context.tool_args)
            ended = time.perf_counter()
            result = tool.normalize_success(context.call_id, output, started, ended)
            state, result = await tool.apost_execute(state, result)
        except ToolValidationError as exc:
            result = tool.normalize_failure(context.call_id, exc, started, code="validation_error")
        except Exception as exc:
            result = tool.normalize_failure(context.call_id, exc, started, code="execution_error")

        state, result = await self._apply_after_hooks_async(state, tool.name, result, started)
        return ToolExecutionOutcome(state=state, result=result)

    def _apply_after_hooks_sync(
        self,
        state: AgentState,
        tool_name: str,
        result: ToolResult,
        started: float,
    ) -> tuple[AgentState, ToolResult]:
        current_state = state
        current_result = result
        for hook in self._after_hooks:
            try:
                current_state, current_result = maybe_await_sync(
                    hook(current_state, tool_name, current_result)
                )
            except Exception as exc:
                current_result = ToolResult(
                    tool_name=tool_name,
                    call_id=current_result.call_id,
                    status="failed",
                    error=ToolError(code="after_hook_error", message=str(exc)),
                    started_at=started,
                    ended_at=time.perf_counter(),
                )
                if current_result.started_at and current_result.ended_at:
                    current_result.duration_ms = (
                        current_result.ended_at - current_result.started_at
                    ) * 1000
                break
        return current_state, current_result

    async def _apply_after_hooks_async(
        self,
        state: AgentState,
        tool_name: str,
        result: ToolResult,
        started: float,
    ) -> tuple[AgentState, ToolResult]:
        current_state = state
        current_result = result
        for hook in self._after_hooks:
            try:
                current_state, current_result = await maybe_await(
                    hook(current_state, tool_name, current_result)
                )
            except Exception as exc:
                current_result = ToolResult(
                    tool_name=tool_name,
                    call_id=current_result.call_id,
                    status="failed",
                    error=ToolError(code="after_hook_error", message=str(exc)),
                    started_at=started,
                    ended_at=time.perf_counter(),
                )
                if current_result.started_at and current_result.ended_at:
                    current_result.duration_ms = (
                        current_result.ended_at - current_result.started_at
                    ) * 1000
                break
        return current_state, current_result
