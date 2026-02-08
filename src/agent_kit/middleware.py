from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from agent_kit.config import ToolError, ToolResult
from agent_kit.utils.async_utils import maybe_await, maybe_await_sync

if TYPE_CHECKING:
    from agent_kit.state import AgentState


@dataclass(slots=True)
class ToolInvocationContext:
    state: AgentState
    tool_name: str
    call_id: str
    tool_args: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ToolExecutionOutcome:
    state: AgentState
    result: ToolResult


NextCallable = Callable[
    [ToolInvocationContext], ToolExecutionOutcome | Awaitable[ToolExecutionOutcome]
]
Middleware = Callable[
    [ToolInvocationContext, NextCallable],
    ToolExecutionOutcome | Awaitable[ToolExecutionOutcome],
]


def run_middleware_sync(
    context: ToolInvocationContext,
    middlewares: list[Middleware],
    terminal: NextCallable,
) -> ToolExecutionOutcome:
    handler = terminal
    for middleware in reversed(middlewares):
        next_handler = handler

        def wrapped(
            ctx: ToolInvocationContext,
            *,
            mw: Middleware = middleware,
            nxt: NextCallable = next_handler,
        ) -> ToolExecutionOutcome:
            return maybe_await_sync(mw(ctx, nxt))

        handler = wrapped
    return maybe_await_sync(handler(context))


async def run_middleware_async(
    context: ToolInvocationContext,
    middlewares: list[Middleware],
    terminal: NextCallable,
) -> ToolExecutionOutcome:
    async def call_at(index: int, ctx: ToolInvocationContext) -> ToolExecutionOutcome:
        if index >= len(middlewares):
            return await maybe_await(terminal(ctx))
        middleware = middlewares[index]

        async def next_call(next_ctx: ToolInvocationContext) -> ToolExecutionOutcome:
            return await call_at(index + 1, next_ctx)

        return await maybe_await(middleware(ctx, next_call))

    return await call_at(0, context)


def logging_middleware(logger: logging.Logger | None = None) -> Middleware:
    active_logger = logger or logging.getLogger("agent_kit")

    def middleware(context: ToolInvocationContext, next_call: NextCallable) -> ToolExecutionOutcome:
        active_logger.info("tool_start name=%s call_id=%s", context.tool_name, context.call_id)
        outcome = maybe_await_sync(next_call(context))
        active_logger.info(
            "tool_end name=%s call_id=%s status=%s",
            context.tool_name,
            context.call_id,
            outcome.result.status,
        )
        return outcome

    return middleware


@dataclass(slots=True)
class RetryMiddleware:
    max_retries: int = 1

    async def __call__(
        self,
        context: ToolInvocationContext,
        next_call: NextCallable,
    ) -> ToolExecutionOutcome:
        attempt = 0
        last_outcome: ToolExecutionOutcome | None = None
        while attempt <= self.max_retries:
            outcome = await maybe_await(next_call(context))
            last_outcome = outcome
            if outcome.result.status == "success":
                return outcome
            if not outcome.result.error or not outcome.result.error.retriable:
                return outcome
            attempt += 1
        if last_outcome is not None:
            return last_outcome
        return ToolExecutionOutcome(
            state=context.state,
            result=ToolResult(
                tool_name=context.tool_name,
                call_id=context.call_id,
                status="failed",
                error=ToolError(
                    code="retry_error", message="Retry middleware produced no outcome."
                ),
            ),
        )


@dataclass(slots=True)
class TimeoutMiddleware:
    timeout_seconds: float

    async def __call__(
        self,
        context: ToolInvocationContext,
        next_call: NextCallable,
    ) -> ToolExecutionOutcome:
        started = time.perf_counter()
        try:
            return await asyncio.wait_for(
                maybe_await(next_call(context)), timeout=self.timeout_seconds
            )
        except TimeoutError:
            state = context.state
            result = ToolResult(
                tool_name=context.tool_name,
                call_id=context.call_id,
                status="failed",
                error=ToolError(
                    code="timeout",
                    message=f"Tool exceeded timeout of {self.timeout_seconds} seconds.",
                    retriable=True,
                ),
            )
            result.started_at = started
            result.ended_at = time.perf_counter()
            result.duration_ms = (result.ended_at - started) * 1000
            return ToolExecutionOutcome(state=state, result=result)


@dataclass(slots=True)
class MetricsMiddleware:
    key_prefix: str = "metrics"

    def __call__(
        self, context: ToolInvocationContext, next_call: NextCallable
    ) -> ToolExecutionOutcome:
        started = time.perf_counter()
        outcome = maybe_await_sync(next_call(context))
        elapsed = (time.perf_counter() - started) * 1000
        metadata = dict(outcome.state.get("metadata", {}))
        metrics = dict(metadata.get(self.key_prefix, {}))
        tool_metrics = dict(metrics.get(context.tool_name, {}))
        tool_metrics["count"] = int(tool_metrics.get("count", 0)) + 1
        tool_metrics["last_duration_ms"] = elapsed
        metrics[context.tool_name] = tool_metrics
        metadata[self.key_prefix] = metrics
        outcome.state["metadata"] = metadata
        return outcome
