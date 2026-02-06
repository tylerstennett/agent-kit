from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from ..config import ToolDecision, ToolResult
from ..state import AgentState
from ..utils.async_utils import maybe_await, maybe_await_sync
from .base import ArgumentSpec, BaseTool, infer_argument_specs

ToolCallable = Callable[..., Any]
PreHook = Callable[[AgentState], ToolDecision | Awaitable[ToolDecision]]
PostHook = Callable[
    [AgentState, ToolResult],
    tuple[AgentState, ToolResult] | Awaitable[tuple[AgentState, ToolResult]],
]
F = TypeVar("F", bound=ToolCallable)


class FunctionTool(BaseTool):
    def __init__(
        self,
        func: ToolCallable,
        *,
        name: str | None = None,
        description: str | None = None,
        pre_execute_hook: PreHook | None = None,
        post_execute_hook: PostHook | None = None,
        arg_specs: dict[str, ArgumentSpec] | None = None,
    ) -> None:
        self._func = func
        self._pre_hook = pre_execute_hook
        self._post_hook = post_execute_hook
        derived_description = description or (inspect.getdoc(func) or "")
        super().__init__(
            name=name or func.__name__,
            description=derived_description,
            arg_specs=arg_specs or infer_argument_specs(func),
            allow_extra_args=_accepts_var_kwargs(func),
        )

    def pre_execute(self, state: AgentState) -> ToolDecision:
        if self._pre_hook is None:
            return super().pre_execute(state)
        return maybe_await_sync(self._pre_hook(state))

    async def apre_execute(self, state: AgentState) -> ToolDecision:
        if self._pre_hook is None:
            return await super().apre_execute(state)
        return await maybe_await(self._pre_hook(state))

    def execute(self, state: AgentState, **kwargs: Any) -> Any:
        return self._func(state, **kwargs)

    async def aexecute(self, state: AgentState, **kwargs: Any) -> Any:
        output = self._func(state, **kwargs)
        if inspect.isawaitable(output):
            return await output
        return output

    def post_execute(self, state: AgentState, result: ToolResult) -> tuple[AgentState, ToolResult]:
        if self._post_hook is None:
            return super().post_execute(state, result)
        return maybe_await_sync(self._post_hook(state, result))

    async def apost_execute(
        self, state: AgentState, result: ToolResult
    ) -> tuple[AgentState, ToolResult]:
        if self._post_hook is None:
            return await super().apost_execute(state, result)
        return await maybe_await(self._post_hook(state, result))


def tool(
    func: F | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    pre_execute: PreHook | None = None,
    post_execute: PostHook | None = None,
    arg_specs: dict[str, ArgumentSpec] | None = None,
) -> Callable[[F], FunctionTool] | FunctionTool:
    def decorate(inner: F) -> FunctionTool:
        return FunctionTool(
            inner,
            name=name,
            description=description,
            pre_execute_hook=pre_execute,
            post_execute_hook=post_execute,
            arg_specs=arg_specs,
        )

    if func is None:
        return decorate
    return decorate(func)


def _accepts_var_kwargs(func: ToolCallable) -> bool:
    signature = inspect.signature(func)
    return any(
        param.kind is inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()
    )
