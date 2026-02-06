from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import Any, Protocol

from ..config import GraphBuildConfig, InvocationConfig
from ..events import StreamEvent
from ..model_adapter import ModelAdapter
from ..state import AgentState
from ..tools.base import BaseTool


class CompiledGraph(Protocol):
    def invoke(self, state: AgentState, config: InvocationConfig) -> AgentState: ...

    async def ainvoke(self, state: AgentState, config: InvocationConfig) -> AgentState: ...

    def stream(self, state: AgentState, config: InvocationConfig) -> Iterator[StreamEvent]: ...

    def astream(self, state: AgentState, config: InvocationConfig) -> AsyncIterator[StreamEvent]:
        ...


class GraphBuilder(Protocol):
    def build(
        self,
        model: ModelAdapter,
        tools: list[BaseTool],
        config: GraphBuildConfig,
        *,
        before_hooks: list[Any] | None = None,
        after_hooks: list[Any] | None = None,
        middlewares: list[Any] | None = None,
    ) -> CompiledGraph: ...
