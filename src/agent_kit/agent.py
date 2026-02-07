from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator, Iterator
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from agent_kit.config import (
    AfterToolHook,
    AgentConfig,
    BeforeToolHook,
    GraphBuildConfig,
    InvocationConfig,
    InvocationRequest,
)
from agent_kit.errors import SyncInAsyncContextError
from agent_kit.events import StreamEvent
from agent_kit.graph import CompiledGraph, GraphBuilder, ReActGraphBuilder
from agent_kit.invocation import build_initial_state, normalize_request
from agent_kit.middleware import Middleware
from agent_kit.model_adapter import LangChainModelAdapter, ModelAdapter
from agent_kit.nested import NestedAgentPolicy, agent_as_tool
from agent_kit.state import AgentState
from agent_kit.tools.base import BaseTool
from agent_kit.tools.model_binding import bind_model_tools
from agent_kit.tools.registry import ToolRegistry
from agent_kit.tools.schema import tool_schema_signature, tools_to_model_schemas
from agent_kit.utils.state_utils import merge_metadata


class Agent:
    def __init__(
        self,
        *,
        model: BaseChatModel | ModelAdapter,
        tools: list[BaseTool] | None = None,
        graph_builder: GraphBuilder | None = None,
        config: AgentConfig | None = None,
    ) -> None:
        self._config = config or AgentConfig()
        self._graph_builder = graph_builder or ReActGraphBuilder()
        self._model_source = model
        self._model_adapter: ModelAdapter | None = None
        self._last_tool_schema_signature: str | None = None
        self._registry = ToolRegistry(tools or [])
        self._compiled: CompiledGraph | None = None
        self._dirty = True
        self._compile_lock = threading.RLock()

    @property
    def config(self) -> AgentConfig:
        return self._config

    @property
    def tools(self) -> list[BaseTool]:
        return self._registry.values()

    @property
    def last_tool_schema_signature(self) -> str | None:
        return self._last_tool_schema_signature

    def add_tool(self, tool: BaseTool) -> None:
        self._registry.add_tool(tool)
        self._mark_dirty()

    def remove_tool(self, name: str) -> None:
        self._registry.remove_tool(name)
        self._mark_dirty()

    def set_tools(self, tools: list[BaseTool]) -> None:
        self._registry.set_tools(tools)
        self._mark_dirty()

    def add_before_tool_hook(self, hook: BeforeToolHook) -> None:
        self._config.before_tool_hooks.append(hook)
        self._mark_dirty()

    def add_after_tool_hook(self, hook: AfterToolHook) -> None:
        self._config.after_tool_hooks.append(hook)
        self._mark_dirty()

    def add_middleware(self, middleware: Middleware) -> None:
        self._config.middlewares.append(middleware)
        self._mark_dirty()

    def rebuild(
        self,
        *,
        graph_builder: GraphBuilder | None = None,
        tools: list[BaseTool] | None = None,
        graph_config: GraphBuildConfig | None = None,
    ) -> None:
        if graph_builder is not None:
            self._graph_builder = graph_builder
        if tools is not None:
            self._registry.set_tools(tools)
        if graph_config is not None:
            self._config.graph_config = graph_config
        self._mark_dirty()

    def invoke(
        self,
        request: InvocationRequest | str,
        *,
        metadata: dict[str, Any] | None = None,
        system_prompt: str | None = None,
        config: InvocationConfig | None = None,
    ) -> AgentState:
        normalized = normalize_request(
            request,
            metadata=metadata,
            system_prompt=system_prompt,
            config=config,
        )
        state = build_initial_state(
            normalized,
            default_metadata=self._config.default_metadata,
            metadata_merge=merge_metadata,
        )
        compiled = self._compiled_graph()
        return compiled.invoke(state, normalized.config)

    async def ainvoke(
        self,
        request: InvocationRequest | str,
        *,
        metadata: dict[str, Any] | None = None,
        system_prompt: str | None = None,
        config: InvocationConfig | None = None,
    ) -> AgentState:
        normalized = normalize_request(
            request,
            metadata=metadata,
            system_prompt=system_prompt,
            config=config,
        )
        state = build_initial_state(
            normalized,
            default_metadata=self._config.default_metadata,
            metadata_merge=merge_metadata,
        )
        compiled = self._compiled_graph()
        return await compiled.ainvoke(state, normalized.config)

    def run(
        self,
        user_message: str,
        *,
        system_prompt: str | None = None,
        metadata: dict[str, Any] | None = None,
        config: InvocationConfig | None = None,
    ) -> AgentState:
        self._ensure_not_running_loop()
        return self.invoke(
            user_message,
            system_prompt=system_prompt,
            metadata=metadata,
            config=config,
        )

    async def arun(
        self,
        user_message: str,
        *,
        system_prompt: str | None = None,
        metadata: dict[str, Any] | None = None,
        config: InvocationConfig | None = None,
    ) -> AgentState:
        return await self.ainvoke(
            user_message,
            system_prompt=system_prompt,
            metadata=metadata,
            config=config,
        )

    def stream(
        self,
        request: InvocationRequest | str,
        *,
        metadata: dict[str, Any] | None = None,
        system_prompt: str | None = None,
        config: InvocationConfig | None = None,
    ) -> Iterator[StreamEvent]:
        self._ensure_not_running_loop()
        normalized = normalize_request(
            request,
            metadata=metadata,
            system_prompt=system_prompt,
            config=config,
        )
        state = build_initial_state(
            normalized,
            default_metadata=self._config.default_metadata,
            metadata_merge=merge_metadata,
        )
        compiled = self._compiled_graph()
        return compiled.stream(state, normalized.config)

    async def astream(
        self,
        request: InvocationRequest | str,
        *,
        metadata: dict[str, Any] | None = None,
        system_prompt: str | None = None,
        config: InvocationConfig | None = None,
    ) -> AsyncIterator[StreamEvent]:
        normalized = normalize_request(
            request,
            metadata=metadata,
            system_prompt=system_prompt,
            config=config,
        )
        state = build_initial_state(
            normalized,
            default_metadata=self._config.default_metadata,
            metadata_merge=merge_metadata,
        )
        compiled = self._compiled_graph()
        async for event in compiled.astream(state, normalized.config):
            yield event

    def as_tool(
        self,
        *,
        name: str | None = None,
        description: str = "Delegate to a child agent.",
        policy: NestedAgentPolicy | None = None,
    ) -> BaseTool:
        return agent_as_tool(self, name=name, description=description, policy=policy)

    def _compiled_graph(self) -> CompiledGraph:
        with self._compile_lock:
            if self._compiled is None or self._dirty:
                tools = self._registry.values()
                model_adapter = self._resolve_model_adapter(tools)
                self._compiled = self._graph_builder.build(
                    model_adapter,
                    tools,
                    self._config.graph_config,
                    before_hooks=self._config.before_tool_hooks,
                    after_hooks=self._config.after_tool_hooks,
                    middlewares=self._config.middlewares,
                )
                self._model_adapter = model_adapter
                self._dirty = False
            return self._compiled

    def _resolve_model_adapter(self, tools: list[BaseTool]) -> ModelAdapter:
        model = self._model_source
        if isinstance(model, BaseChatModel):
            return self._resolve_langchain_model(model, tools)
        self._last_tool_schema_signature = None
        return model

    def _resolve_langchain_model(self, model: BaseChatModel, tools: list[BaseTool]) -> ModelAdapter:
        mode = self._config.model_tool_binding_mode
        if mode == "auto":
            if not tools:
                self._last_tool_schema_signature = None
                return LangChainModelAdapter(model)
            schemas = tools_to_model_schemas(tools, policy=self._config.tool_schema_sync_policy)
            bound_model = bind_model_tools(
                model,
                tools,
                self._config.tool_schema_sync_policy,
                schemas=schemas,
            )
            self._last_tool_schema_signature = tool_schema_signature(schemas)
            return LangChainModelAdapter(bound_model)

        self._last_tool_schema_signature = None
        return LangChainModelAdapter(model)

    def _mark_dirty(self) -> None:
        with self._compile_lock:
            self._dirty = True

    def _ensure_not_running_loop(self) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        if loop.is_running():
            raise SyncInAsyncContextError(
                "run/stream cannot be called inside an active event loop. Use arun/astream."
            )
