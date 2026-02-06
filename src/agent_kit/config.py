from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

if TYPE_CHECKING:
    from .middleware import Middleware
    from .state import AgentState

    AgentStateLike: TypeAlias = AgentState
else:
    AgentStateLike: TypeAlias = dict[str, Any]

ToolStatus = Literal["success", "failed", "skipped"]
ExecutionMode = Literal["sequential", "parallel"]


@dataclass(slots=True)
class ToolError:
    code: str
    message: str
    retriable: bool = False
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ToolResult:
    tool_name: str
    call_id: str
    status: ToolStatus
    data: Any = None
    error: ToolError | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    started_at: float | None = None
    ended_at: float | None = None
    duration_ms: float | None = None


@dataclass(slots=True)
class ToolDecision:
    action: Literal["continue", "skip"] = "continue"
    reason: str | None = None
    result_override: ToolResult | None = None


BeforeToolHook = Callable[
    [AgentStateLike, str, dict[str, Any]],
    AgentStateLike | Awaitable[AgentStateLike],
]
AfterToolHook = Callable[
    [AgentStateLike, str, ToolResult],
    tuple[AgentStateLike, ToolResult] | Awaitable[tuple[AgentStateLike, ToolResult]],
]


@dataclass(slots=True)
class InvocationConfig:
    execution_mode: ExecutionMode | None = None
    recursion_limit: int | None = None
    max_steps: int | None = None
    include_state_snapshots: bool = False
    thread_id: str | None = None
    configurable: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    max_parallel_workers: int | None = None
    emit_llm_tokens: bool = True


@dataclass(slots=True)
class InvocationRequest:
    input_text: str | None = None
    messages: list[Any] = field(default_factory=list)
    system_prompt: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    state_patch: dict[str, Any] = field(default_factory=dict)
    config: InvocationConfig = field(default_factory=InvocationConfig)


@dataclass(slots=True)
class GraphBuildConfig:
    max_steps: int = 25
    default_execution_mode: ExecutionMode = "sequential"
    max_parallel_workers: int = 4
    checkpointer: Any | None = None
    model_config_overrides: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AgentConfig:
    default_metadata: dict[str, Any] = field(default_factory=dict)
    graph_config: GraphBuildConfig = field(default_factory=GraphBuildConfig)
    before_tool_hooks: list[BeforeToolHook] = field(default_factory=list)
    after_tool_hooks: list[AfterToolHook] = field(default_factory=list)
    middlewares: list[Middleware] = field(default_factory=list)
