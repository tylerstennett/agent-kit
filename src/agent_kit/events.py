from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from .config import ToolResult


@dataclass(slots=True)
class RunStartEvent:
    type: Literal["run_start"] = "run_start"
    run_id: str = ""


@dataclass(slots=True)
class LLMTokenEvent:
    type: Literal["llm_token"] = "llm_token"
    run_id: str = ""
    token: str = ""


@dataclass(slots=True)
class ToolStartEvent:
    type: Literal["tool_start"] = "tool_start"
    run_id: str = ""
    tool_name: str = ""
    call_id: str = ""
    tool_args: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ToolEndEvent:
    type: Literal["tool_end"] = "tool_end"
    run_id: str = ""
    result: ToolResult | None = None


@dataclass(slots=True)
class StateUpdateEvent:
    type: Literal["state_update"] = "state_update"
    run_id: str = ""
    delta: dict[str, Any] = field(default_factory=dict)
    snapshot: dict[str, Any] | None = None


@dataclass(slots=True)
class RunEndEvent:
    type: Literal["run_end"] = "run_end"
    run_id: str = ""
    termination: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ErrorEvent:
    type: Literal["error"] = "error"
    run_id: str = ""
    message: str = ""


StreamEvent = (
    RunStartEvent
    | LLMTokenEvent
    | ToolStartEvent
    | ToolEndEvent
    | StateUpdateEvent
    | RunEndEvent
    | ErrorEvent
)
