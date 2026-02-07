from __future__ import annotations

import inspect
import time
import types
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Union, get_args, get_origin, get_type_hints
from uuid import uuid4

from agent_kit.config import ToolDecision, ToolError, ToolResult
from agent_kit.errors import ToolValidationError
from agent_kit.state import AgentState


@dataclass(slots=True)
class ArgumentSpec:
    annotation: Any
    required: bool


def _is_any(annotation: Any) -> bool:
    return annotation is Any or annotation is inspect.Signature.empty


def _matches_type(value: Any, annotation: Any) -> bool:
    if _is_any(annotation):
        return True
    origin = get_origin(annotation)
    if origin is None:
        if isinstance(annotation, type):
            return isinstance(value, annotation)
        return True
    if origin in (list, tuple, set):
        return isinstance(value, origin)
    if origin is dict:
        return isinstance(value, dict)
    if origin is Callable:
        return callable(value)
    if origin in (Union, types.UnionType):
        return any(_matches_type(value, option) for option in get_args(annotation))
    return True


def _annotation_label(annotation: object) -> str:
    if annotation is inspect.Signature.empty:
        return "Any"
    if hasattr(annotation, "__name__"):
        return str(annotation.__name__)
    return str(annotation)


def infer_argument_specs(func: Callable[..., Any]) -> dict[str, ArgumentSpec]:
    signature = inspect.signature(func)
    try:
        resolved_hints = get_type_hints(func)
    except Exception:
        resolved_hints = {}
    specs: dict[str, ArgumentSpec] = {}
    for name, param in signature.parameters.items():
        if name in {"self", "state"}:
            continue
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        specs[name] = ArgumentSpec(
            annotation=resolved_hints.get(name, param.annotation),
            required=param.default is inspect.Signature.empty,
        )
    return specs


def validate_args(
    specs: dict[str, ArgumentSpec],
    args: dict[str, Any],
    *,
    allow_extra: bool = False,
) -> None:
    unknown = [key for key in args if key not in specs]
    if unknown and not allow_extra:
        unknown_sorted = sorted(unknown)
        raise ToolValidationError(
            f"Unexpected arguments: {', '.join(unknown_sorted)}",
            details={
                "kind": "unexpected_args",
                "unexpected": unknown_sorted,
            },
        )
    missing = [name for name, spec in specs.items() if spec.required and name not in args]
    if missing:
        missing_sorted = sorted(missing)
        raise ToolValidationError(
            f"Missing required arguments: {', '.join(missing_sorted)}",
            details={
                "kind": "missing_args",
                "missing": missing_sorted,
            },
        )
    for name, value in args.items():
        if allow_extra and name not in specs:
            continue
        spec = specs[name]
        if not _matches_type(value, spec.annotation):
            raise ToolValidationError(
                f"Invalid argument type for '{name}'",
                details={
                    "kind": "invalid_type",
                    "arg_name": name,
                    "expected": _annotation_label(spec.annotation),
                    "actual_type": type(value).__name__,
                },
            )


class BaseTool(ABC):
    def __init__(
        self,
        *,
        name: str | None = None,
        description: str = "",
        arg_specs: dict[str, ArgumentSpec] | None = None,
        allow_extra_args: bool | None = None,
    ) -> None:
        self.name = name or self.__class__.__name__.replace("Tool", "").lower()
        self.description = description
        self.arg_specs = arg_specs or infer_argument_specs(self.execute)
        self.allow_extra_args = (
            allow_extra_args if allow_extra_args is not None else _accepts_var_kwargs(self.execute)
        )

    def pre_execute(self, state: AgentState) -> ToolDecision:
        return ToolDecision(action="continue")

    async def apre_execute(self, state: AgentState) -> ToolDecision:
        return self.pre_execute(state)

    @abstractmethod
    def execute(self, state: AgentState, **kwargs: Any) -> Any:
        pass

    async def aexecute(self, state: AgentState, **kwargs: Any) -> Any:
        return self.execute(state, **kwargs)

    def post_execute(self, state: AgentState, result: ToolResult) -> tuple[AgentState, ToolResult]:
        return state, result

    async def apost_execute(
        self, state: AgentState, result: ToolResult
    ) -> tuple[AgentState, ToolResult]:
        return self.post_execute(state, result)

    def normalize_success(
        self, call_id: str, output: Any, started_at: float, ended_at: float
    ) -> ToolResult:
        if isinstance(output, ToolResult):
            result = output
            if not result.tool_name:
                result.tool_name = self.name
            if not result.call_id:
                result.call_id = call_id
            if result.started_at is None:
                result.started_at = started_at
            if result.ended_at is None:
                result.ended_at = ended_at
            if result.duration_ms is None:
                result.duration_ms = (ended_at - started_at) * 1000
            return result
        return ToolResult(
            tool_name=self.name,
            call_id=call_id,
            status="success",
            data=output,
            started_at=started_at,
            ended_at=ended_at,
            duration_ms=(ended_at - started_at) * 1000,
        )

    def normalize_failure(
        self,
        call_id: str,
        error: Exception,
        started_at: float,
        *,
        retriable: bool = False,
        code: str = "tool_error",
    ) -> ToolResult:
        ended_at = time.perf_counter()
        return ToolResult(
            tool_name=self.name,
            call_id=call_id,
            status="failed",
            error=ToolError(code=code, message=str(error), retriable=retriable),
            started_at=started_at,
            ended_at=ended_at,
            duration_ms=(ended_at - started_at) * 1000,
        )

    def next_call_id(self) -> str:
        return f"call-{uuid4().hex[:8]}"


def _accepts_var_kwargs(func: Callable[..., Any]) -> bool:
    signature = inspect.signature(func)
    return any(
        param.kind is inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()
    )
