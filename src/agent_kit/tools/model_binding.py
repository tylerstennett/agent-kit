from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel

from agent_kit.config import ToolSchemaSyncPolicy
from agent_kit.errors import ModelToolBindingError
from agent_kit.tools.base import BaseTool
from agent_kit.tools.schema import tools_to_model_schemas


def bind_model_tools(
    model: BaseChatModel,
    tools: list[BaseTool],
    policy: ToolSchemaSyncPolicy,
    *,
    schemas: list[dict[str, object]] | None = None,
) -> object:
    bind_method = getattr(model, "bind_tools", None)
    if bind_method is None or not callable(bind_method):
        raise ModelToolBindingError(
            "Model does not support tool binding. Set model_tool_binding_mode='manual' to bypass."
        )
    effective_schemas = schemas or tools_to_model_schemas(tools, policy=policy)
    try:
        bound_model = bind_method(effective_schemas)
    except Exception as exc:
        raise ModelToolBindingError(f"Model tool binding failed: {exc}") from exc
    if not hasattr(bound_model, "invoke") or not hasattr(bound_model, "ainvoke"):
        raise ModelToolBindingError("Bound model does not expose invoke/ainvoke.")
    return bound_model
