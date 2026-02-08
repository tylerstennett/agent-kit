from __future__ import annotations

import json
from typing import Any
from uuid import uuid4

from conduit import Conduit, SyncConduit
from conduit.models.messages import (
    ChatResponse,
    ImageUrlPart,
    Message,
    RequestContext,
    Role,
    TextPart,
)
from conduit.tools.schema import ToolCall as ConduitToolCall
from conduit.tools.schema import ToolDefinition
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from agent_kit.config import InvocationConfig
from agent_kit.llm.types import ModelResponse, ToolCall

CONDUIT_CONTEXT_METADATA_KEY = "conduit_context_metadata"
CONDUIT_RUNTIME_OVERRIDES_KEY = "conduit_runtime_overrides"


def tool_schemas_to_conduit_tools(schemas: list[dict[str, object]]) -> list[ToolDefinition]:
    tool_definitions: list[ToolDefinition] = []
    for schema in schemas:
        name_value = schema.get("name")
        if not isinstance(name_value, str) or not name_value:
            raise TypeError("Tool schema must include a non-empty 'name'.")

        description_value = schema.get("description")
        description = description_value if isinstance(description_value, str) else ""

        parameters_value = schema.get("parameters")
        if isinstance(parameters_value, dict):
            parameters: dict[str, Any] = dict(parameters_value)
        else:
            parameters = {"type": "object", "properties": {}}

        tool_definitions.append(
            ToolDefinition(
                name=name_value,
                description=description,
                parameters=parameters,
            )
        )
    return tool_definitions


class ConduitModelAdapter:
    def __init__(
        self, model: Conduit | SyncConduit, *, tools: list[ToolDefinition] | None = None
    ) -> None:
        self._model = model
        self._tools = list(tools) if tools else None

    def complete(self, messages: list[BaseMessage], config: InvocationConfig) -> ModelResponse:
        if isinstance(self._model, Conduit):
            raise TypeError(
                "ConduitModelAdapter with Conduit is async-only. Use arun/ainvoke/astream, "
                "or provide SyncConduit for sync APIs."
            )

        config_overrides, context, runtime_overrides = self._resolve_request_options(config)
        response = self._model.chat(
            messages=self._to_conduit_messages(messages),
            tools=self._tools,
            tool_choice="auto" if self._tools else None,
            config_overrides=config_overrides,
            context=context,
            runtime_overrides=runtime_overrides,
        )
        return self._to_model_response(response)

    async def acomplete(
        self, messages: list[BaseMessage], config: InvocationConfig
    ) -> ModelResponse:
        if isinstance(self._model, SyncConduit):
            raise TypeError(
                "ConduitModelAdapter with SyncConduit is sync-only. Use run/invoke/stream, "
                "or provide Conduit for async APIs."
            )

        config_overrides, context, runtime_overrides = self._resolve_request_options(config)
        response = await self._model.chat(
            messages=self._to_conduit_messages(messages),
            tools=self._tools,
            tool_choice="auto" if self._tools else None,
            config_overrides=config_overrides,
            context=context,
            runtime_overrides=runtime_overrides,
        )
        return self._to_model_response(response)

    def _resolve_request_options(
        self, config: InvocationConfig
    ) -> tuple[dict[str, Any] | None, RequestContext | None, dict[str, Any] | None]:
        configurable = dict(config.configurable)

        context_metadata = self._ensure_dict_or_none(
            configurable.pop(CONDUIT_CONTEXT_METADATA_KEY, None),
            CONDUIT_CONTEXT_METADATA_KEY,
        )
        runtime_overrides = self._ensure_dict_or_none(
            configurable.pop(CONDUIT_RUNTIME_OVERRIDES_KEY, None),
            CONDUIT_RUNTIME_OVERRIDES_KEY,
        )

        request_context = self._build_request_context(config, context_metadata)
        config_overrides = configurable or None
        return config_overrides, request_context, runtime_overrides

    def _build_request_context(
        self,
        config: InvocationConfig,
        context_metadata: dict[str, Any] | None,
    ) -> RequestContext | None:
        has_context = (
            config.thread_id is not None
            or bool(config.tags)
            or bool(context_metadata)
        )
        if not has_context:
            return None
        return RequestContext(
            thread_id=config.thread_id,
            tags=list(config.tags),
            metadata=context_metadata or {},
        )

    def _ensure_dict_or_none(self, value: object, key: str) -> dict[str, Any] | None:
        if value is None:
            return None
        if isinstance(value, dict):
            return dict(value)
        raise TypeError(f"'{key}' in InvocationConfig.configurable must be a dictionary.")

    def _to_model_response(self, response: ChatResponse) -> ModelResponse:
        tool_calls = self._to_agent_tool_calls(response.tool_calls)
        content = response.content if isinstance(response.content, str) else ""
        message = AIMessage(
            content=content,
            tool_calls=[
                {
                    "id": call["id"],
                    "name": call["name"],
                    "args": call["args"],
                }
                for call in tool_calls
            ],
        )
        tokens = [content] if content else []
        return ModelResponse(message=message, tool_calls=tool_calls, tokens=tokens)

    def _to_agent_tool_calls(
        self,
        tool_calls: list[ConduitToolCall] | None,
    ) -> list[ToolCall]:
        if not tool_calls:
            return []

        parsed: list[ToolCall] = []
        for call in tool_calls:
            parsed.append(
                {
                    "id": call.id,
                    "name": call.name,
                    "args": dict(call.arguments),
                }
            )
        return parsed

    def _to_conduit_messages(self, messages: list[BaseMessage]) -> list[Message]:
        converted: list[Message] = []
        for message in messages:
            converted.append(self._to_conduit_message(message))
        return converted

    def _to_conduit_message(self, message: BaseMessage) -> Message:
        content = self._to_content_parts(message.content)

        if isinstance(message, SystemMessage):
            return Message(role=Role.SYSTEM, content=content)

        if isinstance(message, HumanMessage):
            return Message(role=Role.USER, content=content)

        if isinstance(message, AIMessage):
            return Message(
                role=Role.ASSISTANT,
                content=content,
                tool_calls=self._extract_assistant_tool_calls(message),
            )

        if isinstance(message, ToolMessage):
            tool_call_id = getattr(message, "tool_call_id", None)
            if not isinstance(tool_call_id, str) or not tool_call_id:
                tool_call_id = None
            tool_name = message.name if isinstance(message.name, str) and message.name else None
            return Message(
                role=Role.TOOL,
                content=content,
                tool_call_id=tool_call_id,
                name=tool_name,
            )

        role_value = getattr(message, "role", None)
        if role_value == "system":
            return Message(role=Role.SYSTEM, content=content)
        if role_value == "user":
            return Message(role=Role.USER, content=content)
        if role_value == "assistant":
            return Message(
                role=Role.ASSISTANT,
                content=content,
                tool_calls=self._extract_assistant_tool_calls(message),
            )
        if role_value == "tool":
            tool_call_id = getattr(message, "tool_call_id", None)
            if not isinstance(tool_call_id, str) or not tool_call_id:
                tool_call_id = None
            return Message(role=Role.TOOL, content=content, tool_call_id=tool_call_id)

        raise TypeError(f"Unsupported message type for ConduitModelAdapter: {type(message)!r}")

    def _to_content_parts(
        self, content: object
    ) -> list[TextPart | ImageUrlPart | dict[str, Any]] | None:
        if content is None:
            return None
        if isinstance(content, str):
            return [TextPart(text=content)]

        if isinstance(content, list):
            parts: list[TextPart | ImageUrlPart | dict[str, Any]] = []
            for item in content:
                converted = self._to_content_part(item)
                if converted is not None:
                    parts.append(converted)
            return parts or None

        raise TypeError("Conduit message content must be a string, list of parts, or None.")

    def _to_content_part(self, item: object) -> TextPart | ImageUrlPart | dict[str, Any] | None:
        if item is None:
            return None
        if isinstance(item, TextPart | ImageUrlPart):
            return item
        if isinstance(item, str):
            return TextPart(text=item)

        if isinstance(item, dict):
            item_type = item.get("type")
            if item_type == "text":
                text_value = item.get("text")
                if isinstance(text_value, str):
                    if len(item) == 2:
                        return TextPart(text=text_value)
                    return dict(item)
                raise TypeError("Text content part must include a string 'text' field.")

            if item_type == "image_url":
                image_payload = item.get("image_url")
                if isinstance(image_payload, dict):
                    url_value = image_payload.get("url")
                    if isinstance(url_value, str):
                        if len(item) == 2 and len(image_payload) == 1:
                            return ImageUrlPart(url=url_value)
                        return dict(item)
                raise TypeError("Image content part must include image_url.url as a string.")

            return dict(item)

        raise TypeError("Content parts must be strings, dictionaries, or typed content parts.")

    def _extract_assistant_tool_calls(self, message: BaseMessage) -> list[ConduitToolCall] | None:
        tool_calls_raw = getattr(message, "tool_calls", [])
        if not tool_calls_raw and isinstance(getattr(message, "additional_kwargs", None), dict):
            tool_calls_raw = message.additional_kwargs.get("tool_calls", [])

        parsed_calls: list[ConduitToolCall] = []
        for raw_call in tool_calls_raw:
            if not isinstance(raw_call, dict):
                continue

            function_payload = raw_call.get("function")
            if not isinstance(function_payload, dict):
                function_payload = {}

            call_id = raw_call.get("id")
            if not isinstance(call_id, str) or not call_id:
                call_id = f"call-{uuid4().hex[:8]}"

            name_value = raw_call.get("name")
            if not isinstance(name_value, str) or not name_value:
                function_name = function_payload.get("name")
                if isinstance(function_name, str) and function_name:
                    name_value = function_name

            if not isinstance(name_value, str) or not name_value:
                continue

            args_value = raw_call.get("args")
            if args_value is None:
                args_value = function_payload.get("arguments")
            if args_value is None:
                args_value = raw_call.get("arguments")

            parsed_calls.append(
                ConduitToolCall(
                    id=call_id,
                    name=name_value,
                    arguments=self._parse_tool_arguments(args_value),
                )
            )

        return parsed_calls or None

    def _parse_tool_arguments(self, arguments: object) -> dict[str, Any]:
        if isinstance(arguments, dict):
            return dict(arguments)
        if isinstance(arguments, str):
            try:
                parsed = json.loads(arguments)
            except json.JSONDecodeError:
                return {}
            if isinstance(parsed, dict):
                return parsed
            return {}
        return {}
