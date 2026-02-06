from __future__ import annotations

import hashlib
import inspect
import json
import logging
import types
from typing import Any, Literal, Union, get_args, get_origin

from ..config import ToolSchemaSyncPolicy
from ..errors import ToolSchemaConversionError
from .base import BaseTool

logger = logging.getLogger(__name__)

JsonType = Literal["string", "integer", "number", "boolean", "null", "object", "array"]


def _is_untyped(annotation: object) -> bool:
    return annotation is Any or annotation is inspect.Signature.empty


def _unsupported_annotation(
    annotation: object, *, policy: ToolSchemaSyncPolicy, context: str
) -> dict[str, object]:
    message = f"Unsupported annotation for {context}: {annotation!r}"
    if policy == "strict":
        raise ToolSchemaConversionError(message)
    if policy == "warn":
        logger.warning(message)
    return {}


def _json_type_for_annotation(annotation: object) -> JsonType | None:
    if annotation is str:
        return "string"
    if annotation is bool:
        return "boolean"
    if annotation is int:
        return "integer"
    if annotation is float:
        return "number"
    if annotation is type(None):
        return "null"
    return None


def _json_type_for_literal_value(value: object) -> JsonType | None:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if value is None:
        return "null"
    return None


def _union_schema(annotation: object, *, policy: ToolSchemaSyncPolicy) -> dict[str, object]:
    options = get_args(annotation)
    if not options:
        return {}
    option_schemas: list[dict[str, object]] = []
    for option in options:
        option_schema = annotation_to_json_schema(option, policy=policy)
        option_schemas.append(option_schema or {})
    if len(option_schemas) == 1:
        return option_schemas[0]
    return {"anyOf": option_schemas}


def _sequence_schema(
    annotation: object, origin: object, *, policy: ToolSchemaSyncPolicy
) -> dict[str, object]:
    args = get_args(annotation)
    item_annotation: object | None = None
    if args:
        if origin is tuple:
            if len(args) == 1 or (len(args) == 2 and args[1] is Ellipsis):
                item_annotation = args[0]
            elif len(args) > 1:
                first = args[0]
                if all(option == first for option in args):
                    item_annotation = first
                else:
                    return _unsupported_annotation(
                        annotation,
                        policy=policy,
                        context="tuple annotation",
                    )
        else:
            item_annotation = args[0]

    schema: dict[str, object] = {"type": "array"}
    if origin is set:
        schema["uniqueItems"] = True
    if item_annotation is not None:
        item_schema = annotation_to_json_schema(item_annotation, policy=policy)
        schema["items"] = item_schema or {}
    return schema


def _dict_schema(annotation: object, *, policy: ToolSchemaSyncPolicy) -> dict[str, object]:
    args = get_args(annotation)
    schema: dict[str, object] = {"type": "object"}
    if not args:
        return schema
    if len(args) != 2:
        return _unsupported_annotation(annotation, policy=policy, context="dict annotation")
    key_annotation, value_annotation = args
    if key_annotation not in (str, Any, inspect.Signature.empty):
        return _unsupported_annotation(
            annotation,
            policy=policy,
            context="dict key annotation",
        )
    value_schema = annotation_to_json_schema(value_annotation, policy=policy)
    schema["additionalProperties"] = value_schema or {}
    return schema


def _literal_schema(annotation: object, *, policy: ToolSchemaSyncPolicy) -> dict[str, object]:
    values = list(get_args(annotation))
    if not values:
        return _unsupported_annotation(annotation, policy=policy, context="Literal annotation")
    enum_types: set[JsonType] = set()
    for value in values:
        value_type = _json_type_for_literal_value(value)
        if value_type is None:
            return _unsupported_annotation(
                annotation,
                policy=policy,
                context="Literal value",
            )
        enum_types.add(value_type)
    schema: dict[str, object] = {"enum": values}
    if len(enum_types) == 1:
        schema["type"] = next(iter(enum_types))
    return schema


def annotation_to_json_schema(
    annotation: object, *, policy: ToolSchemaSyncPolicy
) -> dict[str, object]:
    if _is_untyped(annotation):
        return {}

    primitive_type = _json_type_for_annotation(annotation)
    if primitive_type is not None:
        return {"type": primitive_type}

    origin = get_origin(annotation)
    if origin is None:
        return _unsupported_annotation(annotation, policy=policy, context="annotation")

    if origin in (list, set, tuple):
        return _sequence_schema(annotation, origin, policy=policy)
    if origin is dict:
        return _dict_schema(annotation, policy=policy)
    if origin in (Union, types.UnionType):
        return _union_schema(annotation, policy=policy)
    if origin is Literal:
        return _literal_schema(annotation, policy=policy)

    return _unsupported_annotation(annotation, policy=policy, context="annotation origin")


def tool_to_model_schema(tool: BaseTool, *, policy: ToolSchemaSyncPolicy) -> dict[str, object]:
    properties: dict[str, object] = {}
    required: list[str] = []
    for name, spec in tool.arg_specs.items():
        try:
            properties[name] = annotation_to_json_schema(spec.annotation, policy=policy)
        except ToolSchemaConversionError as exc:
            raise ToolSchemaConversionError(
                f"Unsupported annotation for tool '{tool.name}' argument '{name}': "
                f"{spec.annotation!r}"
            ) from exc
        if spec.required:
            required.append(name)

    parameters: dict[str, object] = {
        "type": "object",
        "properties": properties,
        "additionalProperties": bool(tool.allow_extra_args),
    }
    if required:
        parameters["required"] = required

    return {
        "name": tool.name,
        "description": tool.description,
        "parameters": parameters,
    }


def tools_to_model_schemas(
    tools: list[BaseTool], *, policy: ToolSchemaSyncPolicy
) -> list[dict[str, object]]:
    return [tool_to_model_schema(tool, policy=policy) for tool in tools]


def tool_schema_signature(schemas: list[dict[str, object]]) -> str:
    payload = json.dumps(schemas, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
