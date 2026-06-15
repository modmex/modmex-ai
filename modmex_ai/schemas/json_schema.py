from __future__ import annotations

import inspect
import types
from collections.abc import Sequence
from enum import Enum
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints

from modmex import BaseModel


def schema_for_type(annotation: Any) -> dict[str, Any]:
    if annotation is inspect.Signature.empty or annotation is Any:
        return {}

    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin in (Union, types.UnionType):
        non_none = [arg for arg in args if arg is not type(None)]
        if len(non_none) == 1 and len(non_none) != len(args):
            schema = schema_for_type(non_none[0])
            schema["nullable"] = True
            return schema
        return {"anyOf": [schema_for_type(arg) for arg in non_none]}

    if origin in (list, Sequence):
        return {
            "type": "array",
            "items": schema_for_type(args[0] if args else Any),
        }

    if origin is dict:
        return {"type": "object"}

    if get_origin(annotation) is Literal:
        values = list(args)
        return {"enum": values, "type": _json_type(type(values[0])) if values else "string"}

    if inspect.isclass(annotation) and issubclass(annotation, Enum):
        values = [item.value for item in annotation]
        return {"enum": values, "type": _json_type(type(values[0])) if values else "string"}

    if inspect.isclass(annotation) and _is_model_type(annotation):
        return schema_for_model(annotation)

    return {"type": _json_type(annotation)}


def schema_for_model(model_type: type[Any], *, name: str | None = None) -> dict[str, Any]:
    properties: dict[str, Any] = {}
    required: list[str] = []
    hints = get_type_hints(model_type)
    for field_name, annotation in hints.items():
        if field_name.startswith("_"):
            continue
        properties[field_name] = schema_for_type(annotation)
        if not _field_has_default(model_type, field_name) and not _accepts_none(annotation):
            required.append(field_name)

    return {
        "type": "object",
        "title": name or model_type.__name__,
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def function_schema(name: str, description: str, parameters: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": parameters,
            "required": list(parameters.keys()),
            "additionalProperties": False,
        },
        "strict": True,
    }


def _json_type(annotation: type[Any]) -> str:
    if annotation is str:
        return "string"
    if annotation is int:
        return "integer"
    if annotation is float:
        return "number"
    if annotation is bool:
        return "boolean"
    if annotation is list:
        return "array"
    if annotation is dict:
        return "object"
    return "string"


def _field_has_default(model_type: type[Any], field_name: str) -> bool:
    return hasattr(model_type, field_name)


def _accepts_none(annotation: Any) -> bool:
    origin = get_origin(annotation)
    return origin in (Union, types.UnionType) and any(arg is type(None) for arg in get_args(annotation))


def _is_model_type(annotation: type[Any]) -> bool:
    return issubclass(annotation, BaseModel) or hasattr(annotation, "model_dump") or hasattr(annotation, "to_dict")
