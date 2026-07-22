from __future__ import annotations

import json
import types
from dataclasses import MISSING
from typing import Any, Union, get_args, get_origin

from modmex import BaseModel

from modmex_ai.errors import OutputValidationError, ToolValidationError


def validate_model(value: Any, model_type: type[Any], *, error_type: type[Exception] = OutputValidationError) -> Any:
    try:
        if isinstance(value, str):
            value = json.loads(value)
        if isinstance(value, model_type):
            return value
        if isinstance(model_type, type) and issubclass(model_type, BaseModel):
            return model_type(**_normalize_null_defaults(value, model_type))
        return model_type(value)
    except Exception as exc:
        raise error_type(str(exc)) from exc


def validate_tool_args(value: Any, model_type: type[Any]) -> Any:
    return validate_model(value, model_type, error_type=ToolValidationError)


def serialize(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return value


def dumps(value: Any) -> str:
    return json.dumps(serialize(value), separators=(",", ":"), default=serialize)


def _normalize_null_defaults(value: Any, model_type: type[BaseModel]) -> Any:
    """Let a strict-schema null fall back to a non-nullable model default."""
    if not isinstance(value, dict):
        return value
    normalized = dict(value)
    for field in getattr(model_type, "__modmex_fields__", ()):
        if field.name not in normalized:
            continue
        field_value = normalized[field.name]
        if (
            field_value is None
            and _field_has_default(field)
            and not _accepts_none(field.type)
        ):
            normalized.pop(field.name)
            continue
        normalized[field.name] = _normalize_nested_model(field_value, field.type)
    return normalized


def _normalize_nested_model(value: Any, annotation: Any) -> Any:
    if value is None:
        return None
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin in (Union, types.UnionType):
        for candidate in args:
            if candidate is not type(None) and _is_modmex_model(candidate):
                return _normalize_null_defaults(value, candidate)
        return value
    if origin is list and args:
        return [_normalize_nested_model(item, args[0]) for item in value]
    if origin is dict and len(args) == 2:
        return {
            key: _normalize_nested_model(item, args[1])
            for key, item in value.items()
        }
    if _is_modmex_model(annotation):
        return _normalize_null_defaults(value, annotation)
    return value


def _field_has_default(field: Any) -> bool:
    return field.default is not MISSING or field.default_factory is not MISSING


def _accepts_none(annotation: Any) -> bool:
    origin = get_origin(annotation)
    return origin in (Union, types.UnionType) and type(None) in get_args(annotation)


def _is_modmex_model(value: Any) -> bool:
    return isinstance(value, type) and issubclass(value, BaseModel)
