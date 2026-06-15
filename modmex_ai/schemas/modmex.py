from __future__ import annotations

import json
from typing import Any

from modmex import BaseModel

from modmex_ai.errors import OutputValidationError, ToolValidationError


def validate_model(value: Any, model_type: type[Any], *, error_type: type[Exception] = OutputValidationError) -> Any:
    try:
        if isinstance(value, str):
            value = json.loads(value)
        if isinstance(value, model_type):
            return value
        if isinstance(model_type, type) and issubclass(model_type, BaseModel):
            return model_type(**value)
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

