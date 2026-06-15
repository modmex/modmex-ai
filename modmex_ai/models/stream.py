from __future__ import annotations

from enum import StrEnum
from typing import Any

from modmex import BaseModel

from modmex_ai.models.response import ModelResponse, ToolCall


class ModelStreamEventType(StrEnum):
    TEXT_DELTA = "text_delta"
    TOOL_CALL_DELTA = "tool_call_delta"
    COMPLETED = "completed"


class ModelStreamEvent(BaseModel):
    """Provider-neutral incremental output from a model request."""

    type: ModelStreamEventType
    text_delta: str | None = None
    tool_call: ToolCall | None = None
    response: ModelResponse | None = None
    raw: Any = None

    @classmethod
    def completed(cls, response: ModelResponse) -> "ModelStreamEvent":
        return cls(type=ModelStreamEventType.COMPLETED, response=response)
