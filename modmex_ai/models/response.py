from __future__ import annotations

from dataclasses import field
from typing import Any

from modmex import BaseModel

from modmex_ai.models.provider_state import ProviderState
from modmex_ai.models.usage import Usage


class ToolCall(BaseModel):
    tool_call_id: str
    name: str
    arguments: dict[str, Any] | str


class ModelResponse(BaseModel):
    output_text: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw: Any = None
    usage: Usage = field(default_factory=Usage)
    request_id: str | None = None
    status_code: int | None = None
    headers: dict[str, str] = field(default_factory=dict)
    provider: str | None = None
    model: str | None = None
    provider_state: ProviderState | None = None
    latency_ms: float | None = None

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)
