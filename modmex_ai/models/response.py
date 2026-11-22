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
    # Opaque provider payload that must be replayed verbatim (e.g. Anthropic
    # thinking blocks that accompany a tool_use turn). Adapters own its shape.
    provider_data: dict[str, Any] | None = None


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
    compaction_count: int = 0
    stop_reason: str | None = None

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)
