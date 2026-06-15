from __future__ import annotations

from dataclasses import field
from enum import StrEnum
from typing import Any

from modmex import BaseModel

from modmex_ai.agents.result import AgentResult
from modmex_ai.models import ToolCall


class AgentStreamEventType(StrEnum):
    TEXT_DELTA = "text_delta"
    TOOL_CALL_DELTA = "tool_call_delta"
    TOOL_FINISHED = "tool_finished"
    HANDOFF = "handoff"
    COMPLETED = "completed"


class AgentStreamEvent(BaseModel):
    """An incremental event from an Agent, independent of its model provider."""

    type: AgentStreamEventType
    text_delta: str | None = None
    tool_call: ToolCall | None = None
    result: AgentResult | None = None
    data: dict[str, Any] = field(default_factory=dict)
