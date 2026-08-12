from __future__ import annotations

from dataclasses import field
from enum import Enum
from typing import Any

from modmex import BaseModel

from modmex_ai.agents.result import AgentResult
from modmex_ai.models import ToolCall


class MCPProgressEvent(BaseModel):
    """Typed envelope for progress emitted by a remote MCP tool."""

    data: dict[str, Any] = field(default_factory=dict)


class AgentStreamEventType(str, Enum):
    TEXT_DELTA = "text_delta"
    TOOL_CALL_DELTA = "tool_call_delta"
    TOOL_FINISHED = "tool_finished"
    MCP_PROGRESS = "mcp_progress"
    HANDOFF = "handoff"
    COMPLETED = "completed"


class AgentStreamEvent(BaseModel):
    """An incremental event from an Agent, independent of its model provider."""

    type: AgentStreamEventType
    text_delta: str | None = None
    tool_call: ToolCall | None = None
    result: AgentResult | None = None
    mcp_progress: MCPProgressEvent | None = None
    data: dict[str, Any] = field(default_factory=dict)
