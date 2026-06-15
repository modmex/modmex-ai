from __future__ import annotations

from dataclasses import field
from typing import Any, Literal

from modmex import BaseModel


TraceStepType = Literal[
    "model_request",
    "model_response",
    "tool_call",
    "handoff_call",
    "handoff",
    "guardrail",
    "realtime_client_event",
    "realtime_server_event",
    "realtime_tool_call",
    "realtime_handoff",
    "realtime_response",
]


class TraceStep(BaseModel):
    type: TraceStepType
    name: str
    data: dict[str, Any] = field(default_factory=dict)
