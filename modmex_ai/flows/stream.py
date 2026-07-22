from __future__ import annotations

from enum import Enum

from modmex import BaseModel

from modmex_ai.agents import AgentStreamEvent
from modmex_ai.flows.result import FlowResult


class FlowStreamEventType(str, Enum):
    AGENT = "agent"
    COMPLETED = "completed"


class FlowStreamEvent(BaseModel):
    """A provider-neutral event emitted while a Flow is executing."""

    type: FlowStreamEventType
    agent_name: str | None = None
    agent_event: AgentStreamEvent | None = None
    result: FlowResult | None = None
