from __future__ import annotations

from dataclasses import field
from typing import Any

from modmex import BaseModel

from modmex_ai.agents import Agent, AgentResult
from modmex_ai.flows.continuation import FlowContinuation
from modmex_ai.models import ProviderState, Usage
from modmex_ai.sessions import SessionItem
from modmex_ai.tracing import Trace


class FlowResult(BaseModel):
    output: Any
    last_agent: Agent
    last_agent_name: str
    continuation: FlowContinuation
    agent_results: list[AgentResult]
    events: list[dict[str, Any]] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    provider_state: ProviderState | None = None
    input_items: list[SessionItem] = field(default_factory=list)
    output_items: list[SessionItem] = field(default_factory=list)
    trace: Trace = field(default_factory=Trace)

    def to_input_list(self) -> list[dict[str, Any]]:
        return [
            item.to_input()
            for item in [*self.input_items, *self.output_items]
        ]
