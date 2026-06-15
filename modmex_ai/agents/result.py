from __future__ import annotations

from dataclasses import field
from typing import Any

from modmex import BaseModel

from modmex_ai.models import ProviderState, Usage
from modmex_ai.sessions import SessionItem
from modmex_ai.tracing import Trace


class AgentResult(BaseModel):
    output: Any
    agent: str
    handoff_target: str | None = None
    handoff_name: str | None = None
    handoff_input: Any = None
    usage: Usage = field(default_factory=Usage)
    items: list[SessionItem] = field(default_factory=list)
    provider_state: ProviderState | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    trace: Trace = field(default_factory=Trace)
