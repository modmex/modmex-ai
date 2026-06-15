from __future__ import annotations

from modmex import BaseModel

from modmex_ai.flows import FlowContinuation


class VoiceContinuation(BaseModel):
    """Portable voice state; persist it alongside the host's Session history."""

    agent_name: str
    flow_continuation: FlowContinuation

    @classmethod
    def from_flow(cls, continuation: FlowContinuation) -> "VoiceContinuation":
        return cls(
            agent_name=continuation.agent_name,
            flow_continuation=continuation,
        )
