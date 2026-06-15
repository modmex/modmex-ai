from __future__ import annotations

from modmex import BaseModel

from modmex_ai.models import ProviderState


class FlowContinuation(BaseModel):
    """Portable state the host may use to continue a flow in a later run."""

    agent_name: str
    provider_state: ProviderState | None = None
