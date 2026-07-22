from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import field
from enum import Enum
from typing import Any

from modmex import BaseModel

from modmex_ai.approvals import ApprovalRequest
from modmex_ai.flows.continuation import FlowContinuation
from modmex_ai.models import ProviderState, ToolCall
from modmex_ai.sessions import SessionSnapshot


class FlowStateStatus(str, Enum):
    ACTIVE = "active"
    SUSPENDED = "suspended"
    COMPLETED = "completed"
    FAILED = "failed"


class FlowSuspension(BaseModel):
    """Portable checkpoint for a Flow paused before a side effect."""

    approval_request: ApprovalRequest
    pending_tool_call: ToolCall
    active_agent_name: str
    provider_state: ProviderState | None = None
    continuation: FlowContinuation | None = None


class PersistedFlowState(BaseModel):
    """Durable execution checkpoint owned by a host-provided FlowStateStore."""

    flow_instance_id: str
    revision: int = 0
    status: FlowStateStatus = FlowStateStatus.ACTIVE
    idempotency_key: str | None = None
    session_snapshot: SessionSnapshot | None = None
    continuation: FlowContinuation | None = None
    provider_state: ProviderState | None = None
    suspension: FlowSuspension | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    ttl: int | None = None


class FlowStateConflictError(RuntimeError):
    """The flow state changed since the caller loaded its revision."""


class FlowSuspended(RuntimeError):
    """A Flow stopped at an approval boundary and can be resumed from its state."""

    def __init__(self, state: PersistedFlowState) -> None:
        super().__init__(f"Flow {state.flow_instance_id!r} is suspended for approval")
        self.state = state


class FlowStateStore(ABC):
    """Persists Flow checkpoints with optimistic concurrency and idempotency."""

    @abstractmethod
    def load(self, flow_instance_id: str) -> PersistedFlowState | None:
        ...

    @abstractmethod
    def save(self, state: PersistedFlowState, *, expected_revision: int) -> PersistedFlowState:
        ...


class InMemoryFlowStateStore(FlowStateStore):
    def __init__(self) -> None:
        self._states: dict[str, PersistedFlowState] = {}

    def load(self, flow_instance_id: str) -> PersistedFlowState | None:
        return self._states.get(flow_instance_id)

    def save(self, state: PersistedFlowState, *, expected_revision: int) -> PersistedFlowState:
        current = self._states.get(state.flow_instance_id)
        revision = current.revision if current else 0
        if revision != expected_revision:
            raise FlowStateConflictError(f"Flow {state.flow_instance_id!r} revision conflict")
        stored = PersistedFlowState(**{**state.model_dump(), "revision": revision + 1})
        self._states[state.flow_instance_id] = stored
        return stored
