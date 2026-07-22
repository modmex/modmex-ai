from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import field
from enum import Enum
from typing import Any

from modmex import BaseModel


class ApprovalDecisionType(str, Enum):
    APPROVED = "approved"
    REJECTED = "rejected"


class ApprovalRequest(BaseModel):
    request_id: str
    tool_name: str
    arguments: dict[str, Any]
    reason: str | None = None


class ApprovalDecision(BaseModel):
    request_id: str
    decision: ApprovalDecisionType
    signature: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class ApprovalPolicy(ABC):
    """Verifies an external approval decision before a sensitive tool runs."""

    @abstractmethod
    def verify(self, request: ApprovalRequest, decision: ApprovalDecision) -> bool:
        """Return whether the supplied decision is valid for this request."""
