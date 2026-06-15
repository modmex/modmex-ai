from __future__ import annotations

from dataclasses import field
from enum import StrEnum
from typing import Any

from modmex import BaseModel


class VoiceSessionEventType(StrEnum):
    SESSION_STARTED = "session_started"
    TRANSCRIPT_DELTA = "transcript_delta"
    TRANSCRIPT_FINAL = "transcript_final"
    ASSISTANT_TRANSCRIPT_FINAL = "assistant_transcript_final"
    TOOL_FINISHED = "tool_finished"
    HANDOFF_COMPLETED = "handoff_completed"
    RESPONSE_COMPLETED = "response_completed"
    SESSION_ENDED = "session_ended"
    ERROR = "error"


class VoiceTerminationReason(StrEnum):
    COMPLETED = "completed"
    ENDED_BY_CALLER = "ended_by_caller"
    ENDED_BY_APPLICATION = "ended_by_application"
    PROVIDER_DISCONNECTED = "provider_disconnected"
    PROVIDER_ERROR = "provider_error"
    TOOL_ERROR = "tool_error"


class VoiceSessionEvent(BaseModel):
    """A provider-neutral event from a live voice-agent conversation."""

    type: VoiceSessionEventType
    data: dict[str, Any] = field(default_factory=dict)
    provider_event_type: str | None = None
    raw: dict[str, Any] | None = None
