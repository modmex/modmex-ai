from __future__ import annotations

from enum import Enum
from typing import Protocol

from modmex import BaseModel


class VoiceTurnOptions(BaseModel):
    """Retry policy for provider I/O within a single chained voice turn."""

    max_provider_retries: int = 0
    retry_delay_seconds: float = 0.0
    timeout_seconds: float | None = None


class VoiceTurnReason(str, Enum):
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"
    STT_FAILED = "stt_failed"
    AGENT_FAILED = "agent_failed"
    TTS_FAILED = "tts_failed"


class VoiceTurnMetrics(BaseModel):
    stt_latency_ms: int = 0
    agent_latency_ms: int = 0
    tts_latency_ms: int = 0
    total_latency_ms: int = 0
    provider_retry_count: int = 0


class VoiceTurnOutcome(BaseModel):
    reason: VoiceTurnReason
    metrics: VoiceTurnMetrics


class VoiceTurnObserver(Protocol):
    def on_voice_turn_finished(self, outcome: VoiceTurnOutcome) -> None:
        ...
