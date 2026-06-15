from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from modmex_ai.voice.event import VoiceSessionEvent, VoiceTerminationReason


class VoiceAgentSession(ABC):
    """Common live-session contract for speech-to-speech and chained voice runtimes."""

    @abstractmethod
    async def voice_events(self) -> AsyncIterator[VoiceSessionEvent]:
        """Yield provider-neutral voice events for the active conversation."""

    @abstractmethod
    async def close(
        self,
        reason: VoiceTerminationReason = VoiceTerminationReason.ENDED_BY_APPLICATION,
    ) -> None:
        """Release the live voice-session transport and resources."""
