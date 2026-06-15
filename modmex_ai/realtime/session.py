from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any

from modmex_ai.agents import Agent, RunContext
from modmex_ai.errors import RealtimeProtocolError
from modmex_ai.models import Usage
from modmex_ai.realtime.event import RealtimeEvent
from modmex_ai.realtime.transport import RealtimeTransport
from modmex_ai.voice import VoiceAgentSession, VoiceTerminationReason


class RealtimeSession(VoiceAgentSession, ABC):
    """Provider-neutral event loop shared by realtime session adapters."""

    def __init__(
        self,
        *,
        agent: Agent,
        transport: RealtimeTransport,
        agents: list[Agent] | None = None,
        context: RunContext | Any = None,
    ) -> None:
        self.agent = agent
        self.current_agent = agent
        self.agents = {agent.name: agent for agent in [agent, *(agents or [])]}
        self.transport = transport
        self.context = (
            context
            if isinstance(context, RunContext)
            else RunContext(input=None, context=context)
        )
        self.usage = Usage()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        await self.close()

    async def close(
        self,
        reason: VoiceTerminationReason = VoiceTerminationReason.ENDED_BY_APPLICATION,
    ) -> None:
        await self.transport.close()

    async def send(self, event: dict[str, Any]) -> None:
        event_type = event.get("type", "unknown")
        self.context.trace.add("realtime_client_event", event_type)
        await self.transport.send(json.dumps(event, separators=(",", ":")))

    async def receive(self) -> RealtimeEvent:
        raw_message = await self.transport.recv()
        if isinstance(raw_message, bytes):
            raw_message = raw_message.decode("utf-8")
        try:
            raw = json.loads(raw_message)
        except (TypeError, json.JSONDecodeError) as error:
            raise RealtimeProtocolError("Realtime transport returned invalid JSON") from error
        if not isinstance(raw, dict) or not isinstance(raw.get("type"), str):
            raise RealtimeProtocolError("Realtime event must be an object with a type")
        event = RealtimeEvent(
            type=raw["type"],
            data={key: value for key, value in raw.items() if key != "type"},
            raw=raw,
        )
        self.context.trace.add("realtime_server_event", event.type)
        return event

    async def events(self) -> AsyncIterator[RealtimeEvent]:
        while True:
            event = await self.receive()
            await self.handle(event)
            yield event

    @abstractmethod
    async def handle(self, event: RealtimeEvent) -> None:
        """Process a provider event before exposing it to the caller."""
