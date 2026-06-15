from __future__ import annotations

import json
import os
from dataclasses import field
from typing import Any
from urllib.parse import urlencode

from modmex import BaseModel

from modmex_ai.agents import Agent, RunContext
from modmex_ai.agents.handoff import find_handoff
from modmex_ai.errors import (
    GuardrailTriggered,
    RealtimeConnectionError,
    RealtimeProviderError,
    ToolExecutionError,
    ToolValidationError,
    UnknownToolError,
)
from modmex_ai.models import ToolCall, Usage
from modmex_ai.realtime import RealtimeEvent, RealtimeSession, RealtimeTransport
from modmex_ai.schemas import dumps
from modmex_ai.tools import ToolExecutor
from modmex_ai.voice import (
    VoiceSessionEvent,
    VoiceSessionEventType,
    VoiceTerminationReason,
)


class OpenAIRealtimeSessionConfig(BaseModel):
    """OpenAI Realtime session fields shared by WebSocket and SIP calls."""

    model: str = "gpt-realtime-2.1"
    voice: str | None = None
    output_modalities: list[str] = field(default_factory=lambda: ["audio"])
    turn_detection: dict[str, Any] | None = None
    tool_choice: str = "auto"
    extra: dict[str, Any] = field(default_factory=dict)

    def to_accept_payload(self, agent: Agent) -> dict[str, Any]:
        """Build the Realtime Calls accept payload from the active agent."""
        return self._session_payload(agent, include_model=True)

    def to_session_update(self, agent: Agent, *, include_model: bool) -> dict[str, Any]:
        """Build the session body for a live WebSocket update."""
        return self._session_payload(agent, include_model=include_model)

    def _session_payload(self, agent: Agent, *, include_model: bool) -> dict[str, Any]:
        executor = ToolExecutor(agent.tools)
        tool_schemas = [*executor.schemas(), *agent._handoff_schemas()]
        session: dict[str, Any] = {
            "type": "realtime",
            "instructions": agent.instructions,
            "output_modalities": self.output_modalities,
            "tools": [{"type": "function", **tool} for tool in tool_schemas],
            "tool_choice": self.tool_choice,
            **self.extra,
        }
        if include_model:
            session["model"] = self.model
        audio: dict[str, Any] = {}
        if self.voice is not None:
            audio["output"] = {"voice": self.voice}
        if self.turn_detection is not None:
            audio["input"] = {"turn_detection": self.turn_detection}
        if audio:
            session["audio"] = audio
        return session


class OpenAIRealtimeSession(RealtimeSession):
    """Runs a Modmex-AI Agent over an OpenAI Realtime WebSocket."""

    def __init__(
        self,
        *,
        agent: Agent,
        transport: RealtimeTransport,
        config: OpenAIRealtimeSessionConfig | None = None,
        agents: list[Agent] | None = None,
        context: RunContext | Any = None,
        include_model_on_update: bool = True,
    ) -> None:
        super().__init__(
            agent=agent,
            transport=transport,
            agents=agents,
            context=context,
        )
        self.config = config or OpenAIRealtimeSessionConfig()
        self.include_model_on_update = include_model_on_update
        self._handled_tool_call_ids: set[str] = set()
        self._voice_event_queue: list[VoiceSessionEvent] = []
        self._termination_reason: VoiceTerminationReason | None = None

    async def configure(self) -> None:
        """Publish the active agent's instructions and tool surface."""
        await self.send({"type": "session.update", "session": self._session_payload()})

    async def create_response(self, *, instructions: str | None = None) -> None:
        response: dict[str, Any] = {}
        if instructions is not None:
            response["instructions"] = instructions
        await self.send({"type": "response.create", "response": response})

    async def send_text(self, text: str) -> None:
        await self.send({
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": text}],
            },
        })
        await self.create_response()

    async def append_audio(self, audio: str) -> None:
        """Append a base64-encoded audio chunk for a WebSocket media pipeline."""
        await self.send({"type": "input_audio_buffer.append", "audio": audio})

    async def commit_audio(self) -> None:
        """Commit buffered audio when server VAD is disabled."""
        await self.send({"type": "input_audio_buffer.commit"})
        await self.create_response()

    async def voice_events(self):
        """Yield normalized events while preserving raw events through ``events()``."""
        try:
            async for event in self.events():
                while self._voice_event_queue:
                    yield self._voice_event_queue.pop(0)
                voice_event = self._voice_event_from(event)
                if voice_event is not None:
                    yield voice_event
        except RealtimeProviderError as error:
            yield self._error_event(VoiceTerminationReason.PROVIDER_ERROR, error)
            yield self._ended_event(VoiceTerminationReason.PROVIDER_ERROR)
        except (
            ToolExecutionError,
            ToolValidationError,
            UnknownToolError,
            GuardrailTriggered,
        ) as error:
            yield self._error_event(VoiceTerminationReason.TOOL_ERROR, error)
            yield self._ended_event(VoiceTerminationReason.TOOL_ERROR)
        except Exception as error:
            reason = self._termination_reason or VoiceTerminationReason.PROVIDER_DISCONNECTED
            if reason == VoiceTerminationReason.ENDED_BY_APPLICATION:
                yield self._ended_event(reason)
                return
            yield self._error_event(reason, error)
            yield self._ended_event(reason)

    async def close(
        self,
        reason: VoiceTerminationReason = VoiceTerminationReason.ENDED_BY_APPLICATION,
    ) -> None:
        self._termination_reason = reason
        await super().close(reason)

    async def handle(self, event: RealtimeEvent) -> None:
        if event.type == "error":
            error = event.data.get("error", event.data)
            message = (
                error.get("message", "OpenAI Realtime returned an error")
                if isinstance(error, dict)
                else str(error)
            )
            raise RealtimeProviderError(message)
        if event.type in {
            "response.function_call_arguments.done",
            "response.output_item.done",
        }:
            tool_call = self._function_call(event)
            if tool_call is not None and tool_call.tool_call_id not in self._handled_tool_call_ids:
                self._handled_tool_call_ids.add(tool_call.tool_call_id)
                await self._execute_function_call(tool_call)
        if event.type == "response.done":
            response = event.data.get("response", {})
            self.usage.add(_usage_from_response(response))
            self.context.trace.add("realtime_response", self.current_agent.name)

    def _session_payload(self) -> dict[str, Any]:
        return self.config.to_session_update(
            self.current_agent,
            include_model=self.include_model_on_update,
        )

    def _function_call(self, event: RealtimeEvent) -> ToolCall | None:
        if event.type == "response.function_call_arguments.done":
            data = event.data
        else:
            item = event.data.get("item")
            if not isinstance(item, dict) or item.get("type") != "function_call":
                return None
            data = item
        tool_call_id = data.get("call_id")
        name = data.get("name")
        arguments = data.get("arguments", "{}")
        if not isinstance(tool_call_id, str) or not isinstance(name, str):
            return None
        return ToolCall(tool_call_id=tool_call_id, name=name, arguments=arguments)

    async def _execute_function_call(self, tool_call: ToolCall) -> None:
        handoff = find_handoff(tool_call, self.current_agent.handoffs)
        if handoff is not None:
            arguments = (
                json.loads(tool_call.arguments or "{}")
                if isinstance(tool_call.arguments, str)
                else tool_call.arguments
            )
            handoff_input = handoff.invoke(self.context, arguments)
            next_agent = self.agents.get(handoff.agent)
            if next_agent is None:
                raise RealtimeProviderError(
                    f"Handoff {handoff.name!r} targets unknown agent {handoff.agent!r}"
                )
            self.current_agent = next_agent
            self.context.trace.add(
                "realtime_handoff",
                handoff.name,
                to=next_agent.name,
            )
            self._voice_event_queue.append(VoiceSessionEvent(
                type=VoiceSessionEventType.HANDOFF_COMPLETED,
                data={
                    "handoff_name": handoff.name,
                    "agent_name": next_agent.name,
                },
            ))
            await self.configure()
            output: Any = {"transferred": True, "handoff_input": handoff_input}
        else:
            result = await ToolExecutor(self.current_agent.tools).execute_async(
                tool_call,
                context=self.context,
            )
            self.context.trace.add(
                "realtime_tool_call",
                tool_call.name,
                tool_call_id=tool_call.tool_call_id,
            )
            self._voice_event_queue.append(VoiceSessionEvent(
                type=VoiceSessionEventType.TOOL_FINISHED,
                data={
                    "tool_call_id": tool_call.tool_call_id,
                    "tool_name": tool_call.name,
                },
            ))
            output = result.output
        await self.send({
            "type": "conversation.item.create",
            "item": {
                "type": "function_call_output",
                "call_id": tool_call.tool_call_id,
                "output": dumps(output),
            },
        })
        await self.create_response()

    def _voice_event_from(self, event: RealtimeEvent) -> VoiceSessionEvent | None:
        event_type = event.type
        if event_type == "session.created":
            return self._voice_event(VoiceSessionEventType.SESSION_STARTED, event)
        if event_type == "conversation.item.input_audio_transcription.completed":
            return self._voice_event(
                VoiceSessionEventType.TRANSCRIPT_FINAL,
                event,
                transcript=event.data.get("transcript"),
                item_id=event.data.get("item_id"),
            )
        if event_type == "response.output_audio_transcript.done":
            return self._voice_event(
                VoiceSessionEventType.ASSISTANT_TRANSCRIPT_FINAL,
                event,
                transcript=event.data.get("transcript"),
                item_id=event.data.get("item_id"),
            )
        if event_type == "response.done":
            return self._voice_event(
                VoiceSessionEventType.RESPONSE_COMPLETED,
                event,
                usage=event.data.get("response", {}).get("usage"),
            )
        if event_type == "session.ended":
            reason = VoiceTerminationReason.ENDED_BY_CALLER
            self._termination_reason = reason
            return self._voice_event(
                VoiceSessionEventType.SESSION_ENDED,
                event,
                reason=reason,
            )
        return None

    @staticmethod
    def _error_event(
        reason: VoiceTerminationReason,
        error: Exception,
    ) -> VoiceSessionEvent:
        return VoiceSessionEvent(
            type=VoiceSessionEventType.ERROR,
            data={"reason": reason, "message": str(error)},
        )

    @staticmethod
    def _ended_event(reason: VoiceTerminationReason) -> VoiceSessionEvent:
        return VoiceSessionEvent(
            type=VoiceSessionEventType.SESSION_ENDED,
            data={"reason": reason},
        )

    @staticmethod
    def _voice_event(
        type: VoiceSessionEventType,
        event: RealtimeEvent,
        **data: Any,
    ) -> VoiceSessionEvent:
        return VoiceSessionEvent(
            type=type,
            data={key: value for key, value in data.items() if value is not None},
            provider_event_type=event.type,
            raw=event.raw,
        )


class OpenAIRealtimeClient:
    """Creates OpenAI Realtime WebSocket sessions without importing an SDK."""

    websocket_url = "wss://api.openai.com/v1/realtime"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        websocket_url: str | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.websocket_url = websocket_url or self.websocket_url

    async def connect(
        self,
        *,
        agent: Agent,
        config: OpenAIRealtimeSessionConfig | None = None,
        realtime_call_id: str | None = None,
        agents: list[Agent] | None = None,
        context: RunContext | Any = None,
        safety_identifier: str | None = None,
    ) -> OpenAIRealtimeSession:
        if not self.api_key:
            raise RealtimeConnectionError("OPENAI_API_KEY is required for Realtime")
        config = config or OpenAIRealtimeSessionConfig()
        try:
            import websockets
        except ImportError as error:
            raise RealtimeConnectionError(
                "Install modmex-ai[realtime] to use OpenAI Realtime WebSockets"
            ) from error
        query = (
            {"call_id": realtime_call_id}
            if realtime_call_id
            else {"model": config.model}
        )
        headers = {"Authorization": f"Bearer {self.api_key}"}
        if safety_identifier:
            headers["OpenAI-Safety-Identifier"] = safety_identifier
        try:
            transport = await websockets.connect(
                f"{self.websocket_url}?{urlencode(query)}",
                additional_headers=headers,
            )
        except Exception as error:
            raise RealtimeConnectionError("Could not connect to OpenAI Realtime") from error
        return OpenAIRealtimeSession(
            agent=agent,
            transport=transport,
            config=config,
            agents=agents,
            context=context,
            include_model_on_update=realtime_call_id is None,
        )


def _usage_from_response(response: Any) -> Usage:
    if not isinstance(response, dict):
        return Usage()
    usage = response.get("usage")
    if not isinstance(usage, dict):
        return Usage()
    input_details = usage.get("input_token_details", {})
    output_details = usage.get("output_token_details", {})
    return Usage(
        input_tokens=usage.get("input_tokens", 0) or 0,
        output_tokens=usage.get("output_tokens", 0) or 0,
        total_tokens=usage.get("total_tokens", 0) or 0,
        cached_input_tokens=(
            input_details.get("cached_tokens", 0) or 0
            if isinstance(input_details, dict)
            else 0
        ),
        reasoning_output_tokens=(
            output_details.get("reasoning_tokens", 0) or 0
            if isinstance(output_details, dict)
            else 0
        ),
        details={"raw": usage},
    )
