from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from modmex_ai.agents import Agent, AgentStreamEventType, RunContext
from modmex_ai.errors import ToolExecutionError, VoiceTurnCancelled
from modmex_ai.flows import Flow, FlowStreamEventType
from modmex_ai.models import Usage
from modmex_ai.schemas import dumps
from modmex_ai.sessions import InMemorySession, Session
from modmex_ai.voice.continuation import VoiceContinuation
from modmex_ai.voice.event import (
    VoiceSessionEvent,
    VoiceSessionEventType,
    VoiceTerminationReason,
)
from modmex_ai.voice.providers import (
    LiveSpeechToTextProvider,
    SpeechToTextProvider,
    StreamingSpeechToTextProvider,
    StreamingTextToSpeechProvider,
    TextToSpeechProvider,
    VoiceInputEvent,
    VoiceInputEventType,
)
from modmex_ai.voice.segment import SentenceBuffer
from modmex_ai.voice.session import VoiceAgentSession
from modmex_ai.voice.turn import VoiceTurnOptions
from modmex_ai.voice.turn import VoiceTurnMetrics, VoiceTurnOutcome, VoiceTurnReason


class ChainedVoiceSession(VoiceAgentSession):
    """A voice session where the application controls STT, reasoning, and TTS."""

    def __init__(
        self,
        *,
        agent: Agent | None = None,
        flow: Flow | None = None,
        speech_to_text: SpeechToTextProvider,
        text_to_speech: TextToSpeechProvider,
        context: RunContext | Any = None,
        session: Session | None = None,
        continuation: VoiceContinuation | None = None,
        turn_options: VoiceTurnOptions | None = None,
        turn_observers: list[Any] | None = None,
    ) -> None:
        if (agent is None) == (flow is None):
            raise ValueError("Provide exactly one of agent or flow")
        self.agent = agent
        self.flow = flow or Flow(
            name=f"{agent.name}-voice",
            entrypoint=agent,
            agents=[],
        )
        self.session = session or InMemorySession()
        self.continuation = continuation
        initial_agent_name = continuation.agent_name if continuation else self.flow.entrypoint
        self.current_agent = self.flow.agents[initial_agent_name]
        self.speech_to_text = speech_to_text
        self.text_to_speech = text_to_speech
        self.context = (
            context
            if isinstance(context, RunContext)
            else RunContext(input=None, context=context)
        )
        self.usage = Usage()
        self.turn_options = turn_options or VoiceTurnOptions()
        self._turn_cancelled = asyncio.Event()
        self._turn_started_at: float | None = None
        self._turn_stage: str | None = None
        self._turn_metrics = VoiceTurnMetrics()
        self._first_audio_seen = False
        self.last_turn: VoiceTurnOutcome | None = None
        self.turn_observers = turn_observers or []
        self._events: asyncio.Queue[VoiceSessionEvent | None] = asyncio.Queue()
        self._closed = False

    async def process_audio(self, audio: bytes) -> bytes:
        """Process one completed user-audio turn and return synthesized audio."""
        self._begin_turn()
        try:
            if self.turn_options.timeout_seconds is None:
                result = await self._process_audio(audio)
            else:
                result = await asyncio.wait_for(
                    self._process_audio(audio),
                    timeout=self.turn_options.timeout_seconds,
                )
        except VoiceTurnCancelled:
            self._finish_turn(VoiceTurnReason.CANCELLED)
            raise
        except asyncio.TimeoutError:
            self._finish_turn(VoiceTurnReason.TIMED_OUT)
            raise
        except Exception:
            self._finish_turn({
                "stt": VoiceTurnReason.STT_FAILED,
                "tts": VoiceTurnReason.TTS_FAILED,
            }.get(self._turn_stage, VoiceTurnReason.AGENT_FAILED))
            raise
        self._finish_turn(VoiceTurnReason.COMPLETED)
        return result

    async def _process_audio(self, audio: bytes) -> bytes:
        if self._closed:
            raise RuntimeError("Voice session is closed")
        transcript = await self._measure("stt", self._with_provider_retry(
            lambda: self.speech_to_text.transcribe(audio, context=self.context)
        ))
        await self._emit(VoiceSessionEvent(
            type=VoiceSessionEventType.TRANSCRIPT_FINAL,
            data={"transcript": transcript},
        ))
        return await self._respond_to_transcript(transcript)

    async def _respond_to_transcript(self, transcript: str) -> bytes:
        try:
            result = await self._measure("agent", self._run_turn(transcript))
        except ToolExecutionError as error:
            await self._emit(VoiceSessionEvent(
                type=VoiceSessionEventType.ERROR,
                data={
                    "reason": VoiceTerminationReason.TOOL_ERROR,
                    "message": str(error),
                },
            ))
            raise
        await self._emit_agent_events(result)
        self.current_agent = getattr(result, "last_agent", None) or self.agent
        self.usage.add(result.usage)
        response_text = _response_text(result.output)
        await self._emit(VoiceSessionEvent(
            type=VoiceSessionEventType.ASSISTANT_TRANSCRIPT_FINAL,
            data={"transcript": response_text},
        ))
        audio_output = await self._measure("tts", self._with_provider_retry(
            lambda: self.text_to_speech.synthesize(response_text, context=self.context)
        ))
        await self._emit(VoiceSessionEvent(
            type=VoiceSessionEventType.RESPONSE_COMPLETED,
            data={"usage": result.usage.model_dump()},
        ))
        return audio_output

    async def process_live_audio(
        self,
        input_events: AsyncIterator[VoiceInputEvent],
    ) -> AsyncIterator[bytes]:
        """Run a continuous host-driven conversation over one STT connection.

        The host supplies audio chunks and explicitly commits each caller turn. Every
        final transcript uses the same Flow/session/continuation and yields one
        synthesized response. VAD and playback remain host responsibilities.
        """
        if not isinstance(self.speech_to_text, LiveSpeechToTextProvider):
            raise TypeError("speech_to_text does not implement LiveSpeechToTextProvider")
        if self._closed:
            raise RuntimeError("Voice session is closed")

        transcription_session = await self.speech_to_text.connect_live()
        producer = asyncio.create_task(
            self._feed_live_audio(transcription_session, input_events)
        )
        try:
            async for transcription in transcription_session.transcriptions():
                await self._emit(VoiceSessionEvent(
                    type=(
                        VoiceSessionEventType.TRANSCRIPT_FINAL
                        if transcription.is_final
                        else VoiceSessionEventType.TRANSCRIPT_DELTA
                    ),
                    data={"transcript": transcription.text},
                ))
                if transcription.is_final:
                    yield await self._process_transcript_turn(transcription.text)
                    # A cancel issued while the host played that reply targets
                    # playback (barge-in), not the caller's next turn.
                    self._turn_cancelled.clear()
        finally:
            if not producer.done():
                producer.cancel()
            producer_results = await asyncio.gather(producer, return_exceptions=True)
            await transcription_session.close()
            producer_error = producer_results[0]
            if isinstance(producer_error, Exception) and not isinstance(
                producer_error,
                asyncio.CancelledError,
            ):
                raise producer_error

    async def _feed_live_audio(self, transcription_session, input_events: AsyncIterator[VoiceInputEvent]) -> None:
        # Cancellation stops turn processing only; audio keeps flowing so the
        # caller can be heard after an interruption.
        async for event in input_events:
            if event.type == VoiceInputEventType.AUDIO:
                if event.audio is None:
                    raise ValueError("An audio input event requires audio bytes")
                await transcription_session.append_audio(event.audio)
            elif event.type == VoiceInputEventType.COMMIT_TURN:
                await transcription_session.commit_turn()

    async def _process_transcript_turn(self, transcript: str) -> bytes:
        self._begin_turn()
        try:
            if self.turn_options.timeout_seconds is None:
                response = await self._respond_to_transcript(transcript)
            else:
                response = await asyncio.wait_for(
                    self._respond_to_transcript(transcript),
                    timeout=self.turn_options.timeout_seconds,
                )
        except VoiceTurnCancelled:
            self._finish_turn(VoiceTurnReason.CANCELLED)
            raise
        except asyncio.TimeoutError:
            self._finish_turn(VoiceTurnReason.TIMED_OUT)
            raise
        except Exception:
            self._finish_turn({"tts": VoiceTurnReason.TTS_FAILED}.get(
                self._turn_stage,
                VoiceTurnReason.AGENT_FAILED,
            ))
            raise
        self._finish_turn(VoiceTurnReason.COMPLETED)
        return response

    async def process_audio_stream(self, audio: AsyncIterator[bytes]) -> AsyncIterator[bytes]:
        """Run a live STT → text-agent → streaming-TTS turn when providers support it."""
        if not isinstance(self.speech_to_text, StreamingSpeechToTextProvider):
            raise TypeError("speech_to_text does not implement StreamingSpeechToTextProvider")
        if not isinstance(self.text_to_speech, StreamingTextToSpeechProvider):
            raise TypeError("text_to_speech does not implement StreamingTextToSpeechProvider")
        self._begin_turn()
        try:
            async for chunk in self._process_audio_stream(audio):
                yield chunk
        except VoiceTurnCancelled:
            self._finish_turn(VoiceTurnReason.CANCELLED)
            raise
        except Exception:
            self._finish_turn({
                "stt": VoiceTurnReason.STT_FAILED,
                "tts": VoiceTurnReason.TTS_FAILED,
            }.get(self._turn_stage, VoiceTurnReason.AGENT_FAILED))
            raise
        self._finish_turn(VoiceTurnReason.COMPLETED)

    async def _process_audio_stream(self, audio: AsyncIterator[bytes]) -> AsyncIterator[bytes]:
        from time import perf_counter

        self._turn_stage = "stt"
        stt_started = perf_counter()
        final_transcript: str | None = None
        async for transcription in self.speech_to_text.transcribe_stream(audio, context=self.context):
            self._raise_if_turn_cancelled()
            event_type = (
                VoiceSessionEventType.TRANSCRIPT_FINAL
                if transcription.is_final
                else VoiceSessionEventType.TRANSCRIPT_DELTA
            )
            await self._emit(VoiceSessionEvent(type=event_type, data={"transcript": transcription.text}))
            if transcription.is_final:
                final_transcript = transcription.text
        if final_transcript is None:
            raise ValueError("Streaming transcription ended without a final transcript")
        self._turn_metrics.stt_latency_ms = int((perf_counter() - stt_started) * 1000)
        if self._can_speak_while_generating():
            async for chunk in self._speak_while_generating(final_transcript):
                yield chunk
            return
        result = await self._measure("agent", self._run_turn(final_transcript))
        await self._emit_agent_events(result)
        self.current_agent = result.last_agent
        self.usage.add(result.usage)
        response_text = _response_text(result.output)
        await self._emit(VoiceSessionEvent(
            type=VoiceSessionEventType.ASSISTANT_TRANSCRIPT_FINAL,
            data={"transcript": response_text},
        ))
        async for chunk in self._synthesize_segment(response_text):
            yield chunk
        await self._emit(VoiceSessionEvent(
            type=VoiceSessionEventType.RESPONSE_COMPLETED,
            data={"usage": result.usage.model_dump()},
        ))

    def _can_speak_while_generating(self) -> bool:
        """Incremental speech needs native async streaming from every agent's model."""
        if not self.turn_options.speak_while_generating:
            return False
        return all(
            callable(getattr(self.flow.model or agent.model, "astream", None))
            for agent in self.flow.agents.values()
        )

    async def _speak_while_generating(self, transcript: str) -> AsyncIterator[bytes]:
        """Stream the Flow and synthesize each sentence as soon as it is complete.

        A producer task reads model text into a queue while this generator
        synthesizes and yields audio, so the model keeps generating during
        speech. Text from agents with structured output or output guardrails is
        not spoken early (it is not speakable, or could still be rejected); their
        final reply is synthesized once the turn completes.
        """
        import asyncio
        from time import perf_counter

        self._raise_if_turn_cancelled()
        queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()
        agent_started = perf_counter()

        async def say(sentence: str) -> None:
            await queue.put(("say", sentence))
            # Let the consumer start synthesizing before generation (or a tool
            # call) continues; an unbounded put never yields on its own.
            await asyncio.sleep(0)

        async def produce() -> None:
            buffer = SentenceBuffer()
            speaking_agent: str | None = None
            try:
                async for event in self.flow.arun_stream(transcript, **self._flow_kwargs()):
                    if event.type == FlowStreamEventType.COMPLETED:
                        tail = buffer.flush()
                        if tail:
                            await say(tail)
                        # Generation time, excluding the TTS that overlapped it.
                        self._turn_metrics.agent_latency_ms = int(
                            (perf_counter() - agent_started) * 1000
                        )
                        await queue.put(("done", event.result))
                        return
                    agent_event = event.agent_event
                    is_text = (
                        agent_event is not None
                        and agent_event.type == AgentStreamEventType.TEXT_DELTA
                    )
                    if event.agent_name != speaking_agent or not is_text:
                        # The model stopped writing text (tool call, tool result,
                        # handoff, another agent): speak what it wrote so far now,
                        # e.g. "Let me check." before a tool runs. Models often end
                        # that text without trailing whitespace.
                        tail = buffer.flush()
                        if tail:
                            await say(tail)
                        speaking_agent = event.agent_name
                    if is_text and agent_event.text_delta and _speakable(
                        self.flow.agents.get(event.agent_name)
                    ):
                        for sentence in buffer.feed(agent_event.text_delta):
                            await say(sentence)
            except Exception as error:  # surfaced to the consumer; cancellation propagates
                await queue.put(("error", error))

        producer = asyncio.create_task(produce())
        spoken: list[str] = []
        try:
            self._turn_stage = "agent"
            while True:
                kind, value = await queue.get()
                if kind == "error":
                    raise value
                if kind == "done":
                    result = value
                    break
                self._raise_if_turn_cancelled()
                spoken.append(value)
                async for chunk in self._synthesize_segment(value):
                    yield chunk
            self.continuation = VoiceContinuation.from_flow(result.continuation)
            await self._emit_agent_events(result)
            self.current_agent = result.last_agent
            self.usage.add(result.usage)
            if not _speakable(result.last_agent):
                final_text = _response_text(result.output)
                spoken.append(final_text)
                async for chunk in self._synthesize_segment(final_text):
                    yield chunk
            await self._emit(VoiceSessionEvent(
                type=VoiceSessionEventType.ASSISTANT_TRANSCRIPT_FINAL,
                data={"transcript": " ".join(spoken)},
            ))
            await self._emit(VoiceSessionEvent(
                type=VoiceSessionEventType.RESPONSE_COMPLETED,
                data={"usage": result.usage.model_dump()},
            ))
        finally:
            if not producer.done():
                producer.cancel()
            await asyncio.gather(producer, return_exceptions=True)

    async def _synthesize_segment(self, text: str) -> AsyncIterator[bytes]:
        from time import perf_counter

        stage, self._turn_stage = self._turn_stage, "tts"
        started = perf_counter()
        async for chunk in self.text_to_speech.synthesize_stream(text, context=self.context):
            self._raise_if_turn_cancelled()
            self._mark_first_audio()
            yield chunk
        self._turn_metrics.tts_latency_ms += int((perf_counter() - started) * 1000)
        self._turn_stage = stage

    def _mark_first_audio(self) -> None:
        from time import perf_counter

        if not self._first_audio_seen and self._turn_started_at is not None:
            self._first_audio_seen = True
            self._turn_metrics.first_audio_latency_ms = int(
                (perf_counter() - self._turn_started_at) * 1000
            )

    async def voice_events(self) -> AsyncIterator[VoiceSessionEvent]:
        while True:
            event = await self._events.get()
            if event is None:
                return
            yield event

    async def close(
        self,
        reason: VoiceTerminationReason = VoiceTerminationReason.ENDED_BY_APPLICATION,
    ) -> None:
        if self._closed:
            return
        self._closed = True
        await self._emit(VoiceSessionEvent(
            type=VoiceSessionEventType.SESSION_ENDED,
            data={"reason": reason},
        ))
        await self._events.put(None)

    def cancel_current_turn(self) -> None:
        """Request cancellation; streaming stops before the next provider chunk."""
        self._turn_cancelled.set()

    async def _emit(self, event: VoiceSessionEvent) -> None:
        await self._events.put(event)

    async def _run_turn(self, transcript: str):
        self._raise_if_turn_cancelled()
        result = await self.flow.run_async(transcript, **self._flow_kwargs())
        self.continuation = VoiceContinuation.from_flow(result.continuation)
        return result

    def _flow_kwargs(self) -> dict[str, Any]:
        return {
            "starting_agent": self.continuation.agent_name if self.continuation else None,
            "context": self.context.context,
            "session": self.session,
            "provider_state": (
                self.continuation.flow_continuation.provider_state
                if self.continuation
                else None
            ),
        }

    async def _with_provider_retry(self, operation):
        for attempt in range(self.turn_options.max_provider_retries + 1):
            self._raise_if_turn_cancelled()
            try:
                return await operation()
            except asyncio.CancelledError:
                raise
            except Exception:
                if attempt >= self.turn_options.max_provider_retries:
                    raise
                self._turn_metrics.provider_retry_count += 1
                if self.turn_options.retry_delay_seconds:
                    await asyncio.sleep(self.turn_options.retry_delay_seconds)
        raise AssertionError("unreachable")

    def _raise_if_turn_cancelled(self) -> None:
        if self._turn_cancelled.is_set():
            self._turn_cancelled.clear()
            raise VoiceTurnCancelled("Voice turn was cancelled")

    async def _measure(self, stage: str, operation):
        from time import perf_counter

        started_at = perf_counter()
        self._turn_stage = stage
        try:
            return await operation
        finally:
            setattr(
                self._turn_metrics,
                f"{stage}_latency_ms",
                int((perf_counter() - started_at) * 1000),
            )

    def _begin_turn(self) -> None:
        from time import perf_counter

        self._turn_started_at = perf_counter()
        self._turn_stage = None
        self._turn_metrics = VoiceTurnMetrics()
        self._first_audio_seen = False
        self.last_turn = None

    def _finish_turn(self, reason: VoiceTurnReason) -> None:
        from time import perf_counter

        if self._turn_started_at is not None:
            self._turn_metrics.total_latency_ms = int(
                (perf_counter() - self._turn_started_at) * 1000
            )
        self.last_turn = VoiceTurnOutcome(reason=reason, metrics=self._turn_metrics)
        for observer in self.turn_observers:
            try:
                observer.on_voice_turn_finished(self.last_turn)
            except Exception:  # metrics must never change or mask the turn outcome
                pass

    async def _emit_agent_events(self, result: Any) -> None:
        agent_results = getattr(result, "agent_results", [result])
        for agent_result in agent_results:
            for item in agent_result.items:
                if item.type == "function_call_output":
                    await self._emit(VoiceSessionEvent(
                        type=VoiceSessionEventType.TOOL_FINISHED,
                        data={
                            "tool_call_id": item.tool_call_id,
                            "tool_name": item.name,
                        },
                    ))
            if agent_result.handoff_target:
                await self._emit(VoiceSessionEvent(
                    type=VoiceSessionEventType.HANDOFF_COMPLETED,
                    data={
                        "handoff_name": agent_result.handoff_name,
                        "agent_name": agent_result.handoff_target,
                    },
                ))


def _speakable(agent: Agent | None) -> bool:
    """Whether an agent's streamed text can be spoken before its turn completes."""
    return agent is not None and agent.output_type is None and not agent.output_guardrails


def _response_text(output: Any) -> str:
    reply = getattr(output, "reply", None)
    if isinstance(reply, str):
        return reply
    if isinstance(output, str):
        return output
    return dumps(output)
