import asyncio

from modmex_ai import (
    Agent,
    ChainedVoiceSession,
    FakeModel,
    Flow,
    InMemorySession,
    ModelResponse,
    ToolCall,
    Transcription,
    VoiceTurnOptions,
    VoiceTurnReason,
)
from modmex_ai.voice import (
    CallableSpeechToTextProvider,
    CallableTextToSpeechProvider,
    LiveSpeechToTextProvider,
    LiveSpeechToTextSession,
    SpeechToTextProvider,
    StreamingSpeechToTextProvider,
    StreamingTextToSpeechProvider,
    TextToSpeechProvider,
    VoiceSessionEventType,
    VoiceTerminationReason,
    VoiceInputEvent,
)
from modmex_ai.errors import VoiceTurnCancelled
from modmex_ai.errors import ToolExecutionError
import pytest


class FakeSpeechToText(SpeechToTextProvider):
    async def transcribe(self, audio: bytes, *, context):
        assert audio == b"caller-audio"
        return "I need dispatch support"


class FakeTextToSpeech(TextToSpeechProvider):
    async def synthesize(self, text: str, *, context):
        return f"audio:{text}".encode()


class FakeStreamingSpeechToText(StreamingSpeechToTextProvider):
    async def transcribe(self, audio: bytes, *, context):
        return "unused"

    async def transcribe_stream(self, audio, *, context):
        async for _chunk in audio:
            yield Transcription(text="I need", is_final=False)
        yield Transcription(text="I need dispatch support", is_final=True)


class FakeStreamingTextToSpeech(StreamingTextToSpeechProvider):
    async def synthesize(self, text: str, *, context):
        return text.encode()

    async def synthesize_stream(self, text: str, *, context):
        yield b"audio:"
        yield text.encode()


class FakeLiveTranscriptionSession(LiveSpeechToTextSession):
    def __init__(self):
        self.audio = []
        self.commits = 0
        self.closed = False

    async def append_audio(self, chunk):
        self.audio.append(chunk)

    async def commit_turn(self):
        self.commits += 1

    async def close(self):
        self.closed = True

    async def _events(self):
        yield Transcription(text="Need", is_final=False)
        yield Transcription(text="Need dispatch support", is_final=True, item_id="turn-1")
        yield Transcription(text="Need another lane", is_final=True, item_id="turn-2")

    def transcriptions(self):
        return self._events()


class FakeLiveSpeechToText(LiveSpeechToTextProvider):
    def __init__(self):
        self.live_session = FakeLiveTranscriptionSession()

    async def transcribe(self, audio, *, context):
        return "unused"

    async def transcribe_stream(self, audio, *, context):
        if False:
            yield Transcription(text="unused")

    async def connect_live(self):
        return self.live_session


def test_chained_voice_session_preserves_turn_events_and_usage():
    async def run():
        agent = Agent(
            name="dispatcher",
            instructions="Help the carrier.",
            model=FakeModel(["I can help you with that."]),
        )
        session = ChainedVoiceSession(
            agent=agent,
            speech_to_text=FakeSpeechToText(),
            text_to_speech=FakeTextToSpeech(),
        )

        assert await session.process_audio(b"caller-audio") == b"audio:I can help you with that."
        await session.close(VoiceTerminationReason.COMPLETED)
        events = [event async for event in session.voice_events()]

        assert [event.type for event in events] == [
            VoiceSessionEventType.TRANSCRIPT_FINAL,
            VoiceSessionEventType.ASSISTANT_TRANSCRIPT_FINAL,
            VoiceSessionEventType.RESPONSE_COMPLETED,
            VoiceSessionEventType.SESSION_ENDED,
        ]
        assert events[0].data == {"transcript": "I need dispatch support"}
        assert events[-1].data == {"reason": VoiceTerminationReason.COMPLETED}

    asyncio.run(run())


def test_chained_voice_session_runs_a_flow_and_tracks_the_active_agent():
    async def run():
        support = Agent(
            name="support",
            instructions="Resolve the request.",
            model=FakeModel(["I will take care of it."]),
        )
        triage = Agent(
            name="triage",
            instructions="Route the request.",
            handoffs=["support"],
            model=FakeModel([
                ModelResponse(tool_calls=[ToolCall(
                    tool_call_id="handoff-1",
                    name="transfer_to_support",
                    arguments={},
                )]),
            ]),
        )
        flow = Flow(name="voice", entrypoint=triage, agents=[support])
        session = ChainedVoiceSession(
            flow=flow,
            speech_to_text=FakeSpeechToText(),
            text_to_speech=FakeTextToSpeech(),
        )

        assert session.current_agent is triage
        assert await session.process_audio(b"caller-audio") == b"audio:I will take care of it."
        assert session.current_agent is support
        await session.close()
        events = [event async for event in session.voice_events()]
        assert VoiceSessionEventType.HANDOFF_COMPLETED in [event.type for event in events]

    asyncio.run(run())


def test_chained_voice_session_restores_agent_and_history_from_portable_continuation():
    async def run():
        shared_session = InMemorySession(session_id="voice-1")
        agent = Agent(
            name="dispatcher",
            instructions="Continue the carrier conversation.",
            model=FakeModel(["First reply.", "Second reply."]),
        )
        first = ChainedVoiceSession(
            agent=agent,
            speech_to_text=FakeSpeechToText(),
            text_to_speech=FakeTextToSpeech(),
            session=shared_session,
        )
        await first.process_audio(b"caller-audio")

        second = ChainedVoiceSession(
            agent=agent,
            speech_to_text=FakeSpeechToText(),
            text_to_speech=FakeTextToSpeech(),
            session=shared_session,
            continuation=first.continuation,
        )
        assert await second.process_audio(b"caller-audio") == b"audio:Second reply."
        assert second.current_agent is agent
        assert second.continuation.agent_name == "dispatcher"
        assert len(shared_session.get_items()) == 4

    asyncio.run(run())


def test_chained_voice_session_streams_transcription_and_speech():
    async def chunks():
        yield b"first"
        yield b"second"

    async def run():
        session = ChainedVoiceSession(
            agent=Agent(
                name="dispatcher",
                instructions="Help.",
                model=FakeModel(["I can help."]),
            ),
            speech_to_text=FakeStreamingSpeechToText(),
            text_to_speech=FakeStreamingTextToSpeech(),
        )

        response = b"".join([chunk async for chunk in session.process_audio_stream(chunks())])
        await session.close()
        events = [event async for event in session.voice_events()]

        assert response == b"audio:I can help."
        assert [event.type for event in events] == [
            VoiceSessionEventType.TRANSCRIPT_DELTA,
            VoiceSessionEventType.TRANSCRIPT_DELTA,
            VoiceSessionEventType.TRANSCRIPT_FINAL,
            VoiceSessionEventType.ASSISTANT_TRANSCRIPT_FINAL,
            VoiceSessionEventType.RESPONSE_COMPLETED,
            VoiceSessionEventType.SESSION_ENDED,
        ]

    asyncio.run(run())


def test_chained_voice_session_retries_provider_io_without_rerunning_the_agent():
    class FlakySpeechToText(FakeSpeechToText):
        def __init__(self):
            self.calls = 0

        async def transcribe(self, audio, *, context):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("temporary provider failure")
            return await super().transcribe(audio, context=context)

    async def run():
        stt = FlakySpeechToText()
        model = FakeModel(["Recovered reply."])
        session = ChainedVoiceSession(
            agent=Agent(name="dispatcher", instructions="Help.", model=model),
            speech_to_text=stt,
            text_to_speech=FakeTextToSpeech(),
            turn_options=VoiceTurnOptions(max_provider_retries=1),
        )

        assert await session.process_audio(b"caller-audio") == b"audio:Recovered reply."
        assert stt.calls == 2
        assert len(model.requests) == 1

    asyncio.run(run())


def test_chained_voice_session_stops_a_cancelled_streaming_turn():
    async def chunks():
        yield b"first"
        yield b"second"

    async def run():
        session = ChainedVoiceSession(
            agent=Agent(name="dispatcher", instructions="Help.", model=FakeModel(["unused"])),
            speech_to_text=FakeStreamingSpeechToText(),
            text_to_speech=FakeStreamingTextToSpeech(),
        )
        stream = session.process_audio_stream(chunks())
        session.cancel_current_turn()

        try:
            await anext(stream)
        except VoiceTurnCancelled:
            pass
        else:
            raise AssertionError("Expected VoiceTurnCancelled")

    asyncio.run(run())


def test_chained_voice_session_records_timeout_and_stage_metrics():
    class SlowSpeechToText(FakeSpeechToText):
        async def transcribe(self, audio, *, context):
            await asyncio.sleep(0.01)
            return await super().transcribe(audio, context=context)

    async def run():
        session = ChainedVoiceSession(
            agent=Agent(name="dispatcher", instructions="Help.", model=FakeModel(["unused"])),
            speech_to_text=SlowSpeechToText(),
            text_to_speech=FakeTextToSpeech(),
            turn_options=VoiceTurnOptions(timeout_seconds=0.001),
        )

        try:
            await session.process_audio(b"caller-audio")
        except asyncio.TimeoutError:
            pass
        else:
            raise AssertionError("Expected turn timeout")

        assert session.last_turn.reason == VoiceTurnReason.TIMED_OUT
        assert session.last_turn.metrics.total_latency_ms >= 0
        assert session.last_turn.metrics.stt_latency_ms >= 0

    asyncio.run(run())


def test_chained_voice_session_orchestrates_multiple_live_transcription_turns():
    async def input_events():
        yield VoiceInputEvent.audio_chunk(b"first")
        yield VoiceInputEvent.commit_turn()
        yield VoiceInputEvent.audio_chunk(b"second")
        yield VoiceInputEvent.commit_turn()

    async def run():
        stt = FakeLiveSpeechToText()
        session = ChainedVoiceSession(
            agent=Agent(
                name="dispatcher",
                instructions="Help.",
                model=FakeModel(["First reply.", "Second reply."]),
            ),
            speech_to_text=stt,
            text_to_speech=FakeTextToSpeech(),
        )

        replies = [reply async for reply in session.process_live_audio(input_events())]
        await session.close()
        events = [event async for event in session.voice_events()]

        assert replies == [b"audio:First reply.", b"audio:Second reply."]
        assert stt.live_session.audio == [b"first", b"second"]
        assert stt.live_session.commits == 2
        assert stt.live_session.closed
        assert [event.type for event in events].count(
            VoiceSessionEventType.TRANSCRIPT_FINAL
        ) == 2
        assert session.continuation.agent_name == "dispatcher"

    asyncio.run(run())


def test_chained_voice_session_notifies_turn_observers():
    class Observer:
        def __init__(self):
            self.outcomes = []

        def on_voice_turn_finished(self, outcome):
            self.outcomes.append(outcome)

    async def run():
        observer = Observer()
        session = ChainedVoiceSession(
            agent=Agent(name="dispatcher", instructions="Help.", model=FakeModel(["Done."])),
            speech_to_text=FakeSpeechToText(),
            text_to_speech=FakeTextToSpeech(),
            turn_observers=[observer],
        )
        await session.process_audio(b"caller-audio")
        assert observer.outcomes[0].reason == VoiceTurnReason.COMPLETED

    asyncio.run(run())


def test_chained_voice_validates_provider_shapes_and_closed_lifecycle():
    agent = Agent(name="a", instructions="Help.", model=FakeModel(["x"]))
    with pytest.raises(ValueError):
        ChainedVoiceSession(agent=None, flow=None, speech_to_text=FakeSpeechToText(), text_to_speech=FakeTextToSpeech())
    with pytest.raises(ValueError):
        ChainedVoiceSession(agent=agent, flow=Flow(name="f", entrypoint=agent, agents=[]), speech_to_text=FakeSpeechToText(), text_to_speech=FakeTextToSpeech())

    async def run():
        session = ChainedVoiceSession(agent=agent, speech_to_text=FakeSpeechToText(), text_to_speech=FakeTextToSpeech())
        with pytest.raises(TypeError):
            _ = [item async for item in session.process_live_audio(_empty_events())]
        with pytest.raises(TypeError):
            _ = [item async for item in session.process_audio_stream(_empty_audio())]
        await session.close()
        await session.close()
        with pytest.raises(RuntimeError):
            await session.process_audio(b"caller-audio")

    asyncio.run(run())


def test_chained_voice_records_stt_tts_failures_and_rejects_invalid_live_input():
    class BrokenStt(FakeSpeechToText):
        async def transcribe(self, audio, *, context):
            raise RuntimeError("stt unavailable")

    class BrokenTts(FakeTextToSpeech):
        async def synthesize(self, text, *, context):
            raise RuntimeError("tts unavailable")

    class InvalidLive(FakeLiveSpeechToText):
        async def connect_live(self):
            class Session(FakeLiveTranscriptionSession):
                async def _events(self):
                    await asyncio.sleep(0.01)
                    if False:
                        yield None
                def transcriptions(self):
                    return self._events()
            return Session()

    async def invalid_events():
        yield VoiceInputEvent(type="audio")

    async def run():
        agent = Agent(name="a", instructions="x", model=FakeModel(["reply"]))
        stt = ChainedVoiceSession(agent=agent, speech_to_text=BrokenStt(), text_to_speech=FakeTextToSpeech())
        with pytest.raises(RuntimeError):
            await stt.process_audio(b"caller-audio")
        assert stt.last_turn.reason == VoiceTurnReason.STT_FAILED
        tts = ChainedVoiceSession(agent=agent, speech_to_text=FakeSpeechToText(), text_to_speech=BrokenTts())
        with pytest.raises(RuntimeError):
            await tts.process_audio(b"caller-audio")
        assert tts.last_turn.reason == VoiceTurnReason.TTS_FAILED
        live = ChainedVoiceSession(agent=agent, speech_to_text=InvalidLive(), text_to_speech=FakeTextToSpeech())
        with pytest.raises(ValueError):
            _ = [output async for output in live.process_live_audio(invalid_events())]

    asyncio.run(run())


def test_chained_voice_rejects_stream_without_a_final_transcript():
    class NoFinal(FakeStreamingSpeechToText):
        async def transcribe_stream(self, audio, *, context):
            async for _ in audio:
                yield Transcription(text="partial")

    async def chunks():
        yield b"audio"

    async def run():
        session = ChainedVoiceSession(agent=Agent(name="a", instructions="x", model=FakeModel(["unused"])), speech_to_text=NoFinal(), text_to_speech=FakeStreamingTextToSpeech())
        with pytest.raises(ValueError):
            _ = [item async for item in session.process_audio_stream(chunks())]

    asyncio.run(run())


def test_chained_voice_marks_cancelled_and_tool_failed_turns():
    async def run():
        cancelled = ChainedVoiceSession(
            agent=Agent(name="a", instructions="x", model=FakeModel(["unused"])),
            speech_to_text=FakeSpeechToText(), text_to_speech=FakeTextToSpeech(),
        )
        cancelled.cancel_current_turn()
        with pytest.raises(VoiceTurnCancelled):
            await cancelled.process_audio(b"caller-audio")
        assert cancelled.last_turn.reason == VoiceTurnReason.CANCELLED

        def broken_tool():
            raise RuntimeError("tool failed")

        broken = ChainedVoiceSession(
            agent=Agent(name="a", instructions="x", tools=[broken_tool], model=FakeModel([
                ModelResponse(tool_calls=[ToolCall(tool_call_id="t", name="broken_tool", arguments={})]),
            ])),
            speech_to_text=FakeSpeechToText(), text_to_speech=FakeTextToSpeech(),
        )
        with pytest.raises(ToolExecutionError):
            await broken.process_audio(b"caller-audio")
        assert broken.last_turn.reason == VoiceTurnReason.AGENT_FAILED

    asyncio.run(run())


def test_callable_voice_providers_support_sync_and_async_functions():
    async def async_tts(text, context):
        return text.encode()

    async def run():
        sync_stt = CallableSpeechToTextProvider(lambda audio, context: audio.decode())
        async_stt = CallableSpeechToTextProvider(lambda audio, context: _async_text(audio))
        sync_tts = CallableTextToSpeechProvider(lambda text, context: text.encode())
        async_tts_provider = CallableTextToSpeechProvider(async_tts)
        context = None
        assert await sync_stt.transcribe(b"sync", context=context) == "sync"
        assert await async_stt.transcribe(b"async", context=context) == "async"
        assert await sync_tts.synthesize("sync", context=context) == b"sync"
        assert await async_tts_provider.synthesize("async", context=context) == b"async"

    asyncio.run(run())


async def _async_text(audio):
    return audio.decode()


async def _empty_events():
    if False:
        yield VoiceInputEvent.commit_turn()


async def _empty_audio():
    if False:
        yield b""


def test_live_barge_in_cancel_does_not_end_the_conversation():
    async def input_events():
        yield VoiceInputEvent.audio_chunk(b"first")
        yield VoiceInputEvent.commit_turn()
        yield VoiceInputEvent.audio_chunk(b"second")
        yield VoiceInputEvent.commit_turn()

    async def run():
        session = ChainedVoiceSession(
            agent=Agent(name="dispatcher", instructions="Help.", model=FakeModel(["First reply.", "Second reply."])),
            speech_to_text=FakeLiveSpeechToText(),
            text_to_speech=FakeTextToSpeech(),
        )
        replies = []
        async for reply in session.process_live_audio(input_events()):
            replies.append(reply)
            session.cancel_current_turn()  # caller interrupts while this reply plays
        assert replies == [b"audio:First reply.", b"audio:Second reply."]
        assert session.last_turn.reason == VoiceTurnReason.COMPLETED

    asyncio.run(run())


def test_streaming_turn_records_outcome_and_observer_errors_do_not_mask_it():
    class BrokenObserver:
        def on_voice_turn_finished(self, outcome):
            raise RuntimeError("metrics backend down")

    async def chunks():
        yield b"audio"

    async def run():
        session = ChainedVoiceSession(
            agent=Agent(name="dispatcher", instructions="Help.", model=FakeModel(["Streamed reply."])),
            speech_to_text=FakeStreamingSpeechToText(),
            text_to_speech=FakeStreamingTextToSpeech(),
            turn_observers=[BrokenObserver()],
        )
        audio = [chunk async for chunk in session.process_audio_stream(chunks())]
        assert audio
        assert session.last_turn.reason == VoiceTurnReason.COMPLETED
        assert session.last_turn.metrics.total_latency_ms >= 0

    asyncio.run(run())


# --- speak while generating -------------------------------------------------

from modmex_ai.models import ModelProfile, ModelStreamEvent, ModelStreamEventType
from modmex_ai.voice import VoiceTurnOptions
from modmex_ai.voice.segment import SentenceBuffer
from modmex_ai import tool


class ScriptedStreamingModel:
    """Native async-streaming model: each turn is (text deltas, final response)."""

    name = "scripted"
    profile = ModelProfile(supports_structured_output=True)

    def __init__(self, turns, *, gate_after_first_delta=None):
        self.turns = list(turns)
        self.gate = gate_after_first_delta

    def complete(self, request):
        raise AssertionError("voice streaming must not use complete()")

    async def astream(self, request):
        deltas, response = self.turns.pop(0)
        for index, delta in enumerate(deltas):
            yield ModelStreamEvent(type=ModelStreamEventType.TEXT_DELTA, text_delta=delta)
            if index == 0 and self.gate is not None:
                await asyncio.wait_for(self.gate.wait(), 1)  # resumes only once TTS started
        yield ModelStreamEvent.completed(response)


class RecordingStreamingTts(FakeStreamingTextToSpeech):
    def __init__(self, started=None):
        self.texts = []
        self.started = started

    async def synthesize_stream(self, text, *, context):
        self.texts.append(text)
        if self.started is not None:
            self.started.set()
        yield f"audio:{text}".encode()


async def _one_chunk():
    yield b"caller"


def test_sentence_buffer_splits_on_sentence_ends_not_decimals():
    buffer = SentenceBuffer()
    out = []
    for delta in ["Hola, ", "claro. El costo es 3.5 ", "dólares! ¿Algo más", "? Ok.", " Gracias."]:
        out += buffer.feed(delta)
    assert out == ["Hola, claro.", "El costo es 3.5 dólares!", "¿Algo más? Ok."]
    assert buffer.flush() == "Gracias."
    long = SentenceBuffer(max_chars=40)
    assert long.feed("palabra " * 10) == ["palabra palabra palabra palabra palabra"]
    assert long.flush() == "palabra palabra palabra palabra palabra"


def test_streaming_turn_speaks_first_sentence_while_model_is_still_generating():
    async def run():
        tts_started = asyncio.Event()
        model = ScriptedStreamingModel(
            [(["Claro, te ayudo. ", "Tu carga sale mañana ", "a las ocho."],
              ModelResponse(output_text="Claro, te ayudo. Tu carga sale mañana a las ocho."))],
            gate_after_first_delta=tts_started,
        )
        tts = RecordingStreamingTts(started=tts_started)
        session = ChainedVoiceSession(
            agent=Agent(name="dispatcher", instructions="Help.", model=model),
            speech_to_text=FakeStreamingSpeechToText(),
            text_to_speech=tts,
        )
        audio = [chunk async for chunk in session.process_audio_stream(_one_chunk())]
        await session.close()
        events = [event async for event in session.voice_events()]

        assert tts.texts == ["Claro, te ayudo.", "Tu carga sale mañana a las ocho."]
        assert audio == ["audio:Claro, te ayudo.".encode(), "audio:Tu carga sale mañana a las ocho.".encode()]
        final = [e for e in events if e.type == VoiceSessionEventType.ASSISTANT_TRANSCRIPT_FINAL]
        assert final[0].data["transcript"] == "Claro, te ayudo. Tu carga sale mañana a las ocho."
        assert session.last_turn.reason == VoiceTurnReason.COMPLETED
        metrics = session.last_turn.metrics
        assert 0 <= metrics.first_audio_latency_ms <= metrics.total_latency_ms
        assert [item.role for item in session.session.get_items()] == ["user", "assistant"]

    asyncio.run(run())


def test_streaming_turn_speaks_preamble_before_a_tool_runs():
    calls = []

    @tool
    def lookup_load(load_id: str) -> dict:
        """Look up a load."""
        calls.append(load_id)
        return {"status": "in transit"}

    async def run():
        model = ScriptedStreamingModel([
            (["Déjame revisar tu carga. "], ModelResponse(
                output_text="Déjame revisar tu carga.",
                tool_calls=[ToolCall(tool_call_id="t1", name="lookup_load", arguments={"load_id": "L1"})],
            )),
            (["Tu carga va en tránsito."], ModelResponse(output_text="Tu carga va en tránsito.")),
        ])
        tts = RecordingStreamingTts()
        session = ChainedVoiceSession(
            agent=Agent(name="dispatcher", instructions="Help.", tools=[lookup_load], model=model),
            speech_to_text=FakeStreamingSpeechToText(),
            text_to_speech=tts,
        )
        _ = [chunk async for chunk in session.process_audio_stream(_one_chunk())]
        assert calls == ["L1"]
        assert tts.texts == ["Déjame revisar tu carga.", "Tu carga va en tránsito."]

    asyncio.run(run())


def test_structured_output_agent_is_spoken_once_complete():
    from modmex import BaseModel as ModmexModel

    class Reply(ModmexModel):
        reply: str

    async def run():
        model = ScriptedStreamingModel([
            (['{"reply":', '"Listo."}'], ModelResponse(output_text='{"reply":"Listo."}')),
        ])
        tts = RecordingStreamingTts()
        session = ChainedVoiceSession(
            agent=Agent(name="dispatcher", instructions="Help.", output_type=Reply, model=model),
            speech_to_text=FakeStreamingSpeechToText(),
            text_to_speech=tts,
        )
        _ = [chunk async for chunk in session.process_audio_stream(_one_chunk())]
        assert tts.texts == ["Listo."]  # JSON fragments were never spoken

    asyncio.run(run())


def test_speak_while_generating_can_be_disabled():
    async def run():
        model = ScriptedStreamingModel([])
        model.complete = lambda request: ModelResponse(output_text="Una. Dos.")
        tts = RecordingStreamingTts()
        session = ChainedVoiceSession(
            agent=Agent(name="dispatcher", instructions="Help.", model=model),
            speech_to_text=FakeStreamingSpeechToText(),
            text_to_speech=tts,
            turn_options=VoiceTurnOptions(speak_while_generating=False),
        )
        _ = [chunk async for chunk in session.process_audio_stream(_one_chunk())]
        assert tts.texts == ["Una. Dos."]

    asyncio.run(run())
