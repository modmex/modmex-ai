"""Live voice check: OpenAI STT -> Claude (reasoning + tools) -> OpenAI TTS.

Calls the real OpenAI and Anthropic APIs (a few cents).

Run:
    MODMEX_AI_RUN_LIVE=1 OPENAI_API_KEY=... ANTHROPIC_API_KEY=... \
        poetry run python .examples/voice/claude_openai.py

What it does:
  1. Synthesizes a spoken Spanish question with OpenAI TTS (the "caller").
  2. Runs one chained voice turn: OpenAI STT -> Claude agent with a tool ->
     OpenAI TTS, speaking sentence by sentence while Claude generates.
  3. Runs the same turn with speak_while_generating=False, alternating the two
     modes twice (new, old, new, old) and comparing medians, since Claude's
     replies and network latency vary between runs.
  4. Writes the replies to .examples/output/*.wav and prints the latencies.

The load-lookup tool simulates 0.8 s of backend latency, as a real lookup has;
that is when speaking the model's preamble ("Déjame revisar.") saves silence.
"""

from __future__ import annotations

import asyncio
import os
import statistics
import sys
import time
import wave
from pathlib import Path

from modmex_ai import Agent, ModelSettings, tool
from modmex_ai.providers.anthropic import AnthropicMessagesModel
from modmex_ai.providers.openai import OpenAISpeechProvider, OpenAITranscriptionProvider
from modmex_ai.voice import (
    BufferedStreamingSpeechToText,
    ChainedVoiceSession,
    VoiceSessionEventType,
    VoiceTurnOptions,
)


MODEL = os.getenv("ANTHROPIC_MODEL", "claude-opus-5-5")
OUTPUT = Path(__file__).resolve().parents[1] / "output"
QUESTION = "Hola, ¿me puedes decir en qué estatus va la carga L cien y a qué hora llega?"
PCM_RATE = 24_000  # OpenAI "pcm" speech format: 24 kHz, 16-bit, mono

BACKEND_LATENCY_S = 0.8
lookups: list[str] = []


@tool
async def lookup_load(load_id: str) -> dict:
    """Look up a load's status and ETA by its id (e.g. "L-100")."""
    lookups.append(load_id)
    await asyncio.sleep(BACKEND_LATENCY_S)  # simulated backend call
    return {"load_id": "L-100", "status": "en tránsito", "eta": "hoy 18:30", "city": "Monterrey"}


class RecordingTts(OpenAISpeechProvider):
    """OpenAI TTS that records each synthesized segment."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.segments: list[str] = []

    async def synthesize_stream(self, text, *, context):
        self.segments.append(text)
        async for chunk in super().synthesize_stream(text, context=context):
            yield chunk


def build_agent() -> Agent:
    return Agent(
        name="dispatcher",
        instructions=(
            "Eres un despachador de carga que atiende por teléfono. Responde en español, "
            "en oraciones cortas y naturales para voz, sin listas ni formato. Antes de usar "
            "una herramienta di una frase breve como 'Déjame revisar'. Para cargas usa "
            "lookup_load con el id en formato L-<número>. "
            "Latency-sensitive; begin your visible answer immediately."
        ),
        tools=[lookup_load],
        model=AnthropicMessagesModel(MODEL),
        settings=ModelSettings(reasoning_effort="low", max_tokens=4000),
    )


async def run_turn(caller_audio: bytes, *, speak_while_generating: bool) -> dict:
    lookups.clear()
    tts = RecordingTts(response_format="pcm", voice="marin")
    session = ChainedVoiceSession(
        agent=build_agent(),
        speech_to_text=BufferedStreamingSpeechToText(OpenAITranscriptionProvider(language="es")),
        text_to_speech=tts,
        turn_options=VoiceTurnOptions(speak_while_generating=speak_while_generating),
    )

    async def caller():
        yield caller_audio

    started = time.perf_counter()
    first_chunk_at = None
    audio = bytearray()
    async for chunk in session.process_audio_stream(caller()):
        if first_chunk_at is None:
            first_chunk_at = time.perf_counter()
        audio.extend(chunk)
    total = time.perf_counter() - started
    await session.close()
    events = [event async for event in session.voice_events()]
    heard = next(e.data["transcript"] for e in events if e.type == VoiceSessionEventType.TRANSCRIPT_FINAL)
    said = next(e.data["transcript"] for e in events if e.type == VoiceSessionEventType.ASSISTANT_TRANSCRIPT_FINAL)
    await tts.aclose()
    return {
        "heard": heard,
        "said": said,
        "segments": tts.segments,
        "lookups": list(lookups),
        "audio": bytes(audio),
        "first_audio_s": (first_chunk_at - started) if first_chunk_at else None,
        "total_s": total,
        "metrics": session.last_turn.metrics,
    }


def write_wav(path: Path, pcm: bytes) -> float:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as file:
        file.setnchannels(1)
        file.setsampwidth(2)
        file.setframerate(PCM_RATE)
        file.writeframes(pcm)
    return len(pcm) / (PCM_RATE * 2)


def report(name: str, result: dict, path: Path) -> None:
    seconds = write_wav(path, result["audio"])
    m = result["metrics"]
    print(f"\n[{name}]")
    print(f"  escuchó:        {result['heard']!r}")
    print(f"  respondió:      {result['said']!r}")
    print(f"  oraciones TTS:  {len(result['segments'])} -> {result['segments']}")
    print(f"  herramienta:    {result['lookups']}")
    print(f"  primer audio:   {result['first_audio_s']:.2f}s   total: {result['total_s']:.2f}s")
    print(f"  métricas (ms):  stt={m.stt_latency_ms} agente={m.agent_latency_ms} tts={m.tts_latency_ms} primer_audio={m.first_audio_latency_ms}")
    print(f"  audio:          {seconds:.1f}s -> {path}")


def check(condition: bool, message: str) -> None:
    if not condition:
        print(f"\nFAIL: {message}")
        sys.exit(1)


async def main() -> None:
    if os.getenv("MODMEX_AI_RUN_LIVE") != "1":
        raise SystemExit("Set MODMEX_AI_RUN_LIVE=1 to run this live API example.")
    for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        if not os.getenv(key):
            raise SystemExit(f"Set {key}.")
    print(f"modelo: {MODEL}")
    question_tts = OpenAISpeechProvider(response_format="wav", voice="cedar")
    caller_audio = await question_tts.synthesize(QUESTION, context=None)
    await question_tts.aclose()
    print(f"pregunta del usuario: {QUESTION!r} ({len(caller_audio)} bytes de audio)")

    runs: dict[bool, list[dict]] = {True: [], False: []}
    for round_number in (1, 2):
        for speak in (True, False):
            result = await run_turn(caller_audio, speak_while_generating=speak)
            runs[speak].append(result)
            label = "por oraciones (nuevo)" if speak else "respuesta completa (anterior)"
            suffix = "streamed" if speak else "buffered"
            report(f"{label} #{round_number}", result, OUTPUT / f"voice_{suffix}_{round_number}.wav")

    for result in runs[True] + runs[False]:
        check(result["audio"], "no audio was produced")
        check(any("100" in load for load in result["lookups"]), f"tool not called with L-100: {result['lookups']}")
    check(
        any(len(result["segments"]) >= 2 for result in runs[True]),
        f"expected several TTS segments, got {[r['segments'] for r in runs[True]]}",
    )
    new = statistics.median(r["first_audio_s"] for r in runs[True])
    old = statistics.median(r["first_audio_s"] for r in runs[False])
    print(f"\nprimer audio (mediana de 2): nuevo {new:.2f}s vs anterior {old:.2f}s -> {old - new:+.2f}s")
    print(f"(la herramienta simula {BACKEND_LATENCY_S}s de latencia de backend)")
    print("all checks passed")


if __name__ == "__main__":
    asyncio.run(main())
