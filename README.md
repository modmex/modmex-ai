# modmex-ai

Lightweight, provider-neutral agents, flows, tools, and voice runtimes for Python.

[![CI](https://img.shields.io/github/actions/workflow/status/modmex/modmex-ai/ci.yml?branch=main&logo=github&label=CI)](https://github.com/modmex/modmex-ai/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/codecov/c/github/modmex/modmex-ai?label=coverage)](https://codecov.io/gh/modmex/modmex-ai)
[![PyPI](https://img.shields.io/pypi/v/modmex-ai.svg)](https://pypi.org/project/modmex-ai/)
[![Python Versions](https://img.shields.io/pypi/pyversions/modmex-ai.svg)](https://pypi.org/project/modmex-ai/)
[![License](https://img.shields.io/github/license/modmex/modmex-ai.svg)](https://github.com/modmex/modmex-ai/blob/main/LICENSE)

`modmex-ai` is a small runtime for applications that need an agent to reason,
call tools, hand off work, and return a controlled result. It fits equally well
in HTTP APIs, CLIs, background workers, WebSocket servers, notebooks, desktop
applications, voice systems, and event-driven/serverless workloads.

Its core has no provider SDK dependency. OpenAI is the first complete adapter;
the core contracts remain reusable for another text, speech, or realtime
provider.

The basic execution model is intentionally simple:

```text
input -> Flow -> one or more Agents -> output, tools, handoffs, and events
```

## What it provides

- `Agent` for instructions, structured output, tools, guardrails, and streaming.
- `Flow` for multi-agent routing, handoffs, continuation, and sync/async execution.
- Sessions, snapshots, optimistic concurrency contracts, and approval checkpoints.
- Realtime and chained voice contracts that keep STT, reasoning, and TTS composable.
- Provider-neutral tracing, usage, evals, and host-owned observability.

Install the core package in any Python application:

```bash
pip install modmex-ai
```

Add only the optional transport needed by a deployment:

```bash
pip install 'modmex-ai[async]'      # native async HTTP
pip install 'modmex-ai[realtime]'  # WebSocket-based Realtime adapters
```

## Quick start

```python
from modmex import BaseModel
from modmex_ai import Agent, FakeModel, Flow


class TriageOutput(BaseModel):
    intent: str
    confidence: float


triage = Agent(
    name="triage",
    instructions="Classify the incoming message.",
    output_type=TriageOutput,
    handoffs=["support"],
)

flow = Flow(
    name="analyze-message",
    entrypoint=triage,
    agents=[],
    model=FakeModel(['{"intent":"support","confidence":0.91}']),
)

result = flow.run("I need help")
result.output.model_dump()
```

The same `Flow` can be called by a REST handler, a queue consumer, a CLI
command, or a long-lived WebSocket connection. Event-driven systems are a
natural fit, but they are not required.

Multi-agent delegation uses handoff function tools, not a field in the model
output. A handoff defaults to transfer_to_agent_name; its typed input is
validated and delivered to an on_handoff callback before the next agent
continues with the conversation.

## Multimodal inputs

Messages can combine typed text, document, and image parts without exposing a
provider payload in application code. `FileInput` and `ImageInput` accept a
URL, a provider file id, or inline bytes/Base64 as appropriate. The provider
adapter decides which source it supports.

The OpenAI Responses adapter maps these parts to the official `input_text`,
`input_file`, and `input_image` request items. For documents stored privately,
prefer a short-lived signed URL and create it only for the model request; do
not persist it in conversation state or publish it in events.

```python
from modmex_ai import FileInput, InputDetail, Message, TextInput
from modmex_ai.models import ModelRequest
from modmex_ai.providers.openai import OpenAIResponsesModel

request = ModelRequest(
    messages=[
        Message(
            role="user",
            content=[
                TextInput(text="Extract the organization name and expiration date."),
                FileInput(
                    url=presigned_file_url,
                    filename="agreement.pdf",
                    media_type="application/pdf",
                    detail=InputDetail.HIGH,
                ),
            ],
        )
    ]
)

response = OpenAIResponsesModel("gpt-5.6", api_key="...").complete(request)
print(response.output_text)
```

`OpenAIChatModel` deliberately rejects `FileInput` and `ImageInput` until that
adapter implements their distinct wire format. Use `OpenAIResponsesModel` for
multimodal input.

## Durable sessions and approvals

The core provides portable `SessionSnapshot` and `PersistedFlowState` models,
plus optimistic-concurrency store contracts. A host chooses the database and
owns the lifecycle; this keeps the framework usable in a web application, a
worker, or serverless infrastructure without importing a storage SDK into the
core.

Tools can require an externally verified approval. When a protected tool is
requested, a Flow raises `FlowSuspended` with the exact pending tool call and
a serializable checkpoint. Persist it, collect the signed approval in the
host, then resume the Flow without asking the model to decide again.

For Lambda-oriented hosts, install the optional connector integration and use
the DynamoDB stores from `modmex-ai`:

```bash
pip install 'modmex-ai[lambda]'
```

```python
from modmex_ai.persistence import DynamoDbDurableSessionStore
from modmex_lambda.connectors.dynamodb import Connector

store = DynamoDbDurableSessionStore(Connector("sessions-table"))
```

Other applications can implement the same `DurableSessionStore` and
`FlowStateStore` contracts with Postgres, Redis, or their existing storage.

## Realtime voice sessions

`VoiceAgentSession` is the common contract for voice sessions. There are two
implementations with distinct responsibilities:

- `OpenAIRealtimeSession` para speech-to-speech por WebSocket/SIP.
- `ChainedVoiceSession` para conservar el control explícito de
  STT → `Agent`/`Flow` → TTS.

Ambas exponen eventos normalizados, uso acumulado, herramientas y handoffs. El
adaptador de OpenAI conserva el payload específico del proveedor únicamente en
su frontera; el core usa `tool_call_id` para no confundir llamadas de tools con
identificadores de llamadas telefónicas.

Install the optional WebSocket transport only in workloads that need it:

```bash
pip install 'modmex-ai[realtime]'
```

The OpenAI adapter reuses `Agent` tools, handoffs, tool guardrails, context,
usage, and traces. It does not depend on the OpenAI Agents SDK. For a SIP call
that was already accepted by your server, connect the worker using its
`realtime_call_id`:

```python
import asyncio

from modmex_ai import Agent
from modmex_ai.providers.openai import (
    OpenAIRealtimeClient,
    OpenAIRealtimeSessionConfig,
)


async def run_voice_call(realtime_call_id: str) -> None:
    agent = Agent(
        name="support",
        instructions="Help the caller concisely and use tools when needed.",
    )
    client = OpenAIRealtimeClient()
    session = await client.connect(
        agent=agent,
        realtime_call_id=realtime_call_id,
        config=OpenAIRealtimeSessionConfig(voice="marin"),
    )
    async with session:
        await session.configure()
        await session.create_response(
            instructions="Greet the caller and ask how you can help."
        )
        async for event in session.events():
            if event.type == "response.done":
                print(session.usage.model_dump())


asyncio.run(run_voice_call("rtc_u1_example"))
```

When the model emits a function call, the session validates and executes the
registered Modmex-AI tool, sends its output to the socket, and asks Realtime to
continue. A `transfer_to_*` handoff updates the active agent's instructions and
tool surface in the same Realtime session.

For a SIP accept endpoint, build the accept body from the exact same session
configuration used after the WebSocket connects. That prevents model, voice,
instructions and tool definitions from drifting between both phases:

```python
accept_payload = OpenAIRealtimeSessionConfig(voice="marin").to_accept_payload(agent)
```

## Chained voice pipeline

Use a chained session when the application must choose its transcription and
speech providers, or needs text-agent behavior between audio turns. A `Flow`
is preferred when the conversation can hand off to another agent.

```python
session = ChainedVoiceSession(
    flow=conversation_flow,
    speech_to_text=transcriber,
    text_to_speech=synthesizer,
    context={"workspace_id": "workspace-1"},
)

audio_reply = await session.process_audio(caller_audio)
async for event in session.voice_events():
    # transcript_final, tool_finished, handoff_completed,
    # response_completed, session_ended, or error
    handle_voice_event(event)
```

The STT/TTS providers are small async protocols, so concrete provider SDKs stay
outside the core package. Close either session with a `VoiceTerminationReason`
to make lifecycle reporting explicit.

For bounded audio turns, the OpenAI adapters use the Audio API directly through
the lightweight HTTP client—no provider SDK is installed:

```python
from modmex_ai.providers.openai import (
    OpenAISpeechProvider,
    OpenAITranscriptionProvider,
)

session = ChainedVoiceSession(
    flow=conversation_flow,
    speech_to_text=OpenAITranscriptionProvider(model="gpt-4o-mini-transcribe"),
    text_to_speech=OpenAISpeechProvider(voice="marin", response_format="wav"),
)
```

To continue a chained conversation in another process, persist the host's
`Session` implementation and `session.continuation`. The continuation tracks
the active agent and provider state; the host owns durable history storage.
For a small self-contained host, `SessionSnapshot.from_session(session)` can
serialize an `InMemorySession` and later restore it with
`snapshot.to_memory_session()`.

`ChainedVoiceSession.process_audio_stream()` supports providers that implement
streaming STT and TTS contracts. It emits transcript deltas, waits for a final
transcript before invoking the text flow, then yields playable audio chunks.
The callable adapters make it possible to plug in a provider-specific function
without making that provider a dependency of `modmex-ai`.

## Native async models

Models can optionally expose `async def acomplete(request)`. `Agent.run_async`
and `Flow.run_async` use it directly, including async tools; existing sync
models retain their compatible worker-thread fallback. The built-in OpenAI
clients expose `acomplete()` as well. Install the optional native transport to
make them select `httpx` automatically:

```bash
pip install 'modmex-ai[async]'
```

Without that extra, the compatible stdlib transport runs in a worker and keeps
the event loop responsive.

## Realtime transcription

For live microphone or telephony media, use the transcription-only WebSocket
adapter instead of a speech-to-speech agent session. It emits `Transcription`
deltas as audio arrives and one final value after manual commit:

```python
from modmex_ai.providers.openai import OpenAIRealtimeTranscriptionProvider

transcriber = OpenAIRealtimeTranscriptionProvider()
async for transcript in transcriber.transcribe_stream(pcm_chunks, context=context):
    render(transcript.text, final=transcript.is_final)
```

The adapter uses 24 kHz PCM by default and requires `modmex-ai[realtime]`.
For completed audio turns, `OpenAITranscriptionProvider` remains the simpler
request-based alternative.

For a long-lived call, open one reusable session and commit each user turn:

```python
session = await transcriber.connect()
await session.append_audio(pcm_chunk)
await session.commit_turn()
async for transcript in session.events():
    correlate(transcript.item_id, transcript.text, transcript.is_final)
```

## Turn cancellation and provider retries

`ChainedVoiceSession` accepts `VoiceTurnOptions(max_provider_retries=...)`.
Retries apply only to STT/TTS provider I/O, never to the text flow, tools, or
handoffs. Call `cancel_current_turn()` to cooperatively stop a streaming turn
before its next provider chunk; hosts can still cancel the owning task when an
immediate interruption of a blocked transport is required.

After every completed, cancelled, timed-out, or failed bounded turn,
`ChainedVoiceSession.last_turn` contains a `VoiceTurnOutcome`. Its metrics
separate STT, agent, TTS, total latency, and provider retry count.
Pass a `VoiceTurnObserver` to export those outcomes to the host's own logs or
metrics backend without adding an observability dependency to this package.

## Roadmap

Model Context Protocol (MCP) is not supported yet. MCP client and tool-server
integration are planned for a future release; today, applications can register
local tools directly through `Agent`.
