# Live examples

These scripts call real provider APIs and may incur cost.

They are not part of the test suite and are never run by `pytest`.

The OpenAI flow example runs a three-turn pizzeria conversation with the same
session, so it demonstrates continuation state and can make multiple model
calls. It includes typed handoffs, a menu tool, typed order output, and an
output guardrail.

Run them manually with an explicit live flag:

```bash
MODMEX_AI_RUN_LIVE=1 OPENAI_API_KEY=... poetry run python .examples/openai/flow.py
```

Or use a local env file without adding runtime dependencies:

```bash
cp .examples/.env.example .examples/.env
set -a
source .examples/.env
set +a
poetry run python .examples/openai/flow.py
```

`.examples/.env` is ignored by git.

## Anthropic smoke test

`anthropic/smoke.py` validates the Anthropic adapter against the real API in five
checks (plain completion, structured output, a tool loop that replays signed
thinking blocks, streaming, async). It runs at low effort and costs a few cents.

```bash
MODMEX_AI_RUN_LIVE=1 ANTHROPIC_API_KEY=... poetry run python .examples/anthropic/smoke.py
```

`ANTHROPIC_MODEL` overrides the model (default `claude-opus-5-5`).

## Voice: Claude + OpenAI STT/TTS

`voice/claude_openai.py` runs one chained voice turn end to end: a spoken Spanish
question (generated with OpenAI TTS) → OpenAI STT → a Claude agent with a tool →
OpenAI TTS spoken sentence by sentence. It repeats the turn with
`speak_while_generating=False` and prints the time-to-first-audio difference.
Both replies are written to `.examples/output/` (ignored by git) so you can listen.

```bash
MODMEX_AI_RUN_LIVE=1 OPENAI_API_KEY=... ANTHROPIC_API_KEY=... \
    poetry run python .examples/voice/claude_openai.py
```
