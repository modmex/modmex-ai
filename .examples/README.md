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
