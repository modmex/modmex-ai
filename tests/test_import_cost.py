"""Guard serverless cold starts: a sync Agent handler must not load unused stacks."""

import subprocess
import sys

import pytest


HEAVY = ["asyncio", "modmex_ai.voice", "modmex_ai.realtime", "modmex_ai.flows", "modmex_ai.evals", "modmex_ai.context"]


@pytest.mark.parametrize(
    "imports",
    [
        "from modmex_ai import Agent, tool, ModelSettings",
        "from modmex_ai import Agent; from modmex_ai.providers.anthropic import AnthropicMessagesModel",
        "from modmex_ai import Agent; from modmex_ai.providers.openai import OpenAIResponsesModel, OpenAIChatModel",
    ],
)
def test_sync_agent_imports_stay_light(imports):
    code = f"import sys; {imports}; print([m for m in {HEAVY!r} if m in sys.modules])"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]", out.stdout


def test_lazy_exports_resolve():
    import modmex_ai
    import modmex_ai.providers.openai as openai

    assert all(getattr(modmex_ai, name) is not None for name in modmex_ai.__all__)
    assert all(getattr(openai, name) is not None for name in openai.__all__)
    with pytest.raises(AttributeError):
        modmex_ai.DoesNotExist
