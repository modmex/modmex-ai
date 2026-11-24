"""Pick a model by name instead of by class.

``model_from("anthropic:claude-sonnet-5-5")`` returns the provider's client;
several names separated by commas (``"anthropic:claude-sonnet-5-5,openai:gpt-5.6-terra"``)
return a :class:`FallbackModel` that tries them in order. Keys come from each
provider's usual environment variable, so switching provider is a config change.
"""

from __future__ import annotations

import os
from typing import Any

from modmex_ai.models.base import ModelClient
from modmex_ai.models.fallback import FallbackModel

#: Providers whose key lives in an environment variable. A provider without its key is
#: left out of a fallback list (another one answers); alone, its client reports it on use.
KEYS = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}
PROVIDERS = ("openai", "anthropic", "bedrock")


def parse(spec: str) -> list[tuple[str, str]]:
    """``"provider:model, provider:model"`` → ``[(provider, model), ...]``."""
    pairs = []
    for item in (spec or "").split(","):
        item = item.strip()
        if not item:
            continue
        provider, sep, model = item.partition(":")
        provider, model = provider.strip().lower(), model.strip()
        if not sep or not provider or not model:
            raise ValueError(f"Use provider:model, got {item!r}")
        if provider not in PROVIDERS:
            raise ValueError(f"Unknown provider {provider!r}; use one of {', '.join(PROVIDERS)}")
        pairs.append((provider, model))
    if not pairs:
        raise ValueError("No model given; use provider:model")
    return pairs


def model_from(spec: str, **options: Any) -> ModelClient:
    """The model client for ``spec``; ``options`` (http_client, base_url…) go to each client."""
    pairs = parse(spec)
    if len(pairs) > 1:
        available = [pair for pair in pairs if not KEYS.get(pair[0]) or os.environ.get(KEYS[pair[0]])]
        pairs = available or pairs
    models = [_client(provider, model, options) for provider, model in pairs]
    return models[0] if len(models) == 1 else FallbackModel(models, name=spec)


def _client(provider: str, model: str, options: dict[str, Any]) -> ModelClient:
    if provider == "openai":
        from modmex_ai.providers.openai import OpenAIResponsesModel

        return OpenAIResponsesModel(model, **{"api_key": os.environ.get("OPENAI_API_KEY"), **options})
    if provider == "anthropic":
        from modmex_ai.providers.anthropic import AnthropicMessagesModel

        return AnthropicMessagesModel(model, **options)
    from modmex_ai.providers.bedrock import BedrockConverseModel

    return BedrockConverseModel(model, **options)
